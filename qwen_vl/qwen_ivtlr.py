import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import logging
logging.basicConfig(
    filename='qwenvl_32_infer_sqa_time_epoch4.log',
    level=logging.DEBUG,         
    format='[%(asctime)s] %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'  
)
import pdb
from transformers.cache_utils import DynamicCache

Outputs = namedtuple(
    "Outputs",
    [
        "loss",
        "ce_loss",
        "nvt_loss",
        "qvr_loss",
        "causal_loss",
        "inputs_embeds",
        "logits",
        "latent_attn_trace",
        "token_norms",
    ],
)
MAX_N_LATENT = 4 


class IVTLR(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        image_token_id,
        visual_start_id,
        visual_end_id,
        num_selected_patches: int = 32,
        patch_reuse_policy: str = "never",
        patch_sampling_strategy: str = "attention_topk",
        processor_model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        enable_nvt_loss: bool = False,
        nvt_loss_weight: float = 0.0,
        nvt_loss_epsilon: float = 1e-8,
        enable_qvr_loss: bool = False,
        qvr_loss_weight: float = 0.0,
        qvr_loss_epsilon: float = 1e-8,
        qvr_num_layers: int = 4,
        enable_causal_loss: bool = False,
        causal_loss_weight: float = 0.0,
        causal_loss_epsilon: float = 1e-8,
    ):

        super(IVTLR, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id
        self.num_selected_patches = num_selected_patches
        valid_policies = {"never", "next_step_only", "always"}
        if patch_reuse_policy not in valid_policies:
            raise ValueError(f"Invalid patch_reuse_policy={patch_reuse_policy}. Expected one of {valid_policies}.")
        self.patch_reuse_policy = patch_reuse_policy
        valid_sampling_strategies = {"attention_topk", "random_image_only", "all_image_patches"}
        if patch_sampling_strategy not in valid_sampling_strategies:
            raise ValueError(
                f"Invalid patch_sampling_strategy={patch_sampling_strategy}. "
                f"Expected one of {valid_sampling_strategies}."
            )
        self.patch_sampling_strategy = patch_sampling_strategy
        self.enable_nvt_loss = enable_nvt_loss and nvt_loss_weight > 0
        self.nvt_loss_weight = nvt_loss_weight
        self.nvt_loss_epsilon = float(nvt_loss_epsilon) if nvt_loss_epsilon is not None else 1e-8
        self.enable_qvr_loss = enable_qvr_loss and qvr_loss_weight > 0
        self.qvr_loss_weight = qvr_loss_weight
        self.qvr_loss_epsilon = float(qvr_loss_epsilon) if qvr_loss_epsilon is not None else 1e-8
        self.qvr_num_layers = max(int(qvr_num_layers), 1)
        self.enable_causal_loss = enable_causal_loss and causal_loss_weight > 0
        self.causal_loss_weight = causal_loss_weight
        self.causal_loss_epsilon = float(causal_loss_epsilon) if causal_loss_epsilon is not None else 1e-8

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
        
        # self.processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
        self.processor = AutoProcessor.from_pretrained(processor_model_id)

    def _compute_nvt_loss(self, attentions, query_index, inserted_spans):
        if not inserted_spans:
            return None

        per_batch_losses = []
        for batch_index, span in enumerate(inserted_spans):
            if span is None:
                continue

            span_start, span_end = span
            if span_end <= span_start:
                continue

            layer_masses = []
            for layer_attn in attentions:
                if query_index >= layer_attn.size(-2) or span_end > layer_attn.size(-1):
                    continue
                token_to_span = layer_attn[batch_index, :, query_index, span_start:span_end].sum(dim=-1)
                layer_masses.append(token_to_span.mean())

            if not layer_masses:
                continue

            mt = torch.stack(layer_masses).mean()
            per_batch_losses.append(-torch.log(mt + self.nvt_loss_epsilon))

        if not per_batch_losses:
            return None

        return torch.stack(per_batch_losses).mean()

    def _compute_qvr_loss(self, attentions, query_positions, inserted_spans, question_mask):
        if not inserted_spans:
            return None

        avg_attn = self._average_last_attentions(attentions, self.qvr_num_layers)
        if avg_attn is None:
            return None

        attn = avg_attn.mean(dim=1)
        seq_len = attn.size(-1)

        per_batch_losses = []
        for batch_index, span in enumerate(inserted_spans):
            if span is None:
                continue

            query_index = query_positions[batch_index]
            if query_index is None or query_index < 0 or query_index >= seq_len:
                continue

            span_start, span_end = span
            if span_end <= span_start:
                continue

            vis_mass = attn[batch_index, query_index, span_start:span_end].sum()
            ques_mass = attn[batch_index, query_index, question_mask[batch_index]].sum()
            h_val = (2.0 * vis_mass * ques_mass) / (vis_mass + ques_mass + self.qvr_loss_epsilon)
            per_batch_losses.append(-torch.log(h_val + self.qvr_loss_epsilon))

        if not per_batch_losses:
            return None

        return torch.stack(per_batch_losses).mean()

    @staticmethod
    def _summarize_norms(norms_tensor: torch.Tensor):
        if norms_tensor.numel() == 0:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        norms_tensor = norms_tensor.float()
        return {
            "count": int(norms_tensor.numel()),
            "mean": float(norms_tensor.mean().item()),
            "std": float(norms_tensor.std(unbiased=False).item()),
            "min": float(norms_tensor.min().item()),
            "max": float(norms_tensor.max().item()),
        }

    @staticmethod
    def _average_last_attentions(attentions, num_layers):
        if not attentions:
            return None
        n_layers = min(num_layers, len(attentions))
        stacked = torch.stack(attentions[-n_layers:], dim=0)
        return stacked.mean(dim=0)

    @staticmethod
    def _compute_answer_logprob(logits, labels, answer_mask):
        seq_len = min(logits.size(1), labels.size(1), answer_mask.size(1))
        logits = logits[..., -seq_len:, :]
        labels = labels[..., -seq_len:]
        answer_mask = answer_mask[..., -seq_len:]

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        shift_answer_mask = answer_mask[..., 1:].bool()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        valid = shift_answer_mask & (shift_labels != -100)
        gathered = gathered.masked_fill(~valid, 0.0)
        token_counts = valid.sum(dim=-1)
        token_counts_clamped = token_counts.clamp(min=1)
        mean_logprob = gathered.sum(dim=-1) / token_counts_clamped
        return mean_logprob, token_counts

    def forward(
        self,
        input_ids: torch.LongTensor,        # shape = (B, S)
        attention_mask: torch.LongTensor,    # shape = (B, S)
        labels: torch.LongTensor,            # shape = (B, S)
        position_ids: torch.LongTensor,      # shape = (B, S)
        pixel_values: torch.FloatTensor,     # shape = (B, 3, H, W)
        image_grid_thw: torch.Tensor = None,
        answer_mask: torch.LongTensor = None,
        return_latent_attn: bool = False,
        return_token_norms: bool = False,
        latent_attn_threshold: float = None,
        latent_attn_threshold_multiplier: float = 5.0,
        **kwargs
    ):

        B, S = input_ids.size()

        # decode
        _ = self.processor.tokenizer.batch_decode(
            input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
        )

        inputs_embeds = self.embedding(input_ids)  # (B, S, D)

        original_mask = torch.ones((B, S), dtype=torch.bool, device=input_ids.device)

        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True)
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype())
            image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_embeds.shape[0]}"
                )
            image_mask_init = (input_ids == self.image_token_id)  # (B, orig_S)
            expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds)
        else:
            image_mask_init = torch.zeros((B, S), dtype=torch.bool, device=input_ids.device)
        

        max_len = 3000
        image_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
        image_mask[:, :S] = image_mask_init
        trace_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
        recently_selected_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
        patch_insert_mask = None
        reasoning_insert_mask = None
        need_insert_origin_masks = return_token_norms or self.enable_qvr_loss or self.enable_causal_loss
        if need_insert_origin_masks:
            patch_insert_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)
            reasoning_insert_mask = torch.zeros((B, max_len), dtype=torch.bool, device=input_ids.device)


        for b in range(B):
            vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
            image_mask[b, vs+1:ve] = True

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == b]
            for b in range(B)
        ]
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        if max_n_latents > 0:
            first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
            end = first_latent_pos
        else:
            end = S
        
        kv_cache = None
        all_logits = []
        nvt_losses = []
        qvr_losses = []
        prev_inserted_spans = None

        if max_n_latents > 0:
            for pass_idx in range(max_n_latents):
                start = 0
                hidden_states_offset = 0
                if kv_cache is None:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],  # (B, end, D)
                        attention_mask=attention_mask[:, start:end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, start:end, :],
                        attention_mask=attention_mask[:, :end],
                        position_ids=position_ids[:, start:end],
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        output_attentions=True,
                        use_cache=True,
                    )

                logits_this = outputs.logits                   
                hidden_states = outputs.hidden_states[-1]      
                attentions    = outputs.attentions              # list of (B, heads, seq_len, seq_len)
                kv_cache      = outputs.past_key_values

                all_logits.append(logits_this)

                if self.enable_nvt_loss and prev_inserted_spans is not None:
                    nvt_loss = self._compute_nvt_loss(attentions, end - 1, prev_inserted_spans)
                    if nvt_loss is not None:
                        nvt_losses.append(nvt_loss)

                if self.enable_qvr_loss and prev_inserted_spans is not None:
                    if B > 0:
                        seq_len_this_pass = attention_mask[:, :end].size(1)
                        positions_this_pass = torch.arange(seq_len_this_pass, device=input_ids.device).unsqueeze(0).expand(B, -1)
                        first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0) if any(len(lst) > 0 for lst in latent_lists) else None
                        if first_latent_pos is not None:
                            question_mask = (
                                original_mask[:, :end]
                                & (~image_mask[:, :end])
                                & (positions_this_pass < first_latent_pos)
                            )
                            query_positions = [
                                (lst[pass_idx] if pass_idx < len(lst) else (lst[-1] if lst else None))
                                for lst in latent_lists
                            ]
                            qvr_loss = self._compute_qvr_loss(
                                attentions,
                                query_positions,
                                prev_inserted_spans,
                                question_mask,
                            )
                            if qvr_loss is not None:
                                qvr_losses.append(qvr_loss)

                #   Top-K
                avg_attn = torch.cat(attentions, dim=1).mean(dim=1)  # (B, seq_len)
                current_seq_len = avg_attn.size(1)
                select_image_embeds = []
                current_selected_mask = torch.zeros_like(image_mask)
                selected_counts = []
                selected_patch_origins = [] if need_insert_origin_masks else None
                selected_reasoning_origins = [] if need_insert_origin_masks else None

                for b in range(B):
                    last_attn = avg_attn[b, end - 1]  # shape=(seq_len,)
                    vs, ve = vs_pos_per_batch[b], ve_pos_per_batch[b]
                    scores = last_attn.clone()

                    image_allowed_positions = image_mask[b, :current_seq_len]
                    trace_allowed_positions = trace_mask[b, :current_seq_len]
                    if self.patch_reuse_policy == "next_step_only":
                        not_recent = ~recently_selected_mask[b, :current_seq_len]
                        image_allowed_positions = image_allowed_positions & not_recent
                        trace_allowed_positions = trace_allowed_positions & not_recent

                    if self.patch_sampling_strategy == "all_image_patches":
                        image_abs_idxs = torch.arange(vs + 1, ve, device=input_ids.device)
                        if image_abs_idxs.numel() == 0:
                            raise ValueError("No image patch positions available for all_image_patches strategy.")
                        abs_idxs = image_abs_idxs
                        trace_abs_idxs = torch.empty(0, dtype=torch.long, device=input_ids.device)
                    elif self.patch_sampling_strategy == "random_image_only":
                        image_pool_mask = image_allowed_positions.clone()
                        image_pool_mask[:vs + 1] = False
                        image_pool_mask[ve:] = False
                        image_candidates = torch.nonzero(image_pool_mask, as_tuple=False).squeeze(-1)

                        if image_candidates.numel() >= self.num_selected_patches:
                            rand_order = torch.randperm(image_candidates.numel(), device=input_ids.device)
                            abs_idxs = image_candidates[rand_order[:self.num_selected_patches]]
                        elif image_candidates.numel() > 0:
                            n_to_fill = self.num_selected_patches - image_candidates.numel()
                            fill_idxs = image_candidates[torch.randint(
                                low=0,
                                high=image_candidates.numel(),
                                size=(n_to_fill,),
                                device=input_ids.device,
                            )]
                            abs_idxs = torch.cat([image_candidates, fill_idxs], dim=0)
                        else:
                            image_span = torch.arange(vs + 1, ve, device=input_ids.device)
                            if image_span.numel() == 0:
                                raise ValueError("No image patch positions available for random_image_only sampling.")
                            fill_rand = torch.randint(
                                low=0,
                                high=image_span.numel(),
                                size=(self.num_selected_patches,),
                                device=input_ids.device,
                            )
                            abs_idxs = image_span[fill_rand]

                        image_abs_idxs = abs_idxs
                        trace_abs_idxs = torch.empty(0, dtype=torch.long, device=input_ids.device)
                    else:
                        if pass_idx == 0:
                            image_quota = self.num_selected_patches
                            trace_quota = 0
                        else:
                            trace_quota = self.num_selected_patches // 2
                            image_quota = self.num_selected_patches - trace_quota

                        image_scores = scores.clone()
                        image_invalid = ~image_allowed_positions
                        image_scores[image_invalid] = float("-inf")
                        image_rel_scores = image_scores[vs + 1 : ve]
                        n_image_candidates = int(image_allowed_positions[vs + 1 : ve].sum().item())
                        image_take = min(image_quota, n_image_candidates)
                        if image_take > 0:
                            topk_image_rel = image_rel_scores.topk(image_take, sorted=False)[1]
                            image_abs_idxs = (vs + 1) + topk_image_rel
                        else:
                            image_abs_idxs = torch.empty(0, dtype=torch.long, device=input_ids.device)

                        trace_scores = scores.clone()
                        trace_invalid = ~trace_allowed_positions
                        trace_scores[trace_invalid] = float("-inf")
                        n_trace_candidates = int(trace_allowed_positions.sum().item())
                        trace_take = min(trace_quota, n_trace_candidates)
                        if trace_take > 0:
                            trace_abs_idxs = trace_scores.topk(trace_take, sorted=False)[1]
                        else:
                            trace_abs_idxs = torch.empty(0, dtype=torch.long, device=input_ids.device)

                        abs_idxs = torch.cat([image_abs_idxs, trace_abs_idxs], dim=0)

                        if abs_idxs.numel() < self.num_selected_patches:
                            combined_allowed = image_allowed_positions | trace_allowed_positions
                            if abs_idxs.numel() > 0:
                                combined_allowed[abs_idxs] = False
                            combined_scores = scores.clone()
                            combined_scores[~combined_allowed] = float("-inf")
                            n_extra_candidates = int(combined_allowed.sum().item())
                            n_to_fill = min(self.num_selected_patches - abs_idxs.numel(), n_extra_candidates)
                            if n_to_fill > 0:
                                extra_abs_idxs = combined_scores.topk(n_to_fill, sorted=False)[1]
                                abs_idxs = torch.cat([abs_idxs, extra_abs_idxs], dim=0)

                        if abs_idxs.numel() < self.num_selected_patches:
                            n_to_fill = self.num_selected_patches - abs_idxs.numel()
                            if abs_idxs.numel() > 0:
                                # Keep selection pool restricted to image/trace by padding from selected indices.
                                repeat_count = (n_to_fill + abs_idxs.numel() - 1) // abs_idxs.numel()
                                pad_abs_idxs = abs_idxs.repeat(repeat_count)[:n_to_fill]
                                abs_idxs = torch.cat([abs_idxs, pad_abs_idxs], dim=0)
                            else:
                                # Safety fallback: only sample from original image span, never generic context tokens.
                                image_span_scores = scores.clone()
                                allowed_image_span = torch.zeros_like(image_span_scores, dtype=torch.bool)
                                allowed_image_span[vs + 1 : ve] = True
                                image_span_scores[~allowed_image_span] = float("-inf")
                                abs_idxs = image_span_scores.topk(self.num_selected_patches, sorted=False)[1]

                    logging.debug(f"selected image idx: {image_abs_idxs}")
                    logging.debug(f"selected trace idx: {trace_abs_idxs}")
                    logging.debug(f"abs idx: {abs_idxs}")
                    if need_insert_origin_masks:
                        selected_patch_origins.append(image_mask[b, abs_idxs].clone())
                        selected_reasoning_origins.append(trace_mask[b, abs_idxs].clone())
                    if self.patch_reuse_policy == "never":
                        image_mask[b, abs_idxs] = False
                        trace_mask[b, abs_idxs] = False
                    elif self.patch_reuse_policy == "next_step_only":
                        current_selected_mask[b, abs_idxs] = True

                    picked = inputs_embeds[b, abs_idxs, :]  # (K, D)
                    select_image_embeds.append(picked)
                    selected_counts.append(abs_idxs.numel())

                select_image_embeds = torch.stack(select_image_embeds, dim=0)  # (B, K, D)
                inputs_embeds_detached = inputs_embeds.detach().clone()
                for b in range(B):
                    if len(latent_lists[b]) > pass_idx:
                        t_idx = latent_lists[b][pass_idx]
                        rel_pos = t_idx - 1 - hidden_states_offset
                        rel_pos = max(0, min(rel_pos, hidden_states.size(1) - 1))
                        inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]

                inputs_embeds.data = inputs_embeds_detached
                new_inputs_embeds = []
                new_attention_mask = []
                new_position_ids = []
                new_original_mask = []
                new_image_mask = []
                new_trace_mask = []
                new_recently_selected_mask = []
                new_patch_insert_mask = [] if need_insert_origin_masks else None
                new_reasoning_insert_mask = [] if need_insert_origin_masks else None
                current_inserted_spans = []
                batch_max_len = 0

                for b in range(B):
                    K_b = selected_counts[b]
                    end_b = end
                    prefix_b = inputs_embeds[b, :end_b, :]    # (end_b, D)
                    suffix_b = inputs_embeds[b, end_b:, :]    # (old_len - end_b, D)
                    v_embed_b = select_image_embeds[b]       # (K, D)
                    merged_b = torch.cat([prefix_b, v_embed_b, suffix_b], dim=0)  # (old_len+K, D)
                    new_inputs_embeds.append(merged_b)
                    current_inserted_spans.append((end_b, end_b + K_b))

                    # attention_mask
                    att_pref = attention_mask[b, :end_b]      # (end_b,)
                    att_suf  = attention_mask[b, end_b:]      # (old_len-end_b,)
                    att_v    = torch.ones(K_b, device=attention_mask.device, dtype=attention_mask.dtype)
                    merged_att = torch.cat([att_pref, att_v, att_suf], dim=0)  # (new_len,)
                    new_attention_mask.append(merged_att)

                    # position_ids 
                    new_pos = torch.arange(merged_b.size(0), device=position_ids.device)
                    new_position_ids.append(new_pos)

                    # original_mask
                    orig_pref = original_mask[b, :end_b]       # (end_b,)
                    orig_suf  = original_mask[b, end_b:]       # (old_len-end_b,)
                    orig_v    = torch.zeros(K_b, device=input_ids.device, dtype=torch.bool)
                    merged_orig = torch.cat([orig_pref, orig_v, orig_suf], dim=0)
                    new_original_mask.append(merged_orig)

                    # image_mask
                    img_pref = image_mask[b, :end_b]
                    img_suf  = image_mask[b, end_b:]
                    img_v    = torch.zeros(K_b, device=input_ids.device, dtype=torch.bool)
                    merged_img = torch.cat([img_pref, img_v, img_suf], dim=0)
                    new_image_mask.append(merged_img)

                    # trace_mask
                    trace_pref = trace_mask[b, :end_b]
                    trace_suf  = trace_mask[b, end_b:]
                    trace_v    = torch.ones(K_b, device=input_ids.device, dtype=torch.bool)
                    merged_trace = torch.cat([trace_pref, trace_v, trace_suf], dim=0)
                    new_trace_mask.append(merged_trace)

                    # recently_selected_mask (for next_step_only)
                    if self.patch_reuse_policy == "next_step_only":
                        recent_pref = current_selected_mask[b, :end_b]
                        recent_suf  = current_selected_mask[b, end_b:]
                        recent_v    = torch.zeros(K_b, device=input_ids.device, dtype=torch.bool)
                        merged_recent = torch.cat([recent_pref, recent_v, recent_suf], dim=0)
                        new_recently_selected_mask.append(merged_recent)

                    if need_insert_origin_masks:
                        patch_pref = patch_insert_mask[b, :end_b]
                        patch_suf = patch_insert_mask[b, end_b:]
                        patch_v = selected_patch_origins[b].to(torch.bool)
                        merged_patch = torch.cat([patch_pref, patch_v, patch_suf], dim=0)
                        new_patch_insert_mask.append(merged_patch)

                        reasoning_pref = reasoning_insert_mask[b, :end_b]
                        reasoning_suf = reasoning_insert_mask[b, end_b:]
                        reasoning_v = selected_reasoning_origins[b].to(torch.bool)
                        merged_reasoning = torch.cat([reasoning_pref, reasoning_v, reasoning_suf], dim=0)
                        new_reasoning_insert_mask.append(merged_reasoning)

                    batch_max_len = max(batch_max_len, merged_b.size(0))

                padded_embeds = []
                padded_att   = []
                padded_pos   = []
                padded_orig  = []
                padded_img   = []
                padded_trace = []
                padded_recent = []
                padded_patch = []
                padded_reasoning = []

                for b in range(B):
                    emb_b = new_inputs_embeds[b]
                    att_b = new_attention_mask[b]
                    pos_b = new_position_ids[b]
                    orig_b = new_original_mask[b]
                    img_b = new_image_mask[b]
                    trace_b = new_trace_mask[b]

                    padded_embeds.append(emb_b.unsqueeze(0))
                    padded_att.append(att_b.unsqueeze(0))
                    padded_pos.append(pos_b.unsqueeze(0))
                    padded_orig.append(orig_b.unsqueeze(0))
                    padded_img.append(img_b.unsqueeze(0))
                    padded_trace.append(trace_b.unsqueeze(0))
                    if self.patch_reuse_policy == "next_step_only":
                        recent_b = new_recently_selected_mask[b]
                        padded_recent.append(recent_b.unsqueeze(0))
                    if need_insert_origin_masks:
                        patch_b = new_patch_insert_mask[b]
                        reasoning_b = new_reasoning_insert_mask[b]
                        padded_patch.append(patch_b.unsqueeze(0))
                        padded_reasoning.append(reasoning_b.unsqueeze(0))

                inputs_embeds = torch.cat(padded_embeds, dim=0)    
                attention_mask = torch.cat(padded_att, dim=0)      
                position_ids    = torch.cat(padded_pos, dim=0)     
                original_mask  = torch.cat(padded_orig, dim=0)
                image_mask     = torch.cat(padded_img, dim=0)   # (B, new_S)
                trace_mask     = torch.cat(padded_trace, dim=0)
                if self.patch_reuse_policy == "next_step_only":
                    recently_selected_mask = torch.cat(padded_recent, dim=0)
                if need_insert_origin_masks:
                    patch_insert_mask = torch.cat(padded_patch, dim=0)
                    reasoning_insert_mask = torch.cat(padded_reasoning, dim=0)
                prev_inserted_spans = current_inserted_spans
                for b in range(B):
                    K_b = selected_counts[b]
                    for i, pos in enumerate(latent_lists[b]):
                        if pos > end:
                            latent_lists[b][i] = pos + K_b
                            logging.debug(f"latent pos: {latent_lists[b][i]}")

                if pass_idx + 1 >= max_n_latents:
                    end = inputs_embeds.size(1)
                else:
                    if B != 1 and self.patch_sampling_strategy == "all_image_patches":
                        raise ValueError("all_image_patches currently supports batch_size=1 only.")
                    end = end + 1 + selected_counts[0]

            output_attentions = self.enable_nvt_loss or return_latent_attn or self.enable_qvr_loss
            if kv_cache:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :end, :],
                    attention_mask=attention_mask[:, :end],
                    position_ids=position_ids[:, :end],
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
            all_logits.append(outputs.logits)
            if self.enable_nvt_loss and prev_inserted_spans is not None and outputs.attentions is not None:
                nvt_loss = self._compute_nvt_loss(outputs.attentions, end - 1, prev_inserted_spans)
                if nvt_loss is not None:
                    nvt_losses.append(nvt_loss)
        else:
            outputs = self.base_causallm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=False,
            )
            all_logits.append(outputs.logits)

        if self.enable_qvr_loss and prev_inserted_spans is not None:
            final_seq_len = inputs_embeds.size(1)
            final_outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, :final_seq_len, :],
                attention_mask=attention_mask[:, :final_seq_len],
                position_ids=position_ids[:, :final_seq_len],
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=False,
                output_attentions=True,
            )
            if any(len(lst) > 0 for lst in latent_lists):
                positions_final = torch.arange(final_seq_len, device=input_ids.device).unsqueeze(0).expand(B, -1)
                first_latent_pos = min(lst[0] for lst in latent_lists if len(lst) > 0)
                question_mask_final = (
                    original_mask[:, :final_seq_len]
                    & (~image_mask[:, :final_seq_len])
                    & (positions_final < first_latent_pos)
                )
                final_query_positions = [lst[-1] if len(lst) > 0 else None for lst in latent_lists]
                final_qvr_loss = self._compute_qvr_loss(
                    final_outputs.attentions,
                    final_query_positions,
                    prev_inserted_spans,
                    question_mask_final,
                )
                if final_qvr_loss is not None:
                    qvr_losses.append(final_qvr_loss)

        latent_attn_trace = None
        if return_latent_attn and outputs.attentions is not None and max_n_latents > 0:
            final_attn = outputs.attentions[-1].mean(dim=1)
            seq_len = final_attn.size(-1)
            threshold = latent_attn_threshold
            if threshold is None:
                threshold = latent_attn_threshold_multiplier / max(seq_len, 1)
            latent_attn_trace = []
            for b in range(B):
                image_mask_b = image_mask[b, :seq_len]
                original_mask_b = original_mask[b, :seq_len]
                text_mask_b = original_mask_b & (~image_mask_b)
                trace_mask_b = trace_mask[b, :seq_len]

                per_latent = []
                for t_idx in latent_lists[b]:
                    if t_idx >= seq_len:
                        continue
                    row = final_attn[b, t_idx]
                    image_mass = row[image_mask_b].sum().item() if image_mask_b.any() else 0.0
                    text_mass = row[text_mask_b].sum().item() if text_mask_b.any() else 0.0
                    trace_mass = row[trace_mask_b].sum().item() if trace_mask_b.any() else 0.0
                    image_above = (image_mask_b & (row > threshold)).nonzero(as_tuple=True)[0].tolist()
                    trace_above = (trace_mask_b & (row > threshold)).nonzero(as_tuple=True)[0].tolist()
                    per_latent.append({
                        "latent_pos": int(t_idx),
                        "image_mass": float(image_mass),
                        "text_mass": float(text_mass),
                        "trace_mass": float(trace_mass),
                        "threshold": float(threshold),
                        "image_above_threshold": image_above,
                        "trace_above_threshold": trace_above,
                    })
                latent_attn_trace.append(per_latent)

        logits = torch.cat(all_logits, dim=-2)  # (B, total_len, V)
        B, final_S, V = logits.size()

        token_norms = None
        if return_token_norms:
            token_norms = []
            seq_len_for_norms = min(end, inputs_embeds.size(1))
            for b in range(B):
                patch_mask_b = patch_insert_mask[b, :seq_len_for_norms]
                reasoning_mask_b = reasoning_insert_mask[b, :seq_len_for_norms]
                patch_embeds = inputs_embeds[b, :seq_len_for_norms, :][patch_mask_b]
                reasoning_embeds = inputs_embeds[b, :seq_len_for_norms, :][reasoning_mask_b]
                patch_norms = torch.norm(patch_embeds, dim=-1) if patch_embeds.numel() > 0 else torch.tensor([])
                reasoning_norms = (
                    torch.norm(reasoning_embeds, dim=-1) if reasoning_embeds.numel() > 0 else torch.tensor([])
                )
                token_norms.append({
                    "patch_norms": patch_norms.float().tolist(),
                    "reasoning_norms": reasoning_norms.float().tolist(),
                    "aggregate_stats": {
                        "patch": self._summarize_norms(patch_norms),
                        "reasoning": self._summarize_norms(reasoning_norms),
                    },
                })

        qvr_loss = torch.stack(qvr_losses).mean() if (self.enable_qvr_loss and qvr_losses) else None


        new_labels = torch.full((B, final_S), -100, device=input_ids.device, dtype=labels.dtype)
        for b in range(B):
            num_labels = labels.size(1)
            new_labels[b, -num_labels:] = labels[b]

        new_answer_mask = None
        if answer_mask is not None:
            new_answer_mask = torch.zeros((B, final_S), device=answer_mask.device, dtype=answer_mask.dtype)
            for b in range(B):
                num_labels = labels.size(1)
                new_answer_mask[b, -num_labels:] = answer_mask[b]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = new_labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nvt_loss = torch.stack(nvt_losses).mean() if (self.enable_nvt_loss and nvt_losses) else None
        causal_loss = None
        if (
            self.enable_causal_loss
            and new_answer_mask is not None
            and patch_insert_mask is not None
            and B > 1
        ):
            final_seq_len = inputs_embeds.size(1)
            perm = torch.randperm(B, device=inputs_embeds.device)
            if torch.any(perm == torch.arange(B, device=inputs_embeds.device)):
                perm = torch.arange(B, device=inputs_embeds.device).roll(1)

            inputs_embeds_corrupt = inputs_embeds.clone()
            for b in range(B):
                tgt_pos = patch_insert_mask[b, :final_seq_len].nonzero(as_tuple=True)[0]
                if tgt_pos.numel() == 0:
                    continue
                donor_pos = patch_insert_mask[perm[b], :final_seq_len].nonzero(as_tuple=True)[0]
                if donor_pos.numel() == 0:
                    continue
                donor_embeds = inputs_embeds[perm[b], donor_pos, :]
                if donor_embeds.size(0) < tgt_pos.numel():
                    repeat_count = (tgt_pos.numel() + donor_embeds.size(0) - 1) // donor_embeds.size(0)
                    donor_embeds = donor_embeds.repeat(repeat_count, 1)[:tgt_pos.numel()]
                elif donor_embeds.size(0) > tgt_pos.numel():
                    donor_embeds = donor_embeds[:tgt_pos.numel()]
                inputs_embeds_corrupt[b, tgt_pos, :] = donor_embeds

            corrupt_outputs = self.base_causallm(
                inputs_embeds=inputs_embeds_corrupt[:, :final_seq_len, :],
                attention_mask=attention_mask[:, :final_seq_len],
                position_ids=position_ids[:, :final_seq_len],
                output_hidden_states=False,
                output_attentions=False,
            )
            s_real, real_counts = self._compute_answer_logprob(logits, new_labels, new_answer_mask)
            s_corrupt, corrupt_counts = self._compute_answer_logprob(
                corrupt_outputs.logits, new_labels, new_answer_mask
            )
            valid = (real_counts > 0) & (corrupt_counts > 0)
            if valid.any():
                causal_loss = F.softplus(-(s_real - s_corrupt))[valid].mean()
        loss = ce_loss
        if nvt_loss is not None:
            loss = loss + self.nvt_loss_weight * nvt_loss
        if qvr_loss is not None:
            loss = loss + self.qvr_loss_weight * qvr_loss
        if causal_loss is not None:
            loss = loss + self.causal_loss_weight * causal_loss

        return Outputs(
            loss=loss,
            ce_loss=ce_loss,
            nvt_loss=nvt_loss,
            qvr_loss=qvr_loss,
            causal_loss=causal_loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            latent_attn_trace=latent_attn_trace,
            token_norms=token_norms,
        )


    def train(self, mode=True):
        self.base_causallm.train(mode)

    def eval(self):
        self.base_causallm.eval()
    
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            image_grid_thw: torch.Tensor = None,
            past_key_values: tuple = None,
            attention_mask: torch.Tensor = None,
            inputs_embeds: torch.FloatTensor = None,
            position_ids: torch.LongTensor = None,
            use_cache: bool = True,
            **kwargs
        ):
        
        self.base_causallm.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        pixel_values,
        image_grid_thw,
        max_new_tokens=16,
        output_embedding=False,
        **kwargs
    ):
        self.gen_forward_cnt = 0
        eos_pos = None

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        
        current_ids = input_ids.clone()

        position_ids = torch.arange(
            0, current_ids.shape[1], 
            dtype=torch.long, 
            device=current_ids.device
        ).reshape(1, -1)

        outputs = self.forward(
            input_ids=current_ids,
            attention_mask=torch.ones_like(current_ids),
            labels=current_ids.clone(),  
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )


        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
            

        current_inputs_embeds = outputs.inputs_embeds  # shape: (1, seq_len_after_insertion, hidden_dim)
        current_seq_len = current_inputs_embeds.shape[1]
        

        current_attention_mask = torch.ones((1, current_seq_len), device=current_inputs_embeds.device)
        

        next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

        self.gen_forward_cnt += 1
        

        past_key_values = None
        

        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                logging.debug(f"no kv_cache, using full embedding sequence")
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                        0, current_inputs_embeds.shape[1], 
                    dtype=torch.long, 
                        device=current_inputs_embeds.device
                ).reshape(1, -1)
            else:
                logging.debug(f"using kv_cache, input_shape: {next_token_embedding.shape}")
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor([[current_inputs_embeds.shape[1] - 1]], device=current_inputs_embeds.device)

            outputs = self.base_causallm.forward(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None, 
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            
            next_token_embedding = self.embedding(torch.tensor([[next_token]], device=current_inputs_embeds.device))
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat([current_attention_mask, torch.ones((1, 1), device=current_inputs_embeds.device)], dim=1)

            self.gen_forward_cnt += 1

            if self.gen_forward_cnt % 10 == 0 and self.gen_forward_cnt >= 10:
                logging.debug(f"gen_forward_cnt: {self.gen_forward_cnt}")

            if next_token == self.eos_token_id:
                logging.debug(f"EOS token encountered at position {len(tokens)}, stopping generation")
                break

        print("generate 315")
        
        
        if output_embedding:
            return torch.tensor(tokens).view(1, -1), current_inputs_embeds
        else:
            return torch.tensor(tokens).view(1, -1)


