from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import AutoProcessor

try:
    from controller import PatchPointerController
except ImportError:
    from qwen_vl.controller import PatchPointerController


@dataclass
class LatentStepTrace:
    latent_step_idx: int
    reasoning_state: torch.Tensor
    image_positions: torch.Tensor
    patch_embeddings: torch.Tensor
    patch_valid_mask: torch.Tensor
    attention_scores: torch.Tensor
    ranked_patch_indices: torch.Tensor
    selected_patch_embeddings: torch.Tensor
    end_position: int
    selected_count: int


@dataclass
class AdaptiveIVTLROutput:
    loss: Optional[torch.Tensor]
    ce_loss: Optional[torch.Tensor]
    logits: torch.Tensor
    inputs_embeds: torch.Tensor
    answer_logprob: Optional[torch.Tensor] = None
    answer_token_counts: Optional[torch.Tensor] = None
    total_inserted_patches: int = 0
    latent_traces: List[LatentStepTrace] = field(default_factory=list)
    controller_trace: List[Dict[str, torch.Tensor]] = field(default_factory=list)


class QwenAdaptiveIVTLR(nn.Module):
    """Adaptive-controller IVT-LR path.

    The existing qwen_ivtlr.IVTLR baseline stays untouched. This wrapper
    mirrors its latent insertion mechanics, then adds teacher traces,
    forced-budget replay, and controller-driven selection. STOP exists only in
    the controller action space and is never inserted into inputs_embeds.
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        image_token_id: int,
        visual_start_id: int,
        visual_end_id: int,
        controller: Optional[PatchPointerController] = None,
        teacher_k: int = 10,
        max_controller_steps: int = 10,
        patch_reuse_policy: str = "never",
        processor_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.visual_start_id = visual_start_id
        self.visual_end_id = visual_end_id
        self.teacher_k = int(teacher_k)
        self.max_controller_steps = int(max_controller_steps)
        if patch_reuse_policy not in {"never", "next_step_only", "always"}:
            raise ValueError(f"Invalid patch_reuse_policy={patch_reuse_policy}")
        self.patch_reuse_policy = patch_reuse_policy
        self.processor = AutoProcessor.from_pretrained(processor_model_id)

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        model_dim = self.embedding.embedding_dim
        self.controller = controller or PatchPointerController(
            model_dim=model_dim,
            max_steps=max_controller_steps,
        )

    def freeze_base_model(self):
        for param in self.base_causallm.parameters():
            param.requires_grad = False
        return self

    def train_controller_only(self):
        self.freeze_base_model()
        for param in self.controller.parameters():
            param.requires_grad = True
        return self

    def _prepare_inputs_embeds(self, input_ids, pixel_values, image_grid_thw):
        inputs_embeds = self.embedding(input_ids)
        if pixel_values is None:
            image_mask_init = torch.zeros_like(input_ids, dtype=torch.bool)
            return inputs_embeds, image_mask_init

        pixel_values = pixel_values.type(self.base_causallm.visual.get_dtype())
        image_embeds = self.base_causallm.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, "
                f"features {image_embeds.shape[0]}"
            )
        image_mask_init = input_ids == self.image_token_id
        expand_mask = image_mask_init.unsqueeze(-1).expand(-1, -1, inputs_embeds.size(-1))
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(expand_mask, image_embeds)
        return inputs_embeds, image_mask_init

    @staticmethod
    def _pad_1d(items: List[torch.Tensor], value=0, dtype=None):
        max_len = max((x.numel() for x in items), default=0)
        if max_len == 0:
            device = items[0].device
            return torch.empty((len(items), 0), device=device, dtype=dtype or items[0].dtype)
        out = []
        for item in items:
            pad_len = max_len - item.numel()
            if pad_len:
                pad = torch.full(
                    (pad_len,),
                    value,
                    device=item.device,
                    dtype=dtype or item.dtype,
                )
                item = torch.cat([item.to(dtype=dtype or item.dtype), pad], dim=0)
            out.append(item)
        return torch.stack(out, dim=0)

    @staticmethod
    def _pad_2d(items: List[torch.Tensor]):
        max_len = max((x.size(0) for x in items), default=0)
        width = items[0].size(-1)
        out = []
        for item in items:
            pad_len = max_len - item.size(0)
            if pad_len:
                pad = torch.zeros(
                    (pad_len, width),
                    device=item.device,
                    dtype=item.dtype,
                )
                item = torch.cat([item, pad], dim=0)
            out.append(item)
        return torch.stack(out, dim=0)

    def _rank_image_patches(
        self,
        inputs_embeds,
        image_mask,
        attn_scores,
        top_k: int,
    ):
        patch_positions = []
        patch_scores = []
        patch_embeds = []
        ranked_indices = []
        valid_masks = []
        selected_embeds = []
        for b in range(inputs_embeds.size(0)):
            positions = image_mask[b, : attn_scores.size(1)].nonzero(as_tuple=True)[0]
            scores = attn_scores[b, positions] if positions.numel() else torch.empty(0, device=attn_scores.device)
            order = torch.argsort(scores, descending=True)
            ranked = order[: min(top_k, order.numel())]
            embeds = inputs_embeds[b, positions, :]
            patch_positions.append(positions)
            patch_scores.append(scores)
            patch_embeds.append(embeds)
            ranked_indices.append(ranked)
            valid_masks.append(torch.ones(positions.numel(), device=inputs_embeds.device, dtype=torch.bool))
            selected_embeds.append(embeds[ranked] if ranked.numel() else embeds[:0])

        return {
            "positions": self._pad_1d(patch_positions, value=-1, dtype=torch.long),
            "scores": self._pad_1d(patch_scores, value=float("-inf")),
            "embeddings": self._pad_2d(patch_embeds),
            "ranked_indices": self._pad_1d(ranked_indices, value=-1, dtype=torch.long),
            "valid_mask": self._pad_1d(valid_masks, value=False, dtype=torch.bool),
            "selected_embeddings": self._pad_2d(selected_embeds),
        }

    def _select_for_step(
        self,
        mode: str,
        step_idx: int,
        reasoning_state: torch.Tensor,
        ranked: Dict[str, torch.Tensor],
        forced_budget: Optional[int],
        teacher_trace: Optional[Sequence[LatentStepTrace]],
    ):
        positions = ranked["positions"]
        if mode == "teacher":
            rel_indices = ranked["ranked_indices"][:, : self.teacher_k]
        elif mode == "forced_budget":
            if teacher_trace is None:
                raise ValueError("forced_budget mode requires teacher_trace")
            rel_indices = teacher_trace[step_idx].ranked_patch_indices[:, : forced_budget]
        elif mode == "adaptive":
            if reasoning_state.size(0) != 1:
                raise ValueError("adaptive controller inference currently expects batch_size=1")
            selection = self.controller.greedy_select(
                reasoning_state,
                ranked["embeddings"],
                ranked["valid_mask"],
                max_steps=self.max_controller_steps,
            )
            rel_indices = selection["selected_indices"][:, : int(selection["lengths"].max().item())]
        elif mode == "adaptive_sample":
            if reasoning_state.size(0) != 1:
                raise ValueError("sampled adaptive controller currently expects batch_size=1")
            selection = self.controller.sample_select(
                reasoning_state,
                ranked["embeddings"],
                ranked["valid_mask"],
                max_steps=self.max_controller_steps,
                temperature=getattr(self, "_controller_sample_temperature", 1.0),
                min_patches=getattr(self, "_controller_min_patches", 0),
            )
            rel_indices = selection["selected_indices"][:, : int(selection["lengths"].max().item())]
        else:
            raise ValueError(f"Unknown adaptive IVT-LR mode: {mode}")

        selected_positions = []
        selected_embeds = []
        selected_counts = []
        for b in range(positions.size(0)):
            rel = rel_indices[b]
            rel = rel[rel >= 0]
            valid_rel = rel[rel < ranked["embeddings"].size(1)]
            pos = positions[b, valid_rel] if valid_rel.numel() else positions[b, :0]
            pos = pos[pos >= 0]
            embeds = (
                ranked["embeddings"][b, valid_rel, :]
                if valid_rel.numel()
                else ranked["embeddings"][b, :0, :]
            )
            selected_positions.append(pos)
            selected_embeds.append(embeds)
            selected_counts.append(embeds.size(0))
        return selected_positions, selected_embeds, selected_counts, rel_indices, selection if mode in {"adaptive", "adaptive_sample"} else None

    def _merge_selected_embeddings(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        original_mask,
        image_mask,
        trace_mask,
        latent_lists,
        end,
        selected_positions,
        selected_embeds,
        selected_counts,
        hidden_states,
        pass_idx,
    ):
        inputs_embeds_detached = inputs_embeds.detach().clone()
        for b in range(inputs_embeds.size(0)):
            if len(latent_lists[b]) > pass_idx:
                t_idx = latent_lists[b][pass_idx]
                rel_pos = max(0, min(t_idx - 1, hidden_states.size(1) - 1))
                inputs_embeds_detached[b, t_idx, :] = hidden_states[b, rel_pos, :]
        inputs_embeds = inputs_embeds_detached

        new_embeds = []
        new_att = []
        new_pos = []
        new_orig = []
        new_img = []
        new_trace = []
        max_len = 0
        for b in range(inputs_embeds.size(0)):
            k_b = selected_counts[b]
            merged = torch.cat(
                [inputs_embeds[b, :end, :], selected_embeds[b], inputs_embeds[b, end:, :]],
                dim=0,
            )
            att = torch.cat(
                [
                    attention_mask[b, :end],
                    torch.ones(k_b, device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask[b, end:],
                ],
                dim=0,
            )
            pos = torch.arange(merged.size(0), device=position_ids.device)
            orig = torch.cat(
                [
                    original_mask[b, :end],
                    torch.zeros(k_b, device=original_mask.device, dtype=torch.bool),
                    original_mask[b, end:],
                ],
                dim=0,
            )
            img = torch.cat(
                [
                    image_mask[b, :end],
                    torch.zeros(k_b, device=image_mask.device, dtype=torch.bool),
                    image_mask[b, end:],
                ],
                dim=0,
            )
            trace = torch.cat(
                [
                    trace_mask[b, :end],
                    torch.ones(k_b, device=trace_mask.device, dtype=torch.bool),
                    trace_mask[b, end:],
                ],
                dim=0,
            )
            new_embeds.append(merged)
            new_att.append(att)
            new_pos.append(pos)
            new_orig.append(orig)
            new_img.append(img)
            new_trace.append(trace)
            max_len = max(max_len, merged.size(0))

        def pad_vec(item, value=0):
            if item.size(0) == max_len:
                return item
            return F.pad(item, (0, max_len - item.size(0)), value=value)

        def pad_embed(item):
            if item.size(0) == max_len:
                return item
            pad = torch.zeros(
                (max_len - item.size(0), item.size(1)),
                device=item.device,
                dtype=item.dtype,
            )
            return torch.cat([item, pad], dim=0)

        inputs_embeds = torch.stack([pad_embed(x) for x in new_embeds], dim=0)
        attention_mask = torch.stack([pad_vec(x) for x in new_att], dim=0)
        position_ids = torch.stack([pad_vec(x) for x in new_pos], dim=0)
        original_mask = torch.stack([pad_vec(x).bool() for x in new_orig], dim=0)
        image_mask = torch.stack([pad_vec(x).bool() for x in new_img], dim=0)
        trace_mask = torch.stack([pad_vec(x).bool() for x in new_trace], dim=0)

        if self.patch_reuse_policy == "never":
            for b, pos in enumerate(selected_positions):
                image_mask[b, pos] = False

        for b, k_b in enumerate(selected_counts):
            for i, pos in enumerate(latent_lists[b]):
                if pos > end:
                    latent_lists[b][i] = pos + k_b

        return inputs_embeds, attention_mask, position_ids, original_mask, image_mask, trace_mask

    @staticmethod
    def _answer_logprob(logits, labels, answer_mask):
        if answer_mask is None:
            answer_mask = labels != -100
        seq_len = min(logits.size(1), labels.size(1), answer_mask.size(1))
        logits = logits[:, -seq_len:, :]
        labels = labels[:, -seq_len:]
        answer_mask = answer_mask[:, -seq_len:].bool()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = answer_mask[:, 1:] & (shift_labels != -100)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
        gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~shift_mask, 0.0)
        counts = shift_mask.sum(dim=-1)
        avg_logprob = gathered.sum(dim=-1) / counts.clamp(min=1)
        return avg_logprob, counts

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        position_ids,
        pixel_values,
        image_grid_thw=None,
        answer_mask=None,
        mode: str = "teacher",
        forced_budget: Optional[int] = None,
        teacher_trace: Optional[Sequence[LatentStepTrace]] = None,
        return_trace: bool = False,
    ) -> AdaptiveIVTLROutput:
        bsz, seq_len = input_ids.shape
        inputs_embeds, image_mask_init = self._prepare_inputs_embeds(
            input_ids, pixel_values, image_grid_thw
        )
        original_mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=input_ids.device)
        image_mask = image_mask_init.clone()
        trace_mask = torch.zeros_like(image_mask)

        vs_indices = (input_ids == self.visual_start_id).nonzero(as_tuple=True)
        ve_indices = (input_ids == self.visual_end_id).nonzero(as_tuple=True)
        vs_pos_per_batch = {b.item(): vs_indices[1][i].item() for i, b in enumerate(vs_indices[0])}
        ve_pos_per_batch = {b.item(): ve_indices[1][i].item() for i, b in enumerate(ve_indices[0])}
        for b in range(bsz):
            if b in vs_pos_per_batch and b in ve_pos_per_batch:
                image_mask[b, vs_pos_per_batch[b] + 1 : ve_pos_per_batch[b]] = True

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == b]
            for b in range(bsz)
        ]
        max_latents = max((len(x) for x in latent_lists), default=0)
        end = min((lst[0] for lst in latent_lists if lst), default=seq_len)
        all_logits = []
        traces = []
        controller_trace = []
        total_inserted = 0

        for pass_idx in range(max_latents):
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, :end, :],
                attention_mask=attention_mask[:, :end],
                position_ids=position_ids[:, :end],
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                output_attentions=True,
            )
            all_logits.append(outputs.logits)
            hidden_states = outputs.hidden_states[-1]
            attentions = outputs.attentions
            avg_attn = torch.cat(attentions, dim=1).mean(dim=1)
            attn_scores = avg_attn[:, end - 1, :]
            reasoning_state = []
            for b in range(bsz):
                t_idx = latent_lists[b][pass_idx]
                rel_pos = max(0, min(t_idx - 1, hidden_states.size(1) - 1))
                reasoning_state.append(hidden_states[b, rel_pos, :])
            reasoning_state = torch.stack(reasoning_state, dim=0)

            ranked = self._rank_image_patches(
                inputs_embeds,
                image_mask[:, : attn_scores.size(1)],
                attn_scores,
                top_k=max(self.teacher_k, forced_budget or 0),
            )
            selected_positions, selected_embeds, selected_counts, selected_rel_indices, selection_info = self._select_for_step(
                mode,
                pass_idx,
                reasoning_state,
                ranked,
                forced_budget,
                teacher_trace,
            )
            total_inserted += sum(selected_counts)
            if return_trace or mode == "teacher":
                traces.append(
                    LatentStepTrace(
                        latent_step_idx=pass_idx,
                        reasoning_state=reasoning_state.detach(),
                        image_positions=ranked["positions"].detach(),
                        patch_embeddings=ranked["embeddings"].detach(),
                        patch_valid_mask=ranked["valid_mask"].detach(),
                        attention_scores=ranked["scores"].detach(),
                        ranked_patch_indices=ranked["ranked_indices"].detach(),
                        selected_patch_embeddings=ranked["selected_embeddings"].detach(),
                        end_position=end,
                        selected_count=max(selected_counts) if selected_counts else 0,
                    )
                )
            if mode in {"adaptive", "adaptive_sample"}:
                step_trace = {
                    "latent_step_idx": pass_idx,
                    "selected_counts": torch.tensor(selected_counts),
                    "selected_patch_indices": selected_rel_indices.detach().cpu(),
                }
                if selection_info is not None and "logprob_sum" in selection_info:
                    step_trace.update(
                        {
                            "logprob_sum": selection_info["logprob_sum"],
                            "entropy_sum": selection_info["entropy_sum"],
                            "action_count": selection_info["action_count"],
                        }
                    )
                controller_trace.append(step_trace)

            (
                inputs_embeds,
                attention_mask,
                position_ids,
                original_mask,
                image_mask,
                trace_mask,
            ) = self._merge_selected_embeddings(
                inputs_embeds,
                attention_mask,
                position_ids,
                original_mask,
                image_mask,
                trace_mask,
                latent_lists,
                end,
                selected_positions,
                selected_embeds,
                selected_counts,
                hidden_states,
                pass_idx,
            )
            if pass_idx + 1 >= max_latents:
                end = inputs_embeds.size(1)
            else:
                end = end + 1 + max(selected_counts)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, :end, :],
            attention_mask=attention_mask[:, :end],
            position_ids=position_ids[:, :end],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=False,
            output_attentions=False,
        )
        all_logits.append(outputs.logits)
        logits = torch.cat(all_logits, dim=1)

        final_s = logits.size(1)
        new_labels = torch.full((bsz, final_s), -100, device=input_ids.device, dtype=labels.dtype)
        new_answer_mask = torch.zeros((bsz, final_s), device=input_ids.device, dtype=torch.bool)
        for b in range(bsz):
            label_len = labels.size(1)
            new_labels[b, -label_len:] = labels[b]
            if answer_mask is not None:
                new_answer_mask[b, -label_len:] = answer_mask[b].bool()
            else:
                new_answer_mask[b, -label_len:] = labels[b] != -100

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = new_labels[:, 1:].contiguous()
        ce_loss = CrossEntropyLoss(ignore_index=-100)(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        answer_logprob, answer_counts = self._answer_logprob(logits, new_labels, new_answer_mask)
        return AdaptiveIVTLROutput(
            loss=ce_loss,
            ce_loss=ce_loss,
            logits=logits,
            inputs_embeds=inputs_embeds,
            answer_logprob=answer_logprob,
            answer_token_counts=answer_counts,
            total_inserted_patches=total_inserted,
            latent_traces=traces,
            controller_trace=controller_trace,
        )

    def controller_teacher_forcing_loss(
        self,
        teacher_trace: Sequence[LatentStepTrace],
        budgets: Sequence[int],
        budget_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = []
        patch_accs = []
        stop_accs = []
        device = budget_weights.device
        for step in teacher_trace:
            reasoning = step.reasoning_state
            patch_embeddings = self._gather_trace_patch_embeddings(step)
            patch_valid = step.patch_valid_mask.bool()
            n_patches = patch_embeddings.size(1)
            for budget_idx, budget in enumerate(budgets):
                k = min(int(budget), step.ranked_patch_indices.size(1))
                targets = step.ranked_patch_indices[:, :k]
                stop = torch.full(
                    (targets.size(0), 1),
                    n_patches,
                    device=targets.device,
                    dtype=torch.long,
                )
                target_actions = torch.cat([targets, stop], dim=1)
                weights = budget_weights[:, budget_idx].to(device)
                stats = self.controller.teacher_forced_sequence_loss(
                    reasoning,
                    patch_embeddings,
                    patch_valid,
                    target_actions,
                    sequence_weights=weights,
                )
                losses.append(stats.loss)
                patch_accs.append(stats.patch_top1_accuracy)
                stop_accs.append(stats.stop_accuracy)
        if not losses:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": zero, "patch_top1_accuracy": zero.detach(), "stop_accuracy": zero.detach()}
        return {
            "loss": torch.stack(losses).mean(),
            "patch_top1_accuracy": torch.stack(patch_accs).mean(),
            "stop_accuracy": torch.stack(stop_accs).mean(),
        }

    @staticmethod
    def _gather_trace_patch_embeddings(step: LatentStepTrace) -> torch.Tensor:
        return step.patch_embeddings

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        max_new_tokens: int = 128,
        output_controller_trace: bool = False,
    ):
        if input_ids.size(0) != 1:
            raise ValueError("Adaptive controller generation currently supports batch_size=1.")

        self.eval()
        position_ids = torch.arange(
            input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        adaptive_out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mode="adaptive",
        )

        tokens = input_ids[0].detach().tolist()
        next_token = torch.argmax(adaptive_out.logits[0, -1]).item()
        tokens.append(next_token)

        current_inputs_embeds = adaptive_out.inputs_embeds
        current_attention_mask = torch.ones(
            (1, current_inputs_embeds.size(1)),
            device=current_inputs_embeds.device,
            dtype=attention_mask.dtype,
        )
        next_token_embedding = self.embedding(
            torch.tensor([[next_token]], device=current_inputs_embeds.device)
        )
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((1, 1), device=current_inputs_embeds.device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )

        past_key_values = None
        for _ in range(max_new_tokens - 1):
            if past_key_values is None:
                inputs_embeds_for_forward = current_inputs_embeds
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.arange(
                    current_inputs_embeds.size(1),
                    dtype=torch.long,
                    device=current_inputs_embeds.device,
                ).unsqueeze(0)
            else:
                inputs_embeds_for_forward = next_token_embedding
                attention_mask_for_forward = current_attention_mask
                position_ids = torch.tensor(
                    [[current_inputs_embeds.size(1) - 1]],
                    dtype=torch.long,
                    device=current_inputs_embeds.device,
                )

            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds_for_forward,
                attention_mask=attention_mask_for_forward,
                position_ids=position_ids,
                pixel_values=pixel_values if past_key_values is None else None,
                image_grid_thw=image_grid_thw if past_key_values is None else None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

            next_token_embedding = self.embedding(
                torch.tensor([[next_token]], device=current_inputs_embeds.device)
            )
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((1, 1), device=current_inputs_embeds.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )

        output_ids = torch.tensor(tokens, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if output_controller_trace:
            return output_ids, adaptive_out.controller_trace
        return output_ids

    def generate_with_sampled_controller(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        max_new_tokens: int = 128,
        controller_temperature: float = 1.0,
        min_patches: int = 0,
    ):
        if input_ids.size(0) != 1:
            raise ValueError("Sampled controller generation currently supports batch_size=1.")

        self._controller_sample_temperature = controller_temperature
        self._controller_min_patches = min_patches
        position_ids = torch.arange(
            input_ids.size(1),
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0)
        adaptive_out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids.clone(),
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mode="adaptive_sample",
        )

        tokens = input_ids[0].detach().tolist()
        next_token = torch.argmax(adaptive_out.logits[0, -1].detach()).item()
        tokens.append(next_token)

        current_inputs_embeds = adaptive_out.inputs_embeds.detach()
        current_attention_mask = torch.ones(
            (1, current_inputs_embeds.size(1)),
            device=current_inputs_embeds.device,
            dtype=attention_mask.dtype,
        )
        next_token_embedding = self.embedding(
            torch.tensor([[next_token]], device=current_inputs_embeds.device)
        ).detach()
        current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones((1, 1), device=current_inputs_embeds.device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )

        past_key_values = None
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                if past_key_values is None:
                    inputs_embeds_for_forward = current_inputs_embeds
                    attention_mask_for_forward = current_attention_mask
                    position_ids = torch.arange(
                        current_inputs_embeds.size(1),
                        dtype=torch.long,
                        device=current_inputs_embeds.device,
                    ).unsqueeze(0)
                else:
                    inputs_embeds_for_forward = next_token_embedding
                    attention_mask_for_forward = current_attention_mask
                    position_ids = torch.tensor(
                        [[current_inputs_embeds.size(1) - 1]],
                        dtype=torch.long,
                        device=current_inputs_embeds.device,
                    )

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds_for_forward,
                    attention_mask=attention_mask_for_forward,
                    position_ids=position_ids,
                    pixel_values=pixel_values if past_key_values is None else None,
                    image_grid_thw=image_grid_thw if past_key_values is None else None,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[0, -1]).item()
                tokens.append(next_token)
                if next_token == self.eos_token_id:
                    break

                next_token_embedding = self.embedding(
                    torch.tensor([[next_token]], device=current_inputs_embeds.device)
                ).detach()
                current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embedding], dim=1)
                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones((1, 1), device=current_inputs_embeds.device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )

        output_ids = torch.tensor(tokens, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        return output_ids, adaptive_out.controller_trace
