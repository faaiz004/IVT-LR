from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ControllerSequenceStats:
    loss: torch.Tensor
    mean_logprob: torch.Tensor
    patch_top1_accuracy: torch.Tensor
    stop_accuracy: torch.Tensor
    token_count: int


class ControllerStateUpdater(nn.Module):
    def __init__(
        self,
        controller_dim: int,
        patch_dim: int,
        max_steps: int = 10,
        use_step_embedding: bool = True,
    ):
        super().__init__()
        self.controller_dim = controller_dim
        self.max_steps = max_steps
        self.use_step_embedding = use_step_embedding
        step_dim = controller_dim if use_step_embedding else 0
        self.step_embedding = (
            nn.Embedding(max_steps, controller_dim) if use_step_embedding else None
        )
        self.mlp = nn.Sequential(
            nn.Linear(controller_dim + patch_dim + step_dim, controller_dim * 4),
            nn.GELU(),
            nn.Linear(controller_dim * 4, controller_dim),
        )
        self.norm = nn.LayerNorm(controller_dim)

    def forward(
        self,
        controller_state: torch.Tensor,
        selected_patch: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        pieces = [controller_state, selected_patch]
        if self.use_step_embedding:
            step = min(step_idx, self.max_steps - 1)
            step_ids = torch.full(
                (controller_state.size(0),),
                step,
                dtype=torch.long,
                device=controller_state.device,
            )
            pieces.append(self.step_embedding(step_ids))
        update = self.mlp(torch.cat(pieces, dim=-1))
        return self.norm(controller_state + update)


class PatchPointerController(nn.Module):
    """Pointer controller over per-example image patch embeddings plus STOP.

    STOP is represented only as the last controller logit. It is never a text
    tokenizer id and should never be appended to the IVT-LR embedding stream.
    """

    def __init__(
        self,
        model_dim: int,
        controller_dim: Optional[int] = None,
        max_steps: int = 10,
        use_step_embedding: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.controller_dim = controller_dim or model_dim
        self.max_steps = max_steps
        self.state_proj = nn.Linear(model_dim, self.controller_dim)
        self.query_proj = nn.Linear(self.controller_dim, self.controller_dim)
        self.key_proj = nn.Linear(model_dim, self.controller_dim)
        self.stop_proj = nn.Linear(self.controller_dim, 1)
        self.updater = ControllerStateUpdater(
            controller_dim=self.controller_dim,
            patch_dim=model_dim,
            max_steps=max_steps,
            use_step_embedding=use_step_embedding,
        )
        self.scale = self.controller_dim ** -0.5

    def initial_state(self, reasoning_state: torch.Tensor) -> torch.Tensor:
        return self.state_proj(reasoning_state)

    def forward(
        self,
        controller_state: torch.Tensor,
        patch_embeddings: torch.Tensor,
        patch_valid_mask: Optional[torch.Tensor] = None,
        selected_mask: Optional[torch.Tensor] = None,
        allow_stop: bool = True,
    ) -> torch.Tensor:
        if patch_embeddings.dim() != 3:
            raise ValueError("patch_embeddings must have shape [B, N, D]")
        bsz, n_patches, _ = patch_embeddings.shape
        q = self.query_proj(controller_state).unsqueeze(-1)
        k = self.key_proj(patch_embeddings)
        patch_logits = torch.bmm(k, q).squeeze(-1) * self.scale

        if patch_valid_mask is not None:
            patch_logits = patch_logits.masked_fill(~patch_valid_mask.bool(), float("-inf"))
        if selected_mask is not None:
            patch_logits = patch_logits.masked_fill(selected_mask.bool(), float("-inf"))

        stop_logit = self.stop_proj(controller_state)
        if not allow_stop:
            stop_logit = stop_logit.fill_(float("-inf"))
        return torch.cat([patch_logits, stop_logit], dim=-1)

    def update_state(
        self,
        controller_state: torch.Tensor,
        selected_patch: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        return self.updater(controller_state, selected_patch, step_idx)

    def gather_patch(
        self,
        patch_embeddings: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        gather_idx = patch_indices.view(-1, 1, 1).expand(-1, 1, patch_embeddings.size(-1))
        return patch_embeddings.gather(1, gather_idx).squeeze(1)

    def teacher_forced_sequence_loss(
        self,
        reasoning_state: torch.Tensor,
        patch_embeddings: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        target_actions: torch.Tensor,
        sequence_weights: Optional[torch.Tensor] = None,
    ) -> ControllerSequenceStats:
        """Sequentially train p1, p2, ..., STOP with teacher forcing.

        target_actions has shape [B, L]. STOP must be encoded as N, where N is
        the padded patch dimension for patch_embeddings.
        """
        bsz, n_patches, _ = patch_embeddings.shape
        stop_index = n_patches
        state = self.initial_state(reasoning_state)
        selected_mask = torch.zeros(
            (bsz, n_patches), dtype=torch.bool, device=patch_embeddings.device
        )
        logprob_sum = torch.zeros(bsz, device=patch_embeddings.device)
        token_counts = torch.zeros(bsz, device=patch_embeddings.device)
        patch_correct = torch.zeros(bsz, device=patch_embeddings.device)
        patch_total = torch.zeros(bsz, device=patch_embeddings.device)
        stop_correct = torch.zeros(bsz, device=patch_embeddings.device)
        stop_total = torch.zeros(bsz, device=patch_embeddings.device)

        for step_idx in range(target_actions.size(1)):
            target = target_actions[:, step_idx]
            active = target >= 0
            if not active.any():
                break

            logits = self.forward(
                state,
                patch_embeddings,
                patch_valid_mask=patch_valid_mask,
                selected_mask=selected_mask,
            )
            log_probs = F.log_softmax(logits, dim=-1)
            safe_target = target.clamp(min=0)
            step_logprob = log_probs.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            logprob_sum = logprob_sum + step_logprob.masked_fill(~active, 0.0)
            token_counts = token_counts + active.float()

            pred = logits.argmax(dim=-1)
            is_stop = target == stop_index
            is_patch = active & ~is_stop
            patch_correct = patch_correct + ((pred == target) & is_patch).float()
            patch_total = patch_total + is_patch.float()
            stop_correct = stop_correct + ((pred == stop_index) & is_stop).float()
            stop_total = stop_total + is_stop.float()

            update_mask = is_patch
            if update_mask.any():
                patch_target = target.clamp(max=n_patches - 1)
                selected_patch = self.gather_patch(patch_embeddings, patch_target)
                updated_state = self.update_state(state, selected_patch, step_idx)
                state = torch.where(update_mask.unsqueeze(-1), updated_state, state)
                selected_update = torch.zeros_like(selected_mask)
                selected_update = selected_update.scatter(1, patch_target.unsqueeze(1), True)
                selected_mask = selected_mask | (selected_update & update_mask.unsqueeze(1))

        if sequence_weights is None:
            sequence_weights = torch.ones_like(logprob_sum)
        sequence_weights = sequence_weights.detach()
        loss = -(sequence_weights * logprob_sum).mean()
        denom = token_counts.clamp(min=1.0)
        mean_logprob = (logprob_sum / denom).mean()
        patch_acc = (patch_correct.sum() / patch_total.sum().clamp(min=1.0)).detach()
        stop_acc = (stop_correct.sum() / stop_total.sum().clamp(min=1.0)).detach()
        return ControllerSequenceStats(
            loss=loss,
            mean_logprob=mean_logprob.detach(),
            patch_top1_accuracy=patch_acc,
            stop_accuracy=stop_acc,
            token_count=int(token_counts.sum().item()),
        )

    @torch.no_grad()
    def greedy_select(
        self,
        reasoning_state: torch.Tensor,
        patch_embeddings: torch.Tensor,
        patch_valid_mask: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        max_steps = max_steps or self.max_steps
        bsz, n_patches, _ = patch_embeddings.shape
        stop_index = n_patches
        state = self.initial_state(reasoning_state)
        selected_mask = torch.zeros(
            (bsz, n_patches), dtype=torch.bool, device=patch_embeddings.device
        )
        selected = torch.full(
            (bsz, max_steps), -1, dtype=torch.long, device=patch_embeddings.device
        )
        lengths = torch.zeros(bsz, dtype=torch.long, device=patch_embeddings.device)
        stopped = torch.zeros(bsz, dtype=torch.bool, device=patch_embeddings.device)

        for step_idx in range(max_steps):
            logits = self.forward(
                state,
                patch_embeddings,
                patch_valid_mask=patch_valid_mask,
                selected_mask=selected_mask,
            )
            action = logits.argmax(dim=-1)
            is_stop = action == stop_index
            take_patch = (~stopped) & (~is_stop)
            if take_patch.any():
                patch_action = action.clamp(max=n_patches - 1)
                selected[:, step_idx] = torch.where(take_patch, patch_action, selected[:, step_idx])
                selected_patch = self.gather_patch(patch_embeddings, patch_action)
                updated_state = self.update_state(state, selected_patch, step_idx)
                state = torch.where(take_patch.unsqueeze(-1), updated_state, state)
                selected_update = torch.zeros_like(selected_mask)
                selected_update = selected_update.scatter(1, patch_action.unsqueeze(1), True)
                selected_mask = selected_mask | (selected_update & take_patch.unsqueeze(1))
                lengths = lengths + take_patch.long()
            stopped = stopped | is_stop
            if stopped.all():
                break

        return {"selected_indices": selected, "lengths": lengths, "stopped": stopped}
