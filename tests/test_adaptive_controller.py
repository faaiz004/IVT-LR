import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN_DIR = os.path.join(REPO_ROOT, "qwen_vl")
if QWEN_DIR not in sys.path:
    sys.path.insert(0, QWEN_DIR)

from controller import PatchPointerController


def test_stop_is_extra_controller_logit_not_patch_embedding():
    controller = PatchPointerController(model_dim=8, controller_dim=8, max_steps=4)
    state = torch.randn(2, 8)
    patches = torch.randn(2, 5, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    logits = controller(controller.initial_state(state), patches, valid)

    assert logits.shape == (2, 6)
    stop_index = patches.size(1)
    target_sequence = torch.tensor([[1, 2, stop_index], [0, 3, stop_index]])
    inserted_patch_count = (target_sequence != stop_index).sum(dim=1)
    assert inserted_patch_count.tolist() == [2, 2]


def test_selected_patches_are_masked_from_future_steps():
    controller = PatchPointerController(model_dim=8, controller_dim=8, max_steps=4)
    state = controller.initial_state(torch.randn(1, 8))
    patches = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    selected = torch.zeros(1, 4, dtype=torch.bool)
    selected[0, 2] = True

    logits = controller(state, patches, valid, selected_mask=selected)
    assert torch.isneginf(logits[0, 2])
    assert not torch.isneginf(logits[0, 4])


def test_teacher_forcing_is_sequential_and_predicts_stop_after_patches():
    torch.manual_seed(0)
    controller = PatchPointerController(model_dim=8, controller_dim=8, max_steps=4)
    reasoning = torch.randn(1, 8)
    patches = torch.randn(1, 4, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    stop_index = patches.size(1)
    targets = torch.tensor([[1, 2, stop_index]])

    stats = controller.teacher_forced_sequence_loss(reasoning, patches, valid, targets)
    assert torch.isfinite(stats.loss)
    assert stats.token_count == 3


if __name__ == "__main__":
    test_stop_is_extra_controller_logit_not_patch_embedding()
    test_selected_patches_are_masked_from_future_steps()
    test_teacher_forcing_is_sequential_and_predicts_stop_after_patches()
    print("adaptive controller sanity tests passed")
