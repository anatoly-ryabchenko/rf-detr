# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for segmentation mask postprocessing in PostProcess.

These pin the memory-bounded (chunked) mask upsampling introduced for high-resolution segmentation validation:
the chunked path must be bit-identical to a single :func:`F.interpolate` over all masks — chunking only bounds
peak memory, it must never change the produced masks (and therefore never the mask mAP).
"""

import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr.models.postprocess import PostProcess


def _segmentation_outputs(
    batch: int, queries: int, classes: int, mask_h: int, mask_w: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Build a minimal raw-output dict for the segmentation postprocess path."""
    generator = torch.Generator().manual_seed(seed)
    return {
        "pred_logits": torch.randn(batch, queries, classes, generator=generator),
        "pred_boxes": torch.rand(batch, queries, 4, generator=generator),
        "pred_masks": torch.randn(batch, queries, mask_h, mask_w, generator=generator),
    }


def test_chunked_mask_upsample_matches_single_pass() -> None:
    """A tiny interpolate budget (forced chunking) yields masks identical to a single full-stack upsample."""
    outputs = _segmentation_outputs(batch=2, queries=25, classes=3, mask_h=16, mask_w=20)
    target_sizes = torch.tensor([[128, 96], [200, 160]], dtype=torch.int64)

    single_pass = PostProcess(num_select=25, mask_interpolate_budget_mb=4096)  # whole stack fits → one F.interpolate
    chunked = PostProcess(num_select=25, mask_interpolate_budget_mb=1)  # tiny budget → multiple chunks per image

    reference = single_pass(outputs, target_sizes)
    result = chunked(outputs, target_sizes)

    for ref_image, chunked_image in zip(reference, result):
        assert chunked_image["masks"].dtype == torch.bool
        assert chunked_image["masks"].shape == ref_image["masks"].shape
        assert torch.equal(chunked_image["masks"], ref_image["masks"])


def test_interpolate_and_binarize_masks_chunk_boundary() -> None:
    """Chunking with a non-divisor chunk size reproduces the reference interpolate-then-threshold exactly."""
    postprocess = PostProcess()
    masks = torch.randn(50, 12, 9, generator=torch.Generator().manual_seed(1))  # K=50, Hm=12, Wm=9
    reference = F.interpolate(masks.unsqueeze(1), size=(40, 33), mode="bilinear", align_corners=False) > 0.0

    postprocess._mask_interpolate_budget_bytes = 40 * 33 * 4 * 7  # chunk == 7, which does not divide 50
    out = postprocess._interpolate_and_binarize_masks(masks, 40, 33)

    assert out.shape == (50, 1, 40, 33)
    assert out.dtype == torch.bool
    assert torch.equal(out, reference)


def test_interpolate_and_binarize_masks_empty() -> None:
    """Zero detections upsample to an empty boolean mask tensor without error."""
    postprocess = PostProcess()
    out = postprocess._interpolate_and_binarize_masks(torch.zeros(0, 8, 8), 16, 16)
    assert out.shape == (0, 1, 16, 16)
    assert out.dtype == torch.bool


def test_native_masks_keep_head_resolution() -> None:
    """``native_masks=True`` keeps masks at the head's ``(Hm, Wm)`` instead of upsampling to ``target_sizes``."""
    outputs = _segmentation_outputs(batch=1, queries=6, classes=3, mask_h=16, mask_w=20)
    target_sizes = torch.tensor([[128, 96]], dtype=torch.int64)
    postprocess = PostProcess(num_select=6)

    native = postprocess(outputs, target_sizes, native_masks=True)[0]["masks"]
    full_res = postprocess(outputs, target_sizes, native_masks=False)[0]["masks"]

    assert native.shape == (6, 1, 16, 20)  # mask head resolution, not upsampled
    assert native.dtype == torch.bool
    assert full_res.shape == (6, 1, 128, 96)  # upsampled to target image size


def test_native_masks_equal_thresholded_logits() -> None:
    """Native-resolution masks are exactly the per-query logits thresholded at zero (no interpolation)."""
    outputs = _segmentation_outputs(batch=1, queries=3, classes=2, mask_h=8, mask_w=10)
    # num_select == queries and descending logits keep query order, so masks[k] == pred_masks[k] > 0.
    outputs["pred_logits"] = torch.tensor([[[5.0, -5.0], [4.0, -5.0], [3.0, -5.0]]])
    target_sizes = torch.tensor([[64, 80]], dtype=torch.int64)

    native = PostProcess(num_select=3)(outputs, target_sizes, native_masks=True)[0]["masks"]
    expected = (outputs["pred_masks"][0] > 0.0).unsqueeze(1)
    assert torch.equal(native, expected)
