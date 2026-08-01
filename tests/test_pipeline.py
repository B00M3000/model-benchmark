"""Pipeline, timing and RLE behaviour, driven by the mock backends."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import rle
from benchmark.models.registry import PAIRINGS, build_detector, build_segmenter
from benchmark.pipeline import RunConfig, planned_frame_count, probe_video, run_pairing
from benchmark.storage import read_jsonl
from benchmark.timing import PIPELINE_STAGES, STAGE_DECODE, StageTimer


# ── timing ──────────────────────────────────────────────────────────────
def test_stage_timer_accumulates_repeated_stages():
    timer = StageTimer()
    for _ in range(3):
        with timer.stage("seg_decode", sync=False):
            pass
    # Three decodes in one frame must sum, not overwrite -- that is what
    # makes seg_decode the per-frame total across every detection.
    assert timer.marks["seg_decode"] > 0
    assert len([k for k in timer.marks if k == "seg_decode"]) == 1


def test_pipeline_ms_excludes_decode():
    timer = StageTimer()
    timer.add(STAGE_DECODE, 100.0)
    for stage in PIPELINE_STAGES:
        timer.add(stage, 1.0)

    assert timer.pipeline_ms() == pytest.approx(len(PIPELINE_STAGES) * 1.0)
    assert timer.e2e_ms() == pytest.approx(timer.pipeline_ms() + 100.0)


def test_timer_sync_is_invoked_for_gpu_stages():
    calls = []
    timer = StageTimer(sync=lambda: calls.append(1))
    with timer.stage("detect"):
        pass
    # Once before and once after -- without this, async CUDA work lands in
    # the next stage's measurement.
    assert len(calls) == 2

    with timer.stage("decode", sync=False):
        pass
    assert len(calls) == 2


# ── RLE ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rle_roundtrip(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((40, 55)) > 0.6
    assert np.array_equal(rle.decode(rle.encode(mask)), mask)


def test_rle_handles_all_set_and_all_clear():
    for mask in (np.ones((8, 9), bool), np.zeros((8, 9), bool)):
        assert np.array_equal(rle.decode(rle.encode(mask)), mask)


def test_rle_mask_starting_with_one():
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    encoded = rle.encode(mask)
    assert encoded["counts"][0] == 0  # leading empty zero-run
    assert np.array_equal(rle.decode(encoded), mask)


def test_rle_area_matches_count():
    rng = np.random.default_rng(3)
    mask = rng.random((30, 30)) > 0.5
    assert rle.area(rle.encode(mask)) == int(np.count_nonzero(mask))


# ── video probing ───────────────────────────────────────────────────────
def test_probe_video(synthetic_video):
    info = probe_video(synthetic_video)
    assert info.width == 320
    assert info.height == 240
    assert info.frame_count == 24


def test_planned_frame_count_respects_stride_and_cap(synthetic_video):
    info = probe_video(synthetic_video)
    assert planned_frame_count(info, RunConfig(stride=1)) == 24
    assert planned_frame_count(info, RunConfig(stride=3)) == 8
    assert planned_frame_count(info, RunConfig(stride=1, max_frames=5)) == 5


# ── the run itself ──────────────────────────────────────────────────────
def test_run_pairing_produces_records(synthetic_video, mock_config, tmp_path):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    frames_path = tmp_path / "frames_a.jsonl"
    masks_path = tmp_path / "masks_a.jsonl"
    run = RunConfig(prompts=["a box"], warmup_frames=2, record_masks=True)

    summary, records = run_pairing(
        synthetic_video, pairing, detector, segmenter, run,
        frames_path=frames_path, masks_path=masks_path,
    )

    assert len(records) == 24
    assert summary.frames_warmup == 2
    assert summary.frames_measured == 22
    assert summary.pipeline.p50 > 0
    assert summary.throughput_fps > 0

    # Every pipeline stage must be present and the total must be consistent.
    first = records[5]
    for stage in ("preprocess", "detect", "postprocess"):
        assert stage in first["stages"]
    total = sum(first["stages"].get(s, 0.0) for s in PIPELINE_STAGES)
    assert first["pipeline_ms"] == pytest.approx(total, abs=0.01)
    assert first["e2e_ms"] >= first["pipeline_ms"]

    # Records were persisted for the video render and the CSV export.
    persisted = list(read_jsonl(frames_path))
    assert len(persisted) == 24
    assert masks_path.exists()

    masks = list(read_jsonl(masks_path))
    assert masks, "masks should have been written by the background writer"
    decoded = rle.decode(masks[0]["masks"][0])
    assert decoded.shape == (240, 320)
    assert decoded.any()


def test_stride_and_max_frames_are_honoured(synthetic_video, mock_config):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    run = RunConfig(stride=4, max_frames=3, warmup_frames=0, record_masks=False)
    _, records = run_pairing(synthetic_video, pairing, detector, segmenter, run)

    assert len(records) == 3
    assert [r["frame_index"] for r in records] == [0, 4, 8]


def test_record_masks_false_writes_nothing(synthetic_video, mock_config, tmp_path):
    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    masks_path = tmp_path / "masks.jsonl"
    run = RunConfig(warmup_frames=0, record_masks=False, max_frames=4)
    run_pairing(synthetic_video, pairing, detector, segmenter, run, masks_path=masks_path)

    assert not masks_path.exists()


def test_cancellation_stops_the_run(synthetic_video, mock_config):
    import threading

    from benchmark.pipeline import CancelledError

    pairing = PAIRINGS[0]
    detector = build_detector(mock_config, "mock")
    detector.load()
    detector.set_prompts(["a box"], 0.1)
    segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
    segmenter.load()

    cancel = threading.Event()
    seen = []

    def on_progress(progress):
        seen.append(progress)
        if len(seen) >= 3:
            cancel.set()

    with pytest.raises(CancelledError):
        run_pairing(
            synthetic_video, pairing, detector, segmenter,
            RunConfig(warmup_frames=0, record_masks=False),
            progress=on_progress, cancel=cancel,
        )
    assert len(seen) < 24


def test_segmenters_report_distinct_costs(synthetic_video, mock_config):
    """The whole study rests on the segment stage differing; assert it does."""
    results = {}
    for pairing in PAIRINGS:
        detector = build_detector(mock_config, "mock")
        detector.load()
        detector.set_prompts(["a box"], 0.1)
        segmenter = build_segmenter(mock_config, "mock", pairing.segmenter)
        segmenter.load()
        summary, _ = run_pairing(
            synthetic_video, pairing, detector, segmenter,
            RunConfig(warmup_frames=2, record_masks=False),
        )
        results[pairing.pairing_id] = summary

    a_seg = results["a"].stages["seg_encode"].mean
    b_seg = results["b"].stages["seg_encode"].mean
    # Mock EfficientViT is configured to be the heavier encoder.
    assert b_seg > a_seg


# ── import-path escape hatch ────────────────────────────────────────────
def test_repo_paths_make_a_clone_importable(tmp_path, mock_config, monkeypatch):
    """A plain git clone must be usable without `pip install -e`.

    JetPack's setuptools/packaging pair frequently breaks editable installs,
    which must not block a benchmark run.
    """
    import sys

    from benchmark.models.registry import ensure_repo_paths, missing_jetson_modules

    clone = tmp_path / "fake_repo"
    (clone / "nanoowl").mkdir(parents=True)
    (clone / "nanoowl" / "__init__.py").write_text("")

    monkeypatch.setattr(sys, "path", list(sys.path))
    assert "nanoowl" in missing_jetson_modules(mock_config)

    mock_config.repo_paths = [str(clone)]
    added = ensure_repo_paths(mock_config)
    assert added == [str(clone)]
    assert "nanoowl" not in missing_jetson_modules(mock_config)

    # Idempotent: a second call must not stack duplicates onto sys.path.
    assert ensure_repo_paths(mock_config) == []
    assert sys.path.count(str(clone)) == 1


def test_repo_paths_handles_a_package_with_no_top_level_init(tmp_path, mock_config, monkeypatch):
    """repo_paths must work for a package shaped exactly like nanosam.

    NanoSAM's upstream repo ships with no nanosam/nanosam/__init__.py at all
    -- only its submodules (nanosam/tools/__init__.py, etc.) have one. A
    repo_paths entry pointed at the clone root therefore resolves nanosam
    itself as a PEP 420 namespace package even when everything is set up
    correctly, which looks identical -- spec.origin is None either way -- to
    the broken case where sys.path is pointed one level too high, at the
    clone root's *parent*, and the real content is a further level deeper
    still. module_status() must tell these apart by checking for real
    content, not just by whether origin is None.
    """
    import sys

    from benchmark.models.registry import ensure_repo_paths, missing_jetson_modules, module_status

    clone = tmp_path / "nanosam"  # the clone root; matches nanosam's own layout
    (clone / "nanosam" / "tools").mkdir(parents=True)
    (clone / "nanosam" / "tools" / "__init__.py").write_text("")
    # Deliberately no (clone / "nanosam" / "__init__.py") -- nanosam has none either.
    (clone / "setup.py").write_text("")  # present in every real clone; must not fool the check

    monkeypatch.setattr(sys, "path", list(sys.path))
    assert "nanosam" in missing_jetson_modules(mock_config)

    # Correct configuration: repo_paths points at the clone root itself.
    mock_config.repo_paths = [str(clone)]
    ensure_repo_paths(mock_config)
    assert module_status("nanosam")[0] == "ok"
    assert "nanosam" not in missing_jetson_modules(mock_config)


def test_repo_paths_still_catches_a_genuine_shadow(tmp_path, mock_config, monkeypatch):
    """The broken case the above test is contrasted with must still be caught.

    Pointing repo_paths one level too high -- at the clone root's parent,
    rather than the clone root itself -- must still be reported as not
    usable, even for a package shaped like nanosam (no top-level __init__.py).
    """
    import sys

    from benchmark.models.registry import ensure_repo_paths, module_status

    parent = tmp_path / "wrong_level"
    clone = parent / "nanosam"
    (clone / "nanosam" / "tools").mkdir(parents=True)
    (clone / "nanosam" / "tools" / "__init__.py").write_text("")
    (clone / "setup.py").write_text("")

    monkeypatch.setattr(sys, "path", list(sys.path))
    mock_config.repo_paths = [str(parent)]  # one level too high
    ensure_repo_paths(mock_config)
    assert module_status("nanosam")[0] == "shadowed"


def test_repo_paths_ignores_missing_directories(mock_config, monkeypatch):
    import sys

    from benchmark.models.registry import ensure_repo_paths

    monkeypatch.setattr(sys, "path", list(sys.path))
    mock_config.repo_paths = ["/nonexistent/path/to/nowhere"]
    assert ensure_repo_paths(mock_config) == []


def _trt_sam_predictor_with_stub_engines(frame_hw, embed=(64, 64), tokens=4):
    """Build a TrtSamPredictor whose 'engines' are cheap stand-ins.

    Exercises the geometry TrtSamPredictor owns -- resize/pad, the box ->
    prompt-frame coordinate transform, mask-token slicing and the upscale/
    crop/resize back to the source frame -- without loading TensorRT or the
    real SAM weights.
    """
    import torch

    import benchmark.models.trt_sam as trt_sam

    seen = {}

    def fake_encoder(tensor):
        seen["encoder_input"] = tuple(tensor.shape)
        return torch.zeros(1, 256, *embed)

    def fake_decoder(features, coords, labels):
        seen["coords"] = coords.clone()
        seen["labels"] = labels.clone()
        # Distinct per-token values so the slice below is observable.
        masks = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens, 1, 1)
        masks = masks.expand(1, tokens, 256, 256).clone() - 1.5
        iou = torch.arange(tokens, dtype=torch.float32).reshape(1, tokens)
        return masks, iou

    engines = {"enc": fake_encoder, "dec": fake_decoder}
    original = trt_sam._load_engine
    trt_sam._load_engine = lambda path, inputs, outputs: engines[path]
    try:
        predictor = trt_sam.TrtSamPredictor("enc", "dec", model="efficientvit-sam-l0", device="cpu")
    finally:
        trt_sam._load_engine = original
    return predictor, seen


def test_trt_sam_predictor_geometry():
    """The TensorRT path must return masks in the source frame's resolution."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("efficientvit")

    import numpy as np

    height, width = 480, 640
    predictor, seen = _trt_sam_predictor_with_stub_engines((height, width))

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    predictor.set_image(frame)

    # The encoder always sees the padded square, whatever the frame's aspect.
    assert seen["encoder_input"] == (1, 3, 512, 512)
    # Prompts live in a 1024-long-side frame, not the encoder's 512 one.
    assert predictor.input_size == (768, 1024)
    assert predictor.original_size == (height, width)

    box = (64.0, 48.0, 320.0, 240.0)
    result = predictor.predict(box=np.array(box), multimask_output=False)
    masks, iou, low_res = result

    # Box corners are scaled into the prompt frame and labelled 2 / 3.
    assert seen["coords"].shape == (1, 2, 2)
    assert seen["labels"].tolist() == [[2.0, 3.0]]
    np.testing.assert_allclose(
        seen["coords"][0].numpy(),
        [[box[0] * 1024 / width, box[1] * 768 / height],
         [box[2] * 1024 / width, box[3] * 768 / height]],
        rtol=1e-5,
    )

    # multimask_output=False takes mask token 0, matching MaskDecoder.forward.
    assert masks.shape == (1, height, width)
    assert low_res.shape == (1, 256, 256)
    assert iou.tolist() == [0.0]
    # Token 0's value is -1.5, i.e. below the 0.0 threshold -> all False.
    assert masks.dtype == bool and not masks.any()


def test_trt_sam_predictor_multimask_slices_tokens_one_onward():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("efficientvit")

    import numpy as np

    predictor, _ = _trt_sam_predictor_with_stub_engines((480, 640))
    predictor.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    masks, iou, _ = predictor.predict(box=np.array([1.0, 2.0, 3.0, 4.0]), multimask_output=True)

    assert masks.shape == (3, 480, 640)
    assert iou.tolist() == [1.0, 2.0, 3.0]


def test_trt_sam_predictor_rejects_unknown_model():
    pytest.importorskip("torch")

    import benchmark.models.trt_sam as trt_sam

    with pytest.raises(ValueError, match="Unknown EfficientViT-SAM model"):
        trt_sam.TrtSamPredictor("enc", "dec", model="efficientvit-sam-xxl")
