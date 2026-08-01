"""End-to-end API tests: upload, run both pairings, read results back."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from benchmark.jobs import COMPLETE, COMPLETE_NO_VIDEO, FAILED, TERMINAL_STATES
from benchmark.server import create_app


@pytest.fixture
def client(mock_config):
    app = create_app(mock_config)
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_terminal(client, job_id, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in TERMINAL_STATES:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _submit(client, video_path, **fields):
    payload = {
        "prompts": "a box",
        "threshold": "0.1",
        "stride": "3",
        "max_frames": "6",
        "warmup_frames": "1",
        "record_masks": "true",
    }
    payload.update(fields)
    with open(video_path, "rb") as handle:
        return client.post(
            "/api/jobs",
            files={"video": ("clip.mp4", handle, "video/mp4")},
            data=payload,
        )


def test_config_endpoint_reports_mock_backend(client):
    body = client.get("/api/config").json()
    assert body["backend"] == "mock"
    assert body["is_mock"] is True
    assert [p["id"] for p in body["pairings"]] == ["a", "b"]


def test_full_job_lifecycle(client, synthetic_video):
    response = _submit(client, synthetic_video)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]

    job = _wait_for_terminal(client, job_id)
    assert job["state"] in {COMPLETE, COMPLETE_NO_VIDEO}, job.get("error")
    assert job["has_results"] is True

    results = client.get(f"/api/jobs/{job_id}/results").json()
    assert set(results["pairings"]) == {"a", "b"}
    assert results["is_mock"] is True

    for pairing_id in ("a", "b"):
        summary = results["pairings"][pairing_id]
        assert summary["frames_measured"] == 5  # 6 processed, 1 warm-up
        assert summary["frames_warmup"] == 1
        assert summary["pipeline"]["p50"] > 0
        assert summary["throughput_fps"] > 0

    assert results["comparison"]["faster_pairing"] in {"a", "b"}
    assert len(results["series"]["a"]) == 6
    assert results["histograms"]["a"]["counts"]

    # Both pairings ran on exactly the same frames -- a precondition for
    # the comparison to mean anything.
    frames_a = client.get(f"/api/jobs/{job_id}/frames?pairing=a").json()["frames"]
    frames_b = client.get(f"/api/jobs/{job_id}/frames?pairing=b").json()["frames"]
    assert [f["frame_index"] for f in frames_a] == [f["frame_index"] for f in frames_b]


def test_detector_is_shared_but_runs_independently(client, synthetic_video):
    """Single process: NanoOWL loads once, yet each pairing detects itself."""
    job_id = _submit(client, synthetic_video).json()["job_id"]
    job = _wait_for_terminal(client, job_id)

    assert job["lifecycle"]["share_detector"] is True
    assert "detector_load_ms" in job["lifecycle"]

    results = client.get(f"/api/jobs/{job_id}/results").json()
    # Detection cost is charged to both runs, proving neither reused cached
    # boxes from the other.
    assert results["pairings"]["a"]["stages"]["detect"]["mean"] > 0
    assert results["pairings"]["b"]["stages"]["detect"]["mean"] > 0

    # ...yet the shared detector must carry no state between runs, so the
    # same frame yields the same boxes in both. Anything else would mean the
    # segmenters were compared on different inputs.
    frames_a = client.get(f"/api/jobs/{job_id}/frames?pairing=a").json()["frames"]
    frames_b = client.get(f"/api/jobs/{job_id}/frames?pairing=b").json()["frames"]
    for fa, fb in zip(frames_a, frames_b):
        assert fa["num_detections"] == fb["num_detections"]
        assert [d["box"] for d in fa["detections"]] == [d["box"] for d in fb["detections"]]


def test_csv_export_has_a_row_per_frame_per_pairing(client, synthetic_video):
    job_id = _submit(client, synthetic_video).json()["job_id"]
    _wait_for_terminal(client, job_id)

    response = client.get(f"/api/jobs/{job_id}/export.csv")
    assert response.status_code == 200
    lines = [line for line in response.text.strip().splitlines() if line]
    assert len(lines) == 1 + 6 * 2  # header + 6 frames x 2 pairings
    assert "pipeline_ms" in lines[0]
    assert "seg_decode_ms" in lines[0]


def test_comparison_video_is_rendered(client, synthetic_video):
    job_id = _submit(client, synthetic_video).json()["job_id"]
    job = _wait_for_terminal(client, job_id)

    if job["state"] == COMPLETE_NO_VIDEO:
        pytest.skip(f"video render unavailable here: {job.get('video_error')}")

    assert job["has_video"] is True
    response = client.get(f"/api/jobs/{job_id}/video")
    assert response.status_code == 200
    assert len(response.content) > 0

    # Which encoder ran is environment-dependent, but the job must always
    # say, and must warn exactly when the codec is one browsers can't play
    # -- otherwise an unopenable video looks like a successful render.
    assert job["video_encoder"]
    playable = b"avc1" in response.content
    assert (job["video_note"] is None) is playable


def test_websocket_streams_progress(client, synthetic_video):
    job_id = _submit(client, synthetic_video).json()["job_id"]

    seen_types = set()
    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        for _ in range(60):
            message = ws.receive_json()
            seen_types.add(message["type"])
            if message["type"] == "job" and message["job"]["is_terminal"]:
                break
    assert "job" in seen_types


def test_rejects_non_video_upload(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("not a video")
    with open(bad, "rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"video": ("notes.txt", handle, "text/plain")},
            data={"prompts": "a box"},
        )
    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_unreadable_video_fails_cleanly(client, tmp_path):
    """A corrupt upload must fail the job, not take the server down."""
    fake = tmp_path / "broken.mp4"
    fake.write_bytes(b"\x00\x01\x02 not really an mp4")
    with open(fake, "rb") as handle:
        response = client.post(
            "/api/jobs",
            files={"video": ("broken.mp4", handle, "video/mp4")},
            data={"prompts": "a box"},
        )
    job_id = response.json()["job_id"]
    job = _wait_for_terminal(client, job_id, timeout=30)
    assert job["state"] == FAILED
    assert job["error"]
    # Server still healthy.
    assert client.get("/api/config").status_code == 200


def test_missing_results_returns_404(client):
    assert client.get("/api/jobs/deadbeef/results").status_code == 404
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_prompts_default_when_blank(client, synthetic_video):
    response = _submit(client, synthetic_video, prompts="   ")
    job = response.json()
    assert job["run_config"]["prompts"] == ["a person"]
