"""FastAPI application: upload, run control, live progress, results.

Runs on the Orin; the UI is opened from a laptop browser at
http://<orin-ip>:8000.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import REPO_ROOT, AppConfig, load_config
from .jobs import QUEUED, JobManager
from .models.registry import PAIRINGS, jetson_backends_available, resolve_backend
from .pipeline import RunConfig

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
WEB_DIR = REPO_ROOT / "web"


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    manager = JobManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The worker thread publishes into this loop, so it must be bound
        # before any job can run.
        manager.events.bind_loop(asyncio.get_running_loop())
        manager.start()
        yield

    app = FastAPI(title="Jetson Ablation Benchmark", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.manager = manager

    # --- meta --------------------------------------------------------------
    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        backend = resolve_backend(config)
        return {
            "backend": backend,
            "is_mock": backend == "mock",
            "jetson_libraries_present": jetson_backends_available(),
            "pairings": [
                {
                    "id": p.pairing_id,
                    "label": p.label,
                    "detector": p.detector,
                    "segmenter": p.segmenter,
                    "description": p.description,
                }
                for p in PAIRINGS
            ],
            "defaults": config.run.model_dump(),
            "lifecycle": config.lifecycle.model_dump(),
            "video_enabled": config.video.enabled,
            "max_upload_mb": config.server.max_upload_mb,
        }

    # --- jobs --------------------------------------------------------------
    @app.get("/api/jobs")
    async def list_jobs() -> dict[str, Any]:
        return {"jobs": [j.to_dict() for j in manager.list()]}

    @app.post("/api/jobs")
    async def create_job(
        video: UploadFile,
        prompts: str = Form(""),
        threshold: float = Form(None),
        stride: int = Form(None),
        max_frames: str = Form(""),
        warmup_frames: int = Form(None),
        record_masks: str = Form("true"),
    ) -> dict[str, Any]:
        suffix = Path(video.filename or "input.mp4").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                400,
                f"Unsupported file type {suffix!r}. Allowed: "
                + ", ".join(sorted(ALLOWED_SUFFIXES)),
            )

        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        if not prompt_list:
            prompt_list = list(config.run.prompts)

        parsed_max: int | None = None
        if max_frames.strip():
            try:
                parsed_max = max(1, int(max_frames))
            except ValueError:
                raise HTTPException(400, "max_frames must be a whole number.")

        run = RunConfig(
            prompts=prompt_list,
            threshold=config.run.threshold if threshold is None else float(threshold),
            stride=max(1, config.run.stride if stride is None else int(stride)),
            max_frames=parsed_max,
            warmup_frames=max(
                0, config.run.warmup_frames if warmup_frames is None else int(warmup_frames)
            ),
            record_masks=str(record_masks).lower() in {"1", "true", "yes", "on"},
        )

        job, target = manager.create(video.filename or "input.mp4", suffix, run)
        limit = config.server.max_upload_mb * 1024 * 1024
        written = 0
        try:
            with open(target, "wb") as handle:
                while chunk := await video.read(1 << 20):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            413, f"Upload exceeds the {config.server.max_upload_mb} MB limit."
                        )
                    handle.write(chunk)
        except HTTPException:
            manager.delete(job.job_id)
            raise
        finally:
            await video.close()

        if written == 0:
            manager.delete(job.job_id)
            raise HTTPException(400, "The uploaded file is empty.")

        job.state = QUEUED
        manager.enqueue(job.job_id)
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        return job.to_dict()

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, Any]:
        if not manager.delete(job_id):
            raise HTTPException(409, "Cannot delete a running or unknown job.")
        return {"deleted": job_id}

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        if not manager.cancel(job_id):
            raise HTTPException(409, "Job is not cancellable.")
        return {"cancelling": job_id}

    @app.get("/api/jobs/{job_id}/results")
    async def get_results(job_id: str) -> JSONResponse:
        results = manager.results(job_id)
        if results is None:
            raise HTTPException(404, "Results are not available yet.")
        return JSONResponse(results)

    @app.get("/api/jobs/{job_id}/frames")
    async def get_frames(job_id: str, pairing: str = "a") -> dict[str, Any]:
        if pairing not in {p.pairing_id for p in PAIRINGS}:
            raise HTTPException(400, "Unknown pairing.")
        return {"pairing": pairing, "frames": manager.frames(job_id, pairing)}

    @app.get("/api/jobs/{job_id}/export.csv")
    async def export_csv(job_id: str) -> StreamingResponse:
        results = manager.results(job_id)
        if results is None:
            raise HTTPException(404, "Results are not available yet.")

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "pairing",
                "pairing_label",
                "seq",
                "frame_index",
                "warmup",
                "pipeline_ms",
                "e2e_ms",
                "decode_ms",
                "preprocess_ms",
                "detect_ms",
                "detect_encode_ms",
                "detect_decode_ms",
                "seg_encode_ms",
                "seg_decode_ms",
                "postprocess_ms",
                "num_detections",
            ]
        )
        for pairing in PAIRINGS:
            label = results["pairings"][pairing.pairing_id]["label"]
            for frame in manager.frames(job_id, pairing.pairing_id):
                stages = frame.get("stages", {})
                writer.writerow(
                    [
                        pairing.pairing_id,
                        label,
                        frame.get("seq"),
                        frame.get("frame_index"),
                        frame.get("warmup"),
                        frame.get("pipeline_ms"),
                        frame.get("e2e_ms"),
                        stages.get("decode", 0),
                        stages.get("preprocess", 0),
                        stages.get("detect", 0),
                        stages.get("detect_encode", 0),
                        stages.get("detect_decode", 0),
                        stages.get("seg_encode", 0),
                        stages.get("seg_decode", 0),
                        stages.get("postprocess", 0),
                        frame.get("num_detections"),
                    ]
                )
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="benchmark_{job_id}.csv"'
            },
        )

    @app.get("/api/jobs/{job_id}/results.json")
    async def download_results(job_id: str) -> FileResponse:
        path = manager.paths(job_id).results_json
        if not path.exists():
            raise HTTPException(404, "Results are not available yet.")
        return FileResponse(
            path, media_type="application/json", filename=f"results_{job_id}.json"
        )

    @app.get("/api/jobs/{job_id}/video")
    async def get_video(job_id: str, request: Request) -> Any:
        path = manager.paths(job_id).video
        if not path.exists():
            raise HTTPException(404, "The comparison video is not available.")
        return _ranged_file_response(path, request)

    # --- live progress -----------------------------------------------------
    @app.websocket("/ws/jobs/{job_id}")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        job = manager.get(job_id)
        if job is None:
            await websocket.send_json({"type": "error", "message": "No such job."})
            await websocket.close()
            return

        # Send current state immediately so a reconnect is never blank.
        await websocket.send_json({"type": "job", "job": job.to_dict()})
        queue_ = manager.events.subscribe(job_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue_.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                    continue
                await websocket.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            manager.events.unsubscribe(job_id, queue_)

    # --- static UI ---------------------------------------------------------
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


def _ranged_file_response(path: Path, request: Request) -> Any:
    """Serve video with Range support so the browser can seek."""
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header or not range_header.startswith("bytes="):
        return FileResponse(path, media_type="video/mp4")

    try:
        start_s, _, end_s = range_header[6:].partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return FileResponse(path, media_type="video/mp4")

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    length = end - start + 1

    def stream() -> Any:
        with open(path, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


app = create_app()


def main() -> None:
    import uvicorn

    config = load_config()
    uvicorn.run(
        "benchmark.server:app",
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
