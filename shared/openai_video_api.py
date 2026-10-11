"""OpenAI-style HTTP API for asynchronous WanGP video generation."""

from __future__ import annotations

import argparse
import mimetypes
import os
import queue
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MAX_REFERENCE_BYTES = 32 * 1024 * 1024
_MAX_QUEUED_JOBS = 100
_STOP = object()


class VideoQueueFullError(RuntimeError):
    pass


@dataclass
class VideoJob:
    id: str
    model: str
    prompt: str
    seconds: int | None
    size: str | None
    created_at: int
    status: str = "queued"
    progress: int = 0
    completed_at: int | None = None
    error: str | None = None
    output_path: str | None = None
    reference_path: str | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    generation_job: Any = None


class VideoJobManager:
    def __init__(self, session) -> None:
        self.session = session
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[VideoJob | object] = queue.Queue(maxsize=_MAX_QUEUED_JOBS)
        self._upload_dir = Path(tempfile.mkdtemp(prefix="wangp-openai-api-"))
        self._worker: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        with self._lock:
            if self._worker is not None:
                return
            self._worker = threading.Thread(target=self._run, name="wangp-openai-video-api", daemon=True)
            self._worker.start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job.status == "queued":
                    job.cancel_requested.set()
                    job.status = "cancelled"
                    job.completed_at = int(time.time())
                    self._remove_reference(job)
                elif job.status == "in_progress":
                    job.cancel_requested.set()
                    if job.generation_job is not None:
                        job.generation_job.cancel()
            worker = self._worker
        self._queue.put(_STOP)
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        shutil.rmtree(self._upload_dir, ignore_errors=True)

    def create(
        self,
        *,
        model: str,
        prompt: str,
        seconds: int | None,
        size: str | None,
        reference_path: str | None = None,
    ) -> VideoJob:
        model = model.strip()
        prompt = prompt.strip()
        if not model:
            raise ValueError("model must be a WanGP video model ID.")
        if not prompt:
            raise ValueError("prompt must not be empty.")
        if seconds is not None and not 1 <= seconds <= 300:
            raise ValueError("seconds must be between 1 and 300.")
        if size is not None and (len(size) > 32 or not size.strip()):
            raise ValueError("size must be a non-empty resolution such as 1280x720.")
        metadata = self.session.get_model_metadata(model)
        if metadata is None:
            raise ValueError(f"Unknown model: {model}")
        main_output = metadata.get("main_output", [])
        if isinstance(main_output, str):
            main_output = [main_output]
        if "video" not in main_output:
            raise ValueError(f"Model {model} does not generate video.")
        job = VideoJob(
            id=f"video_{uuid.uuid4().hex}",
            model=model,
            prompt=prompt,
            seconds=seconds,
            size=size,
            created_at=int(time.time()),
            reference_path=reference_path,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("The video API is shutting down.")
            try:
                self._queue.put_nowait(job)
            except queue.Full as error:
                raise VideoQueueFullError("The generation queue is full. Retry after a job finishes.") from error
            self._jobs[job.id] = job
        return job

    def get(self, video_id: str) -> VideoJob:
        with self._lock:
            job = self._jobs.get(video_id)
        if job is None:
            raise KeyError(video_id)
        return job

    def list(self, *, limit: int, after: str | None = None) -> tuple[list[VideoJob], str | None]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: (job.created_at, job.id), reverse=True)
        if after is not None:
            index = next((i for i, job in enumerate(jobs) if job.id == after), None)
            if index is None:
                raise KeyError(after)
            jobs = jobs[index + 1:]
        page = jobs[:limit]
        return page, page[-1].id if len(jobs) > len(page) and page else None

    def cancel(self, video_id: str) -> VideoJob:
        job = self.get(video_id)
        with self._lock:
            if job.status in {"queued", "in_progress"}:
                job.cancel_requested.set()
                if job.generation_job is not None:
                    job.generation_job.cancel()
                elif job.status == "queued":
                    job.status = "cancelled"
                    job.completed_at = int(time.time())
                    self._remove_reference(job)
        return job

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _STOP:
                return
            assert isinstance(job, VideoJob)
            with self._lock:
                if job.status == "cancelled":
                    continue
                job.status = "in_progress"
            try:
                settings = self.session.get_default_settings(job.model)
                settings["model_type"] = job.model
                settings["prompt"] = job.prompt
                if job.seconds is not None:
                    settings["video_length"] = f"{job.seconds}s"
                if job.size is not None:
                    settings["resolution"] = job.size
                if job.reference_path is not None:
                    settings["image_start"] = job.reference_path
                manager = self

                class Callbacks:
                    def on_progress(self, update) -> None:
                        with manager._lock:
                            job.progress = max(0, min(99, int(getattr(update, "progress", 0) or 0)))

                generation_job = self.session.submit_task(settings, callbacks=Callbacks())
                with self._lock:
                    job.generation_job = generation_job
                    if job.cancel_requested.is_set():
                        generation_job.cancel()
                while not generation_job.done:
                    if job.cancel_requested.is_set():
                        generation_job.cancel()
                    time.sleep(0.1)
                result = generation_job.result()
                with self._lock:
                    if job.cancel_requested.is_set() or result.cancelled:
                        job.status = "cancelled"
                    elif result.success:
                        output = self._find_video_output(result)
                        if output is None:
                            job.status = "failed"
                            job.error = "Generation succeeded but did not produce a video file."
                        else:
                            job.status = "completed"
                            job.progress = 100
                            job.output_path = str(output)
                    else:
                        job.status = "failed"
                        job.error = "; ".join(error.message for error in result.errors) or "Video generation failed."
            except Exception as error:
                with self._lock:
                    job.status = "cancelled" if job.cancel_requested.is_set() else "failed"
                    job.error = None if job.status == "cancelled" else str(error)
            finally:
                with self._lock:
                    job.completed_at = int(time.time())
                    job.generation_job = None
                    self._remove_reference(job)

    @staticmethod
    def _find_video_output(result) -> Path | None:
        for raw_path in result.generated_files:
            path = Path(raw_path)
            if path.suffix.lower() in _VIDEO_EXTENSIONS and path.is_file():
                return path.resolve()
        for artifact in result.artifacts:
            if artifact.media_type == "video" and artifact.path:
                path = Path(artifact.path)
                if path.is_file():
                    return path.resolve()
        return None

    def _remove_reference(self, job: VideoJob) -> None:
        if job.reference_path:
            Path(job.reference_path).unlink(missing_ok=True)
            job.reference_path = None


def video_object(job: VideoJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "object": "video",
        "created_at": job.created_at,
        "status": job.status,
        "completed_at": job.completed_at,
        "error": None if job.error is None else {"message": job.error},
        "model": job.model,
        "progress": job.progress,
        "seconds": None if job.seconds is None else str(job.seconds),
        "size": job.size,
    }


def create_app(session, *, api_key: str) -> FastAPI:
    if not api_key:
        raise ValueError("An API key is required.")
    manager = VideoJobManager(session)

    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        try:
            yield
        finally:
            await run_in_threadpool(manager.close)
            await run_in_threadpool(session.close)

    app = FastAPI(title="WanGP OpenAI-compatible Video API", version="1.0.0", lifespan=lifespan)
    app.state.video_jobs = manager

    @app.exception_handler(HTTPException)
    async def openai_http_error(request: Request, error: HTTPException):
        detail = error.detail
        if isinstance(detail, dict):
            message = str(detail.get("message", detail))
            error_type = str(detail.get("type", "invalid_request_error"))
        else:
            message = str(detail)
            error_type = "authentication_error" if error.status_code == 401 else "invalid_request_error"
        return JSONResponse(
            {"error": {"message": message, "type": error_type, "param": None, "code": None}},
            status_code=error.status_code,
            headers=error.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def openai_validation_error(request: Request, error: RequestValidationError):
        first_error = error.errors()[0] if error.errors() else {}
        location = first_error.get("loc", ())
        param = str(location[-1]) if location else None
        message = "Invalid request"
        if param:
            message = f"Invalid or missing value for {param}."
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}},
            status_code=422,
        )

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.casefold() != "bearer" or not secrets.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
            raise HTTPException(
                status_code=401,
                detail={"message": "Invalid or missing API key.", "type": "authentication_error"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def list_models():
        models = session.list_model_metadata(main_output="video")
        return {
            "object": "list",
            "data": [
                {
                    "id": model["model_type"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "wangp",
                }
                for model in models
            ],
        }

    @app.post("/v1/videos", status_code=202, dependencies=[Depends(authorize)])
    async def create_video(
        model: str = Form(...),
        prompt: str = Form(...),
        seconds: int | None = Form(default=None),
        size: str | None = Form(default=None),
        input_reference: UploadFile | None = File(default=None),
    ):
        reference_path = None
        if input_reference is not None:
            suffix = Path(input_reference.filename or "").suffix.lower()
            if suffix not in _IMAGE_EXTENSIONS:
                raise HTTPException(400, "input_reference must be a PNG, JPEG, WebP, or BMP image.")
            target = manager._upload_dir / f"{uuid.uuid4().hex}{suffix}"
            total = 0
            try:
                with target.open("wb") as output:
                    while chunk := await input_reference.read(1024 * 1024):
                        total += len(chunk)
                        if total > _MAX_REFERENCE_BYTES:
                            raise HTTPException(413, "input_reference exceeds the 32 MiB upload limit.")
                        output.write(chunk)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            finally:
                await input_reference.close()
            if total == 0:
                target.unlink(missing_ok=True)
                raise HTTPException(400, "input_reference must not be empty.")
            reference_path = str(target)
        try:
            job = await run_in_threadpool(
                manager.create,
                model=model,
                prompt=prompt,
                seconds=seconds,
                size=size,
                reference_path=reference_path,
            )
        except ValueError as error:
            if reference_path:
                Path(reference_path).unlink(missing_ok=True)
            raise HTTPException(400, {"message": str(error), "type": "invalid_request_error"}) from error
        except VideoQueueFullError as error:
            if reference_path:
                Path(reference_path).unlink(missing_ok=True)
            raise HTTPException(429, str(error), headers={"Retry-After": "5"}) from error
        except Exception:
            if reference_path:
                Path(reference_path).unlink(missing_ok=True)
            raise
        return video_object(job)

    @app.get("/v1/videos", dependencies=[Depends(authorize)])
    def list_videos(limit: int = 20, after: str | None = None):
        if not 1 <= limit <= 100:
            raise HTTPException(400, "limit must be between 1 and 100.")
        try:
            jobs, last_id = manager.list(limit=limit, after=after)
        except KeyError as error:
            raise HTTPException(404, "The cursor video was not found.") from error
        return {
            "object": "list",
            "data": [video_object(job) for job in jobs],
            "has_more": last_id is not None,
            "last_id": last_id,
        }

    @app.get("/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    def retrieve_video(video_id: str):
        try:
            return video_object(manager.get(video_id))
        except KeyError as error:
            raise HTTPException(404, "Video not found.") from error

    @app.get("/v1/videos/{video_id}/content", dependencies=[Depends(authorize)])
    def retrieve_video_content(video_id: str):
        try:
            job = manager.get(video_id)
        except KeyError as error:
            raise HTTPException(404, "Video not found.") from error
        if job.status != "completed" or not job.output_path:
            raise HTTPException(409, "Video content is not ready.")
        path = Path(job.output_path)
        if not path.is_file():
            raise HTTPException(404, "Generated video file is no longer available.")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4", filename=path.name)

    @app.delete("/v1/videos/{video_id}", dependencies=[Depends(authorize)])
    def cancel_video(video_id: str):
        try:
            job = manager.cancel(video_id)
        except KeyError as error:
            raise HTTPException(404, "Video not found.") from error
        return {"id": job.id, "object": "video.deleted", "deleted": job.status == "cancelled"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WanGP's OpenAI-compatible video generation API.")
    parser.add_argument("--host", default=os.getenv("WANGP_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WANGP_API_PORT", "8000")))
    parser.add_argument("--root", type=Path, default=None, help="WanGP installation directory (defaults to this checkout).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the configured output directory.")
    parser.add_argument("--api-key", default=os.getenv("WANGP_API_KEY"), help="Bearer token (or set WANGP_API_KEY).")
    args, cli_args = parser.parse_known_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535.")
    api_key = args.api_key or secrets.token_urlsafe(32)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.api_key:
        parser.error("Set WANGP_API_KEY or pass --api-key before binding the API to a network interface.")
    from shared.api import init

    session = init(
        root=args.root,
        output_dir=args.output_dir,
        cli_args=cli_args,
        console_output=True,
    )
    app = create_app(session, api_key=api_key)
    print(f"WanGP OpenAI-compatible video API: http://{args.host}:{args.port}/v1")
    if not args.api_key:
        print(f"Generated API key (shown once): {api_key}")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
