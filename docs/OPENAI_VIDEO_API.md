# OpenAI-Compatible Video API

WanGP includes a standalone HTTP server for clients that use the OpenAI-style asynchronous video API. It uses WanGP's existing Python generation API and model/settings pipeline.

## Start the server

From the WanGP installation directory, run:

```powershell
python -m shared.openai_video_api --port 8000 --root C:\Wan2GP
```

On localhost, WanGP generates a temporary bearer token and prints it once. Set `WANGP_API_KEY` to choose a persistent key. When binding beyond localhost, such as with `--host 0.0.0.0`, a key must be configured:

```powershell
$env:WANGP_API_KEY = "use-a-long-private-key"
python -m shared.openai_video_api --host 0.0.0.0 --port 8000 --root C:\Wan2GP
```

Keep the API on localhost or a trusted private network unless it is protected by HTTPS and network controls. `--root` selects the WanGP installation; it can be omitted when launched from that directory. Additional WanGP startup flags, such as `--profile 4`, may be passed to the API process.

The API server is a separate generation process and should not run at the same time as another WanGP process using the same GPU.

## Endpoints

All endpoints require `Authorization: Bearer <api-key>`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/v1/models` | List WanGP models whose primary output is video |
| `POST` | `/v1/videos` | Queue a video generation request (multipart form) |
| `GET` | `/v1/videos/{video_id}` | Read queued, running, completed, failed, or cancelled status |
| `GET` | `/v1/videos/{video_id}/content` | Download a completed video |
| `GET` | `/v1/videos` | List recent API jobs; accepts `limit` (1–100) and `after` |
| `DELETE` | `/v1/videos/{video_id}` | Cancel a queued or active generation |

The request's `model` is a WanGP model ID from `/v1/models`. `prompt` is required. Optional `seconds` (1–300) and `size` (for example `1280x720`) map to WanGP's duration and resolution settings. `input_reference` accepts a PNG, JPEG, WebP, or BMP image for image-to-video models, up to 32 MiB. Omitted optional values use the selected model's WanGP defaults. Requests queue FIFO; at most 100 jobs may wait in the queue.

```powershell
$headers = @{ Authorization = "Bearer $env:WANGP_API_KEY" }
$video = curl.exe -sS -H "Authorization: Bearer $env:WANGP_API_KEY" `
    -F "model=ltx2_22B_distilled" `
    -F "prompt=A cinematic rainy street at night" `
    -F "seconds=4" `
    -F "size=1280x720" `
    http://127.0.0.1:8000/v1/videos | ConvertFrom-Json

do {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/videos/$($video.id)" -Headers $headers
} while ($status.status -in @("queued", "in_progress"))

if ($status.status -eq "completed") {
    Invoke-WebRequest -Uri "http://127.0.0.1:8000/v1/videos/$($video.id)/content" -Headers $headers -OutFile generated.mp4
} else {
    $status.error.message
}
```

For image-to-video, send `input_reference` as a multipart file together with the other fields. Generation is asynchronous: creation returns HTTP `202` and a video ID, and the content endpoint returns HTTP `409` until the video is ready. API job records are held in memory and are cleared when the server stops; generated video files remain in the configured output directory.

The API implements the OpenAI-style video job workflow and bearer authorization, not every OpenAI endpoint or model identifier. It does not use an OpenAI service or API key.
