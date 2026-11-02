"""Text -> frames -> video.

1. An LLM turns the idea into a storyboard (one image prompt per frame).
2. Each prompt is sent to an open-source Stable Diffusion model hosted by Hugging Face
   (nothing runs on this machine).
3. The frames are cross-faded and encoded to an MP4 with imageio + ffmpeg.
"""
import io
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, List

import httpx
import imageio
import numpy as np
from huggingface_hub import InferenceClient
from PIL import Image

from . import llm
from .config import get_settings

OUTPUT_DIR = Path(__file__).parent / "outputs"
NEGATIVE = "text, watermark, caption, logo, blurry, low quality, deformed, extra limbs"
FILE_RE = re.compile(r"^(video\.mp4|frame_\d{2}\.png)$")

Progress = Callable[[str, int, int], None]


@lru_cache
def _client() -> InferenceClient:
    s = get_settings()
    return InferenceClient(provider=s.hf_provider, api_key=s.hf_token, timeout=180)


class FatalImageError(RuntimeError):
    """Bad credentials, no credit or unknown model: retrying will not help."""


def _cloudflare_image(prompt: str, seed: int) -> Image.Image:
    s = get_settings()
    url = (f"https://api.cloudflare.com/client/v4/accounts/{s.cf_account_id}"
           f"/ai/run/{s.cf_image_model}")
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {s.cf_api_token}"},
        json={"prompt": prompt, "negative_prompt": NEGATIVE, "width": s.video_width,
              "height": s.video_height, "num_steps": s.cf_steps, "seed": seed},
        timeout=180,
    )
    if r.headers.get("content-type", "").startswith("image/"):
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    try:  # errors come back as JSON: {"success": false, "errors": [{"code": ..., "message": ...}]}
        detail = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in r.json().get("errors", []))
    except Exception:
        detail = r.text[:200]
    msg = f"Cloudflare returned HTTP {r.status_code}. {detail}"
    if r.status_code in (400, 401, 403, 404):
        raise FatalImageError(msg)
    raise RuntimeError(msg)


def _huggingface_image(prompt: str, seed: int) -> Image.Image:
    s = get_settings()
    try:
        return _client().text_to_image(
            prompt, model=s.hf_image_model, width=s.video_width, height=s.video_height,
            negative_prompt=NEGATIVE, seed=seed,
        )
    except Exception as e:
        if any(code in str(e) for code in ("401", "402", "403", "404", "410")):
            raise FatalImageError(f"Hugging Face refused the request: {str(e)[:300]}") from e
        raise


def missing_credentials() -> str:
    """Empty string if the chosen image provider is configured, else what to set."""
    s = get_settings()
    if s.image_provider == "huggingface":
        return "" if s.hf_token else "HF_TOKEN is not set in .env."
    if s.image_provider == "cloudflare":
        return "" if (s.cf_account_id and s.cf_api_token) else \
            "CF_ACCOUNT_ID and CF_API_TOKEN are not set in .env (Cloudflare Workers AI)."
    return f"Unknown IMAGE_PROVIDER '{s.image_provider}' (use cloudflare or huggingface)."


def text_to_image(prompt: str, seed: int) -> Image.Image:
    """One frame from the hosted Stable Diffusion model. Retries on temporary failures."""
    generate = _huggingface_image if get_settings().image_provider == "huggingface" else _cloudflare_image
    last = None
    for attempt in range(3):
        try:
            return generate(prompt, seed)
        except FatalImageError as e:
            raise RuntimeError(str(e)) from e
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Image API failed after 3 tries: {last}")


def build_video(frames: List[Image.Image], out_path: Path, fps: int, hold_s: float) -> None:
    """Hold each frame for hold_s seconds and cross-fade into the next one."""
    w, h = frames[0].size
    arrays = [np.asarray(f.convert("RGB").resize((w, h))) for f in frames]
    hold = max(1, round(hold_s * fps))
    fade = max(1, round(min(0.75, hold_s) * fps))
    with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=16) as writer:
        for i, a in enumerate(arrays):
            for _ in range(hold):
                writer.append_data(a)
            if i + 1 < len(arrays):
                b = arrays[i + 1].astype(np.float32)
                a32 = a.astype(np.float32)
                for k in range(1, fade + 1):
                    t = k / (fade + 1)
                    writer.append_data(((1 - t) * a32 + t * b).astype(np.uint8))


def make_video(idea: str, n_frames: int, fps: int, seconds_per_frame: float,
               job_id: str, progress: Progress) -> dict:
    out = OUTPUT_DIR / job_id
    out.mkdir(parents=True, exist_ok=True)

    progress("Writing the storyboard", 0, n_frames)
    board = llm.storyboard_chain().invoke({"idea": idea, "n": n_frames})
    scenes = [s.strip() for s in board.scenes if s.strip()][:n_frames]
    if len(scenes) < 2:
        raise RuntimeError("The storyboard came back with fewer than 2 scenes; try rephrasing the idea")

    seed = random.randint(0, 2**31 - 1)  # one seed for every frame keeps the look consistent
    frames: List[Image.Image] = []
    for i, scene in enumerate(scenes):
        progress("Generating frame", i, len(scenes))
        img = text_to_image(f"{board.style} {scene}", seed)
        img.save(out / f"frame_{i + 1:02d}.png")
        frames.append(img)

    progress("Assembling the video", len(scenes), len(scenes))
    build_video(frames, out / "video.mp4", fps, seconds_per_frame)

    base = f"/video/files/{job_id}"
    return {
        "video": f"{base}/video.mp4",
        "frames": [f"{base}/frame_{i + 1:02d}.png" for i in range(len(scenes))],
        "scenes": scenes,
        "style": board.style,
    }
