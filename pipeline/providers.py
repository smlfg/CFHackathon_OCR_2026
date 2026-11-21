"""Model providers: one small surface per runtime.

`mlx`       -> scripts/mlx_server.py on the host (Metal). Fast path on Apple Silicon.
`ollama`    -> host Ollama, for models that only exist as GGUF.
`tesseract` -> local Tesseract, the non-optimized floor baseline (CPU, no weights).

The first two are addressed over HTTP, so the container never needs a GPU or model
weights. Tesseract runs in-process; it is fast enough per table (~0.2 s) and every
image it needs is on disk anyway.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from . import config


def _post(url: str, payload: dict, timeout: int | None = None) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout or config.REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode())


def _get(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


class ProviderError(RuntimeError):
    pass


class MlxProvider:
    """Talks to scripts/mlx_server.py (host process, Metal)."""

    name = "mlx"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or config.MLX_BASE_URL).rstrip("/")

    def healthy(self) -> bool:
        try:
            return bool(_get(f"{self.base_url}/health")["ok"])
        except (urllib.error.URLError, OSError, KeyError):
            return False

    def load(self, repo: str) -> dict:
        try:
            return _post(f"{self.base_url}/load", {"repo": repo}, timeout=900)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"mlx load {repo}: HTTP {exc.code} {exc.read()[:400]!r}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ProviderError(
                f"mlx server unreachable at {self.base_url}: {exc}\n"
                "start it on the host: uv run --with mlx-vlm --with fastapi --with uvicorn "
                "python scripts/mlx_server.py --port 8099"
            ) from exc

    def generate(self, image_png: bytes, prompt: str, *, max_tokens: int = 4096,
                 temperature: float = 0.0) -> tuple[str, dict]:
        payload = {
            "model": "current",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,"
                                          + base64.b64encode(image_png).decode()}},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        started = time.time()
        try:
            out = _post(f"{self.base_url}/v1/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"mlx generate: HTTP {exc.code} {exc.read()[:400]!r}") from exc
        text = out["choices"][0]["message"]["content"]
        usage = dict(out.get("usage") or {})
        usage["seconds"] = round(time.time() - started, 2)
        return text, usage

    def preload(self, repo: str) -> dict:
        """Ask the server to load a repo into its second slot, ahead of time."""
        try:
            return _post(f"{self.base_url}/preload", {"repo": repo}, timeout=30)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise ProviderError(f"mlx preload {repo}: {exc}") from exc

    def prefetch(self, repo: str) -> dict:
        """Ask the server to warm the HF cache for a repo we will load next."""
        try:
            return _post(f"{self.base_url}/prefetch", {"repo": repo}, timeout=30)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise ProviderError(f"mlx prefetch {repo}: {exc}") from exc

    def prune(self, repo: str, delete_weights: bool = True) -> dict:
        return _post(f"{self.base_url}/prune",
                     {"repo": repo, "delete_weights": delete_weights}, timeout=120)


class OllamaProvider:
    """Host Ollama. Also pulls the model if it is not there yet."""

    name = "ollama"

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self._model: str | None = None

    def healthy(self) -> bool:
        try:
            return bool(_get(f"{self.base_url}/api/version"))
        except (urllib.error.URLError, OSError):
            return False

    def load(self, repo: str) -> dict:
        self._model = repo
        tags = _get(f"{self.base_url}/api/tags")
        names = {m["name"] for m in tags.get("models", [])}
        if repo in names or f"{repo}:latest" in names:
            return {"loaded": repo, "pulled": False}
        started = time.time()
        try:
            _post(f"{self.base_url}/api/pull", {"model": repo, "stream": False}, timeout=3600)
        except (urllib.error.URLError, OSError) as exc:
            raise ProviderError(f"ollama pull {repo} failed: {exc}") from exc
        return {"loaded": repo, "pulled": True, "seconds": round(time.time() - started, 1)}

    def generate(self, image_png: bytes, prompt: str, *, max_tokens: int = 4096,
                 temperature: float = 0.0) -> tuple[str, dict]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(image_png).decode()]}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.time()
        try:
            out = _post(f"{self.base_url}/api/chat", payload)
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"ollama generate: HTTP {exc.code} {exc.read()[:400]!r}") from exc
        usage = {
            "prompt_tokens": out.get("prompt_eval_count", 0),
            "completion_tokens": out.get("eval_count", 0),
            "seconds": round(time.time() - started, 2),
        }
        return out["message"]["content"], usage


class TesseractProvider:
    """Local Tesseract + the naive row/column clustering of the floor baseline.

    Deliberately not optimized: English model, default page segmentation, no
    deskew/denoise/upscale, no span detection. It exists so every model on the
    board has a measurable floor.
    """

    name = "tesseract"
    base_url = "local"

    def healthy(self) -> bool:
        return shutil.which("tesseract") is not None

    def version(self) -> str:
        try:
            out = subprocess.run(["tesseract", "--version"], capture_output=True,
                                 text=True, timeout=10)
            return out.stdout.splitlines()[0].strip() if out.stdout else "tesseract"
        except (OSError, subprocess.SubprocessError):
            return "tesseract"

    def load(self, repo: str) -> dict:
        if not self.healthy():
            raise ProviderError(
                "tesseract not on PATH (macOS: brew install tesseract, "
                "Debian: apt install tesseract-ocr)")
        return {"loaded": self.version(), "pulled": False}

    def generate(self, image_png: bytes, prompt: str, *, max_tokens: int = 4096,
                 temperature: float = 0.0) -> tuple[str, dict]:
        from PIL import Image

        from baselines.baseline_tesseract import to_html_image

        started = time.time()
        image = Image.open(io.BytesIO(image_png))
        try:
            text = to_html_image(image)
        except Exception as exc:  # binary missing, unreadable image, ...
            raise ProviderError(f"tesseract: {exc}") from exc
        return text, {"seconds": round(time.time() - started, 2)}


def get(runtime: str, **kwargs):
    if runtime == "mlx":
        return MlxProvider(kwargs.get("base_url"))
    if runtime == "ollama":
        return OllamaProvider(kwargs.get("base_url"))
    if runtime == "tesseract":
        return TesseractProvider()
    raise ProviderError(f"unknown runtime '{runtime}' (expected mlx, ollama or tesseract)")
