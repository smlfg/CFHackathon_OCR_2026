"""Model providers: one small HTTP surface per runtime.

`mlx`    -> scripts/mlx_server.py on the host (Metal). Fast path on Apple Silicon.
`ollama` -> host Ollama, for models that only exist as GGUF.

Both are addressed over HTTP, so the container never needs a GPU or model weights.
"""

from __future__ import annotations

import base64
import json
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

    def generate(
        self, image_png: bytes, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.0
    ) -> tuple[str, dict]:
        payload = {
            "model": "current",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image_png).decode()
                            },
                        },
                    ],
                }
            ],
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

    def prune(self, repo: str, delete_weights: bool = True) -> dict:
        return _post(
            f"{self.base_url}/prune", {"repo": repo, "delete_weights": delete_weights}, timeout=120
        )


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

    def generate(
        self, image_png: bytes, prompt: str, *, max_tokens: int = 4096, temperature: float = 0.0
    ) -> tuple[str, dict]:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_png).decode()],
                }
            ],
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


def get(runtime: str, **kwargs):
    if runtime == "mlx":
        return MlxProvider(kwargs.get("base_url"))
    if runtime == "ollama":
        return OllamaProvider(kwargs.get("base_url"))
    raise ProviderError(f"unknown runtime '{runtime}' (expected mlx or ollama)")
