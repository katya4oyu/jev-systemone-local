"""Proxy to another Jev-compatible server (e.g. Jeff or Kev running in their own process).

The upstream keeps its own weights and runtime; this server only forwards `/v1/systemone` and reports the answer
under the name it was registered with. Nothing falls back to another model: an unreachable or failing upstream is
reported as an error (502/503), never answered by a different model.
"""

from __future__ import annotations

from typing import Any

import httpx


class UpstreamError(RuntimeError):
    """The upstream server could not answer. `status_code` is what this server should return (502 or 503)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProxyBackend:
    backend_name = "proxy"
    needs_lock = False  # the upstream is a separate process; it serialises its own inference

    def __init__(self, model_name: str, url: str, upstream_model: str, timeout: float = 30.0,
                 client: httpx.Client | None = None) -> None:
        self.model_name = model_name
        self.upstream_model = upstream_model
        self.url = url.rstrip("/")
        self.checkpoint = f"proxy:{httpx.URL(self.url).netloc.decode()}/{upstream_model}"
        self.description = f"Proxy to a separate Jev-compatible server ({upstream_model}); not served by this process"
        self.client = client or httpx.Client(base_url=self.url, timeout=timeout)

    def probe(self) -> str | None:
        """Return None when the upstream lists its models, otherwise a short reason. Used only for a startup warning."""
        try:
            self.client.get("/v1/models").raise_for_status()
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"[:160]
        return None

    def evaluate(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        try:
            response = self.client.post("/v1/systemone", json={
                "state": state, "model": self.upstream_model, "questions": questions})
        except httpx.TimeoutException as exc:
            raise UpstreamError(f"upstream for {self.model_name!r} timed out", 503) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream for {self.model_name!r} is unreachable ({type(exc).__name__})") from exc
        if response.status_code == 503:
            raise UpstreamError(f"upstream for {self.model_name!r} is busy: {_detail(response)}", 503)
        if 400 <= response.status_code < 500:
            raise ValueError(_detail(response))  # the request itself was rejected (e.g. too many options)
        if response.status_code != 200:
            raise UpstreamError(f"upstream for {self.model_name!r} returned {response.status_code}: {_detail(response)}")
        body = response.json()
        if "answers" not in body:
            raise UpstreamError(f"upstream for {self.model_name!r} returned an unexpected body")
        return {"model": self.model_name, "answers": body["answers"],
                "usage": body.get("usage", {"input_tokens": 0, "output_tokens": 0})}


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return str(detail)[:300]
