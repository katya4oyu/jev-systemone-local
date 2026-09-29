import httpx
import pytest
from fastapi.testclient import TestClient

from jev_systemone_local.app import create_app
from jev_systemone_local.extra_models import load_proxy_backends, parse_proxy_spec
from jev_systemone_local.proxy_backend import ProxyBackend, UpstreamError
from test_app import FakeBackend, request


def proxy(handler, name="kev-4b", upstream="kev-latest"):
    client = httpx.Client(base_url="http://upstream.test", transport=httpx.MockTransport(handler))
    return ProxyBackend(name, "http://upstream.test", upstream, client=client)


def ok(request_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"model": "kev-latest", "latency_ms": 12, "usage": {"input_tokens": 5, "output_tokens": 9},
                                     "answers": {"yes": {"type": "noul", "noul": 0.75}}})


def test_parse_proxy_spec_defaults_and_errors():
    assert parse_proxy_spec("kev-4b=http://127.0.0.1:8009, jeff-2b=http://h:8765|jeff-qwen3.5-2b", set()) == {
        "kev-4b": ("http://127.0.0.1:8009", "kev-latest"), "jeff-2b": ("http://h:8765", "jeff-qwen3.5-2b")}
    for bad, taken in [("kev-4b", set()), ("kev-4b=127.0.0.1:8009", set()), ("=http://x", set()),
                       ("a=http://x,a=http://y", set()), ("jev-latest=http://x", set()), ("used=http://x", {"used"})]:
        with pytest.raises(ValueError):
            parse_proxy_spec(bad, taken)


def test_forwards_state_questions_and_the_upstream_model_and_renames_the_answer():
    seen = {}

    def handler(request_):
        seen.update(request_.url.path and {"path": request_.url.path, "body": request_.read()})
        return ok(request_)

    backend = proxy(handler)
    result = backend.evaluate("こんにちは", {"yes": {"type": "noul", "instructions": "挨拶ですか？"}})
    assert seen["path"] == "/v1/systemone"
    assert b'"model":"kev-latest"' in seen["body"].replace(b" ", b"")
    assert result == {"model": "kev-4b", "answers": {"yes": {"type": "noul", "noul": 0.75}},
                      "usage": {"input_tokens": 5, "output_tokens": 9}}
    assert backend.checkpoint == "proxy:upstream.test/kev-latest" and not backend.needs_lock


@pytest.mark.parametrize("response,expected_status", [
    (httpx.Response(503, json={"detail": "The model is busy. Retry shortly."}), 503),
    (httpx.Response(500, text="boom"), 502),
    (httpx.Response(200, json={"unexpected": True}), 502),
])
def test_upstream_failures_are_explicit_errors(response, expected_status):
    with pytest.raises(UpstreamError) as info:
        proxy(lambda r: response).evaluate("x", {"yes": {"type": "noul", "instructions": "?"}})
    assert info.value.status_code == expected_status


def test_unreachable_and_timeout_upstreams():
    def refuse(request_):
        raise httpx.ConnectError("refused")

    def slow(request_):
        raise httpx.ReadTimeout("slow")

    assert pytest.raises(UpstreamError, proxy(refuse).evaluate, "x", {}).value.status_code == 502
    assert pytest.raises(UpstreamError, proxy(slow).evaluate, "x", {}).value.status_code == 503


def test_upstream_rejection_of_the_request_is_a_validation_error():
    backend = proxy(lambda r: httpx.Response(422, json={"detail": "at most 26 options"}))
    with pytest.raises(ValueError, match="26 options"):
        backend.evaluate("x", {})


def test_load_proxy_backends_warns_but_does_not_fail_for_a_down_upstream():
    warnings = []

    def factory(name, url, upstream):
        return proxy(lambda r: httpx.Response(500), name=name, upstream=upstream)

    backends = load_proxy_backends(set(), factory, spec="kev-4b=http://127.0.0.1:1", warn=warnings.append)
    assert set(backends) == {"kev-4b"} and "not answering yet" in warnings[0]


def test_http_routes_to_the_proxy_and_reports_upstream_errors():
    state = {"mode": "ok"}

    def handler(request_):
        if state["mode"] == "down":
            raise httpx.ConnectError("refused")
        if state["mode"] == "reject":
            return httpx.Response(422, json={"detail": "too many options"})
        return ok(request_)

    backends = {"laya-multilingual-mlx": FakeBackend(), "kev-4b": proxy(handler)}
    with TestClient(create_app(backends=backends)) as client:
        assert client.post("/v1/systemone", json=request(model="kev-4b")).json()["model"] == "kev-4b"
        assert client.post("/v1/systemone", json=request(model="jev-latest")).json()["model"] == "laya-multilingual-mlx"
        entry = next(m for m in client.get("/v1/models").json()["models"] if m["name"] == "kev-4b")
        assert entry["backend"] == "proxy" and entry["checkpoint"] == "proxy:upstream.test/kev-latest"
        assert "separate Jev-compatible server" in entry["description"]
        state["mode"] = "down"
        assert client.post("/v1/systemone", json=request(model="kev-4b")).status_code == 502  # no silent fallback
        state["mode"] = "reject"
        assert client.post("/v1/systemone", json=request(model="kev-4b")).status_code == 422
        # a registered proxy counts as available for the recommendation
        body = client.post("/v1/recommend", json={"text": "x", "task": "classify", "available_only": True, "limit": 7}).json()
        assert "kev-4b" in [r["model"] for r in body["recommendations"]]
        assert next(r for r in body["recommendations"] if r["model"] == "kev-4b")["how_to_use"] == "model に kev-4b を指定"
