import asyncio
import contextvars
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from utu.tools.github_toolkit import GitHubToolkit


@pytest.fixture
def toolkit(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "offline-test-token")
    return GitHubToolkit()


@pytest.fixture
def repo_data():
    return {
        "name": "example",
        "owner": {"login": "owner"},
        "stargazers_count": 7,
        "forks_count": 2,
        "watchers_count": 7,
        "license": {"name": "MIT License"},
        "language": "Python",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-02-01T00:00:00Z",
        "description": "An example",
        "open_issues_count": 3,
        "html_url": "https://github.com/owner/example",
    }


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(data).encode()
    result.url = "https://api.github.com/repos/owner/example"
    return result


async def invoke(toolkit, through_dispatch):
    if through_dispatch:
        return await toolkit.call_tool("get_repo_info", {"github_url": "https://github.com/owner/example"})
    return await toolkit.get_repo_info("https://github.com/owner/example")


@pytest.mark.parametrize("through_dispatch", [False, True])
async def test_http_wait_allows_event_loop_progress(toolkit, repo_data, monkeypatch, through_dispatch):
    entered = threading.Event()
    release = threading.Event()
    progressed = asyncio.Event()
    headers = []
    payload = json.dumps(repo_data).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            headers.append(dict(self.headers))
            entered.set()
            release.wait(timeout=1)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    original_get = requests.get

    def local_get(url, **kwargs):
        assert url == "https://api.github.com/repos/owner/example"
        return original_get(f"http://127.0.0.1:{server.server_port}/", timeout=2, **kwargs)

    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", local_get)

    async def heartbeat():
        while not entered.is_set():
            await asyncio.sleep(0)
        progressed.set()
        release.set()

    observer = asyncio.create_task(heartbeat())
    try:
        result = await invoke(toolkit, through_dispatch)
        assert progressed.is_set(), "the HTTP wait blocked the event-loop heartbeat"
        assert result["name"] == "example"
        assert headers[0]["Authorization"] == "Bearer offline-test-token"
        assert headers[0]["X-GitHub-Api-Version"] == "2022-11-28"
    finally:
        release.set()
        observer.cancel()
        await asyncio.gather(observer, return_exceptions=True)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


async def test_response_fields_are_preserved(toolkit, repo_data, monkeypatch):
    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", lambda *a, **kw: response(repo_data))
    assert await invoke(toolkit, True) == {
        "name": "example",
        "owner": "owner",
        "star": 7,
        "fork": 2,
        "watcher": 7,
        "license": "MIT License",
        "language": "Python",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-02-01T00:00:00Z",
        "description": "An example",
        "open_issues_count": 3,
        "html_url": "https://github.com/owner/example",
    }


@pytest.mark.parametrize("license_value", [None, {}])
async def test_optional_response_fields(toolkit, repo_data, monkeypatch, license_value):
    repo_data["license"] = license_value
    del repo_data["language"]
    del repo_data["description"]
    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", lambda *a, **kw: response(repo_data))
    result = await invoke(toolkit, True)
    assert result["license"] == "Not specified"
    assert result["language"] == "Not specified"
    assert result["description"] == "No description"


@pytest.mark.parametrize("status", [403, 404, 500])
async def test_http_errors_are_reported(toolkit, repo_data, monkeypatch, status):
    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", lambda *a, **kw: response(repo_data, status))
    result = await invoke(toolkit, True)
    assert result["error"].startswith(f"Request error: {status}")


async def test_connection_error_is_reported(toolkit, monkeypatch):
    def offline_get(*args, **kwargs):
        raise requests.ConnectionError("offline failure")

    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", offline_get)
    assert await invoke(toolkit, True) == {"error": "Request error: offline failure"}


async def test_missing_response_field_is_reported(toolkit, repo_data, monkeypatch):
    del repo_data["owner"]
    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", lambda *a, **kw: response(repo_data))
    assert await invoke(toolkit, True) == {"error": "Failed to parse data: 'owner'"}


async def test_invalid_url_does_not_request(toolkit, monkeypatch):
    def unexpected_get(*args, **kwargs):
        pytest.fail("invalid repository URL must not make an HTTP request")

    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", unexpected_get)
    assert await toolkit.get_repo_info("https://github.com/owner") == {"error": "Invalid GitHub repository URL"}


async def test_request_receives_caller_context(toolkit, repo_data, monkeypatch):
    request_id = contextvars.ContextVar("github_test_request_id")
    token = request_id.set("request-123")

    def get_with_context(*args, **kwargs):
        assert request_id.get() == "request-123"
        return response(repo_data)

    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", get_with_context)
    try:
        assert (await invoke(toolkit, True))["name"] == "example"
    finally:
        request_id.reset(token)


async def test_cancelling_dispatch_does_not_wait_for_request(toolkit, repo_data, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_get(*args, **kwargs):
        entered.set()
        try:
            release.wait(timeout=1)
            return response(repo_data)
        finally:
            finished.set()

    monkeypatch.setattr("utu.tools.github_toolkit.requests.get", blocking_get)
    task = asyncio.create_task(invoke(toolkit, True))
    try:
        while not entered.is_set():
            await asyncio.sleep(0)
        assert not task.done(), "dispatch must yield while the synchronous request is pending"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 2)


def test_only_public_tool_is_registered(toolkit):
    assert list(toolkit.get_tools_map_func()) == ["get_repo_info"]
    tools = toolkit.get_tools_in_agents()
    assert [tool.name for tool in tools] == ["get_repo_info"]
    assert "github_url" in tools[0].params_json_schema["properties"]
