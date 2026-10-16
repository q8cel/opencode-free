import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opencode_free.gate import READ_TOOL, USER_AGENT, session_id
from opencode_free.proxy import ProxyHandler

JSON_HEADERS = {"content-type": "application/json"}


class FixtureUpstream(BaseHTTPRequestHandler):
    """Records what the proxy sends upstream and answers with a canned payload."""

    received: list[dict] = []
    status = 200
    content_type = "application/json"
    payload = b'{"object":"chat.completion","choices":[]}'

    def do_GET(self) -> None:
        self.record(b"")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.record(self.rfile.read(length))

    def record(self, body: bytes) -> None:
        type(self).received.append(
            {
                "path": self.path,
                "headers": {name.lower(): value for name, value in self.headers.items()},
                "body": body,
                "method": self.command,
            },
        )
        self.send_response(type(self).status)
        self.send_header("content-type", type(self).content_type)
        self.send_header("content-length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, format: str, *args) -> None:
        pass


class ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FixtureUpstream)
        threading.Thread(target=cls.upstream.serve_forever, daemon=True).start()
        ProxyHandler.upstream = f"http://127.0.0.1:{cls.upstream.server_port}/v1"
        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.proxy.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.proxy.shutdown()
        cls.upstream.shutdown()

    def setUp(self) -> None:
        FixtureUpstream.received.clear()
        FixtureUpstream.status = 200
        FixtureUpstream.content_type = "application/json"
        FixtureUpstream.payload = b'{"object":"chat.completion","choices":[]}'

    def send(self, path: str, payload: dict | None = None, headers: dict | None = None) -> tuple[int, bytes, dict]:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers or JSON_HEADERS,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)

    def test_post_forwards_with_harness_gates(self) -> None:
        status, body, headers = self.send(
            "/v1/chat/completions",
            {"model": "big-pickle", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, FixtureUpstream.payload)
        self.assertEqual(headers["Content-Type"], "application/json")
        upstream_headers = {name.lower(): value for name, value in FixtureUpstream.received[0]["headers"].items()}
        self.assertEqual(upstream_headers["user-agent"], USER_AGENT)
        session = upstream_headers["x-opencode-session"]
        self.assertEqual(len(session), 30)
        self.assertTrue(session.startswith("ses_"))
        forwarded = FixtureUpstream.received[0]
        self.assertEqual(forwarded["path"], "/v1/chat/completions")
        tools = json.loads(forwarded["body"])["tools"]
        self.assertEqual(tools, [READ_TOOL])

    def test_existing_read_tool_not_duplicated(self) -> None:
        self.send("/v1/chat/completions", {"model": "m", "tools": [READ_TOOL]})
        tools = json.loads(FixtureUpstream.received[0]["body"])["tools"]
        self.assertEqual(tools, [READ_TOOL])

    def test_stream_forced_upstream(self) -> None:
        for client_stream in (True, False):
            with self.subTest(client_stream=client_stream):
                FixtureUpstream.received.clear()
                self.send("/v1/chat/completions", {"model": "m", "stream": client_stream})
                forwarded = json.loads(FixtureUpstream.received[0]["body"])
                self.assertTrue(forwarded["stream"])
                self.assertEqual(forwarded["tools"], [READ_TOOL])

    def test_non_stream_client_gets_aggregated_json(self) -> None:
        FixtureUpstream.content_type = "text/event-stream; charset=utf-8"
        FixtureUpstream.payload = b"\n".join(
            [
                b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"big-pickle","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}',
                b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"big-pickle","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}',
                b'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"big-pickle","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                b"data: [DONE]",
                b"",
            ]
        )
        status, body, headers = self.send("/v1/chat/completions", {"model": "big-pickle", "stream": False})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        completion = json.loads(body)
        self.assertEqual(completion["object"], "chat.completion")
        self.assertEqual(completion["model"], "big-pickle")
        self.assertEqual(completion["choices"][0]["message"]["content"], "hello")
        self.assertEqual(completion["choices"][0]["finish_reason"], "stop")
        self.assertEqual(completion["usage"]["total_tokens"], 5)

    def test_stream_client_gets_sse_passthrough(self) -> None:
        FixtureUpstream.content_type = "text/event-stream; charset=utf-8"
        FixtureUpstream.payload = b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        status, body, headers = self.send("/v1/chat/completions", {"model": "big-pickle", "stream": True})
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        self.assertEqual(body, FixtureUpstream.payload)

    def test_get_models_passthrough(self) -> None:
        status, body, _ = self.send("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body, FixtureUpstream.payload)
        self.assertEqual(FixtureUpstream.received[0]["path"], "/v1/models")

    def test_client_authorization_forwarded_when_present(self) -> None:
        self.send("/v1/chat/completions", {"model": "m"}, headers={**JSON_HEADERS, "authorization": "Bearer key"})
        headers = {name.lower(): value for name, value in FixtureUpstream.received[0]["headers"].items()}
        self.assertEqual(headers["authorization"], "Bearer key")

    def test_no_authorization_added_when_absent(self) -> None:
        self.send("/v1/chat/completions", {"model": "m"})
        headers = {name.lower(): value for name, value in FixtureUpstream.received[0]["headers"].items()}
        self.assertNotIn("authorization", headers)

    def test_upstream_error_status_forwarded(self) -> None:
        FixtureUpstream.status = 403
        FixtureUpstream.payload = b'{"type":"error","error":{"type":"FreeTierError"}}'
        status, body, _ = self.send("/v1/chat/completions", {"model": "m"})
        self.assertEqual(status, 403)
        self.assertIn(b"FreeTierError", body)

    def test_forged_session_shape(self) -> None:
        # The proxy forges a fresh id per request; validate one against the known-good shape.
        session = session_id()
        self.assertEqual(len(session), 30)
        int(session[4:16], 16)


if __name__ == "__main__":
    unittest.main()
