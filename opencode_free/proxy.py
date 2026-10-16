"""Local proxy that makes any OpenAI-compatible client look like the opencode harness.

Point your harness at ``http://127.0.0.1:8901/v1`` and it can use OpenCode's
free-tier models: the proxy forwards every request to the inference upstream
while injecting the headers, the ``read`` tool and ``stream: true`` described in
:mod:`opencode_free.gate`. Responses (including SSE streams) are passed through
chunk by chunk, and SSE is aggregated back into JSON for non-streaming clients.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .gate import READ_TOOL, UPSTREAM, USER_AGENT, prepare_payload, request_headers
from . import __version__

JSON_TYPE = "application/json"


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = f"opencode-free/{__version__}"
    upstream = UPSTREAM
    user_agent = USER_AGENT
    forward_auth = False
    timeout = 300

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type, authorization, x-api-key")
        self.end_headers()

    def do_GET(self) -> None:
        self.forward(b"")

    def do_POST(self) -> None:
        body = self.read_body()
        if body is not None:
            self.forward(body)

    def do_PUT(self) -> None:
        body = self.read_body()
        if body is not None:
            self.forward(body)

    def do_PATCH(self) -> None:
        body = self.read_body()
        if body is not None:
            self.forward(body)

    def do_DELETE(self) -> None:
        body = self.read_body()
        if body is not None:
            self.forward(body)

    def read_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self.reply(
                400,
                {"error": {"type": "invalid_request_error", "message": "chunked request bodies are not supported"}},
            )
            return None
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def forward(self, body: bytes) -> None:
        path = self.path[3:] if self.path.startswith("/v1/") else "" if self.path == "/v1" else self.path
        payload, wants_stream = prepare_payload(body)
        headers = {"content-type": self.headers.get("Content-Type", JSON_TYPE), **request_headers(self.user_agent)}
        authorization = self.headers.get("Authorization")
        if authorization and self.forward_auth:
            headers["authorization"] = authorization
        if authorization and not self.forward_auth:
            # An unrecognized credential is a hard 401 upstream, while no credential
            # at all is free-tier anonymous, so client keys are dropped by default.
            print("dropping client Authorization: upstream stays keyless", file=sys.stderr)
        request = urllib.request.Request(
            self.upstream + path,
            data=payload or None,
            headers=headers,
            method=self.command,
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            response = error
        except urllib.error.URLError as error:
            self.reply(502, {"error": {"type": "api_error", "message": f"inference routing failed: {error.reason}"}})
            return
        with response:
            content_type = response.headers.get("Content-Type", JSON_TYPE)
            if not wants_stream and path.endswith("/chat/completions") and content_type.startswith("text/event-stream"):
                self.reply(response.status, aggregate_chat(response.read()))
                return
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            self._cors()
            self.end_headers()
            while chunk := response.read(65536):
                self.wfile.write(chunk)
                self.wfile.flush()

    def reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", JSON_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} {format % args}\n")


def aggregate_chat(stream: bytes) -> dict:
    """Collapse an SSE ``chat.completion.chunk`` stream into one JSON completion.

    The free tier only answers SSE, so clients that asked for ``stream: false``
    get their expected shape back from here.
    """
    completion: dict = {
        "id": None,
        "object": "chat.completion",
        "created": None,
        "model": None,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}],
        "usage": None,
    }
    message = completion["choices"][0]["message"]
    tool_calls: dict[int, dict] = {}
    for line in stream.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        for key in ("id", "created", "model"):
            if chunk.get(key) is not None:
                completion[key] = chunk[key]
        if chunk.get("usage"):
            completion["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                message["content"] += delta["content"]
            for field in ("reasoning", "reasoning_content"):
                if isinstance(delta.get(field), str):
                    message[field] = message.get(field, "") + delta[field]
            for call in delta.get("tool_calls") or []:
                entry = tool_calls.setdefault(
                    call.get("index", 0),
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    entry["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    entry["function"]["name"] += function["name"]
                if function.get("arguments"):
                    entry["function"]["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                completion["choices"][0]["finish_reason"] = choice["finish_reason"]
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return completion


def check(upstream: str, user_agent: str) -> int:
    """Ask the free tier a tiny question; 0 when the gate accepts us."""
    payload, _ = prepare_payload(
        json.dumps(
            {
                "model": "big-pickle",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with: ok"}],
                "tools": [READ_TOOL],
            }
        ).encode()
    )
    request = urllib.request.Request(
        f"{upstream}/chat/completions",
        data=payload,
        headers={"content-type": JSON_TYPE, **request_headers(user_agent)},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        body = error.read().decode()
        print(f"free tier rejected the request ({error.code}): {body[:300]}")
        return 1
    except urllib.error.URLError as error:
        print(f"could not reach {upstream}: {error.reason}")
        return 1
    print(f"free tier is reachable: {body[:300]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opencode-free", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8901, help="port to listen on (default: 8901)")
    parser.add_argument("--upstream", default=UPSTREAM, help=f"inference base URL (default: {UPSTREAM})")
    parser.add_argument("--user-agent", default=USER_AGENT, help="User-Agent sent upstream")
    parser.add_argument(
        "--forward-auth",
        action="store_true",
        help="pass the client's Authorization header upstream (default: drop it and use the keyless free tier)",
    )
    parser.add_argument("--check", action="store_true", help="verify the free tier accepts a forged request, then exit")
    args = parser.parse_args(argv)

    if args.check:
        return check(args.upstream, args.user_agent)

    ProxyHandler.upstream = args.upstream
    ProxyHandler.user_agent = args.user_agent
    ProxyHandler.forward_auth = args.forward_auth
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(f"opencode-free {__version__}: http://{args.host}:{args.port}/v1 -> {args.upstream}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0
