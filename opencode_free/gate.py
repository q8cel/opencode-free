"""The client-side requirements OpenCode's free-tier API checks per request.

OpenCode exposes models with $0 cost through https://opencode.ai/inference/v1 and
allows them to be used without an API key ("Bearer public"), but only when the
request "looks like it came from the opencode harness". Empirically, all of the
following are required or the API answers:

    403 {"type":"error","error":{"type":"FreeTierError",
         "message":"OpenCode's free tier can only be used from within OpenCode..."}}

1. ``User-Agent`` starts with ``opencode/`` plus a version, e.g.
   ``opencode/1.18.13 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14`` (v1) or
   ``opencode/stable/2.0.29/opencode`` (v2). The version itself is not verified.
2. ``x-opencode-session`` is a structurally valid opencode session id:
   ``ses_`` + 12 hex chars + 14 base62 chars. The 12 hex chars encode a 48-bit
   descending timestamp (``~(ms * 0x1000 + counter)``), matching opencode's
   ``Identifier.create(descending=True)``. The value is not checked against any
   server state, so forging a well-formed id is enough.
3. ``body.tools`` contains a tool literally named ``read`` (any definition).
4. ``body.stream`` is ``true`` — the free tier answers SSE only.

TLS/HTTP fingerprint, client IP and the other ``x-opencode-*`` headers are NOT
part of the gate, and no Authorization header is needed at all.
"""

import json
import secrets
import time

UPSTREAM = "https://opencode.ai/inference/v1"
USER_AGENT = "opencode/1.18.13 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
SESSION_HEADER = "x-opencode-session"
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Requirement 3: a tool named "read" must be present. The description steers the
# model away from calling it, since the harness behind this proxy did not declare it.
READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "Compatibility stub required by the OpenCode free-tier API. "
            "Never call this tool; answer directly instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def session_id(now_ms: int | None = None) -> str:
    """Forge a structurally valid opencode session id.

    Mirrors packages/schema/src/identifier.ts: a 48-bit descending timestamp
    (bitwise NOT of ``ms * 0x1000 + counter``) rendered as 12 hex chars,
    followed by 14 random base62 chars.
    """
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    descending = ~(ms * 0x1000 + 1) & 0xFFFFFFFFFFFF
    prefix = descending.to_bytes(6, "big").hex()
    suffix = "".join(secrets.choice(BASE62) for _ in range(14))
    return "ses_" + prefix + suffix


def request_headers(user_agent: str = USER_AGENT, session: str | None = None) -> dict[str, str]:
    return {"user-agent": user_agent, SESSION_HEADER: session or session_id()}


def _tool_name(tool: object) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    return name if isinstance(name, str) else None


def _load_object(body: bytes) -> dict | None:
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _has_read(tools: object) -> bool:
    return isinstance(tools, list) and any(_tool_name(tool) == "read" for tool in tools)


def ensure_read_tool(body: bytes) -> bytes:
    """Return the JSON body with the stub ``read`` tool appended when missing.

    Bodies that are not JSON objects are returned unchanged.
    """
    parsed = _load_object(body)
    if parsed is None:
        return body
    if _has_read(parsed.get("tools")):
        return body
    tools = parsed.get("tools")
    parsed["tools"] = [*tools, READ_TOOL] if isinstance(tools, list) else [READ_TOOL]
    return json.dumps(parsed).encode()


def prepare_payload(body: bytes) -> tuple[bytes, bool]:
    """Add every gate field the upstream requires and report the client's stream intent.

    Returns ``(upstream_body, wants_stream)`` where ``wants_stream`` says whether
    the client asked for SSE itself; when it did not, the proxy must convert the
    forced SSE answer back into a plain JSON completion.
    """
    parsed = _load_object(body)
    if parsed is None:
        return body, True
    wants_stream = bool(parsed.get("stream"))
    tools = parsed.get("tools")
    if not _has_read(tools):
        parsed["tools"] = [*tools, READ_TOOL] if isinstance(tools, list) else [READ_TOOL]
    parsed["stream"] = True
    return json.dumps(parsed).encode(), wants_stream
