# opencode-free

Use OpenCode's free-tier models from **any** OpenAI-compatible harness (Pi, one-api,
librechat, or plain `curl`) through a tiny local proxy.

OpenCode publishes a list of $0-cost models on `https://opencode.ai/inference/v1` and
lets the opencode harness use them without any API key — but requests from anything
else are rejected with:

```json
{"type":"error","error":{"type":"FreeTierError",
 "message":"OpenCode's free tier can only be used from within OpenCode. Download it at https://opencode.ai and keep building."}}
```

This repo documents exactly what the harness does differently and proxies your traffic
so it matches. Zero dependencies — Python 3.10+ standard library only.

## The gate

Every request to the free tier is checked for four things. All of them are required:

| # | Requirement | Value used by this proxy |
|---|-------------|--------------------------|
| 1 | `User-Agent` starts with `opencode/` plus a version | `opencode/1.18.13 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14` |
| 2 | `x-opencode-session` is a structurally valid opencode session id: `ses_` + 12 hex chars (48-bit *descending* timestamp) + 14 base62 chars | forged per request, never verified server-side |
| 3 | `body.tools` contains a tool literally named `read` (any definition) | a stub tool with a "never call me" description is appended when missing |
| 4 | `body.stream` is `true` — the free tier answers SSE only | forced on every request; non-streaming clients get their JSON aggregated back |

Verified **not** part of the gate: TLS/HTTP fingerprint (works from curl, python, node,
bun), client IP (a separate per-model `RegionError` exists), the other `x-opencode-*`
headers (`client`, `project`, `request`), and authentication — no `Authorization`
header is needed at all (the harness itself sends `Bearer public`).

The session-id encoding mirrors opencode's `Identifier.create(descending=True)` in
`packages/schema/src/identifier.ts`: `~(ms * 0x1000 + counter)` masked to 48 bits,
big-endian, hex-encoded, plus 14 random base62 characters.

The models.dev catalog entry for the `opencode` provider has pointed at both
`https://opencode.ai/inference/v1` and `https://opencode.ai/inference/openai/v1`;
both accept the gated requests today. The default is the former — override with
`--upstream` if the catalog moves again.

## Quick start

```sh
python3 -m opencode_free            # proxy on http://127.0.0.1:8901/v1
python3 -m opencode_free --check    # verify the free tier still accepts forged requests
```

or install it:

```sh
pip install .
opencode-free --port 8901
```

Then configure your harness with an OpenAI-compatible provider:

| setting  | value |
|----------|-------|
| base URL | `http://127.0.0.1:8901/v1` |
| API key  | *(anything or nothing — auth is not checked)* |
| models   | `GET /v1/models` lists them; free ones include `big-pickle`, `glm-4.7-free`, `kimi-k2.5-free`, `deepseek-v4.1-flash-free`, … |

Example with `curl`:

```sh
curl http://127.0.0.1:8901/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"big-pickle","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

SSE streams pass through chunk by chunk, so streaming completions work normally.
Clients that request `stream: false` get a standard JSON `chat.completion` back —
the proxy forces SSE upstream and aggregates the chunks.

## Repository layout

```
opencode_free/gate.py    the gate requirements: header building, session-id
                         forging, read-tool injection, stream forcing
opencode_free/proxy.py   the threading HTTP proxy, SSE→JSON aggregation for
                         non-streaming clients, and CLI (--host --port --upstream
                         --user-agent --check)
tests/                   stdlib unittest suite (local fixture upstream, no mocks)
```

Run the tests:

```sh
python3 -m unittest discover -s tests -v
```

## Limitations & notes

- The proxy appends a stub `read` tool when your client's request has none (or lacks
  one). Its description tells the model never to call it, so harnesses that did not
  declare a `read` tool will rarely see one; if yours does, implement or block it.
- SSE→JSON conversion applies to `POST .../chat/completions`; other endpoints are
  forwarded as-is, so use `stream: true` with them.
- Chunked request bodies are rejected with a clear 400 — every harness tested sends
  fixed-length JSON bodies.
- Free models only. Paid models on the same endpoint require a real `OPENCODE_API_KEY`.
- This is not affiliated with OpenCode. The checks were derived by observation and may
  change at any time; `python3 -m opencode_free --check` tells you whether they still
  hold. Please respect OpenCode's terms and keep usage reasonable.
