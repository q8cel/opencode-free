import json
import time
import unittest

from opencode_free.gate import (
    BASE62,
    READ_TOOL,
    SESSION_HEADER,
    USER_AGENT,
    ensure_read_tool,
    prepare_payload,
    request_headers,
    session_id,
)


class SessionIdTest(unittest.TestCase):
    def test_structure(self) -> None:
        value = session_id()
        self.assertTrue(value.startswith("ses_"))
        self.assertEqual(len(value), 30)
        prefix, suffix = value[4:16], value[16:]
        int(prefix, 16)  # first 12 chars are hex
        self.assertTrue(set(suffix) <= set(BASE62))

    def test_encodes_descending_timestamp(self) -> None:
        now = 1792096000123
        prefix = session_id(now_ms=now)[4:16]
        # The 48-bit prefix keeps only the low 36 bits of the millisecond timestamp.
        decoded = (~int(prefix, 16) & 0xFFFFFFFFFFFF) >> 12
        expected = ((now * 0x1000 + 1) & 0xFFFFFFFFFFFF) >> 12
        self.assertEqual(decoded, expected)

    def test_prefix_tracks_current_time(self) -> None:
        prefix = int(session_id()[4:16], 16)
        decoded_ms = (~prefix & 0xFFFFFFFFFFFF) >> 12
        now = int(time.time() * 1000) & ((1 << 36) - 1)
        self.assertLess(abs(decoded_ms - now), 5000)

    def test_randomness(self) -> None:
        self.assertNotEqual(session_id(), session_id())

    def test_matches_opencode_identifier_layout(self) -> None:
        # packages/schema/src/identifier.ts: 6 big-endian bytes of ~(ms * 0x1000 + counter) + 14 base62 chars.
        ms = 1792096000123
        expected = ~((ms * 0x1000 + 1) & 0xFFFFFFFFFFFF) & 0xFFFFFFFFFFFF
        prefix = int(session_id(now_ms=ms)[4:16], 16)
        self.assertEqual(prefix, expected)


class HeadersTest(unittest.TestCase):
    def test_defaults(self) -> None:
        headers = request_headers()
        self.assertEqual(headers["user-agent"], USER_AGENT)
        self.assertTrue(headers["user-agent"].startswith("opencode/"))
        self.assertEqual(len(headers[SESSION_HEADER]), 30)

    def test_user_agent_override_and_explicit_session(self) -> None:
        headers = request_headers("opencode/stable/2.0.29/opencode", session="ses_fixed")
        self.assertEqual(headers["user-agent"], "opencode/stable/2.0.29/opencode")
        self.assertEqual(headers[SESSION_HEADER], "ses_fixed")


class PreparePayloadTest(unittest.TestCase):
    def test_forces_stream_and_reports_intent(self) -> None:
        for client_stream, wants in [(True, True), (False, False), (None, False)]:
            with self.subTest(client_stream=client_stream):
                payload = {"model": "m"}
                if client_stream is not None:
                    payload["stream"] = client_stream
                prepared, wants_stream = prepare_payload(json.dumps(payload).encode())
                self.assertTrue(json.loads(prepared)["stream"])
                self.assertEqual(wants_stream, wants)

    def test_appends_read_tool(self) -> None:
        prepared, _ = prepare_payload(json.dumps({"model": "m", "tools": [{"function": {"name": "bash"}}]}).encode())
        names = [tool["function"]["name"] for tool in json.loads(prepared)["tools"]]
        self.assertEqual(names, ["bash", "read"])

    def test_keeps_existing_read_tool(self) -> None:
        prepared, _ = prepare_payload(json.dumps({"tools": [READ_TOOL]}).encode())
        self.assertEqual(json.loads(prepared)["tools"], [READ_TOOL])

    def test_non_json_passes_through_as_stream(self) -> None:
        self.assertEqual(prepare_payload(b"not json"), (b"not json", True))
        self.assertEqual(prepare_payload(b""), (b"", True))


class EnsureReadToolTest(unittest.TestCase):
    def test_appends_when_tools_missing(self) -> None:
        parsed = json.loads(ensure_read_tool(json.dumps({"model": "m", "messages": []}).encode()))
        self.assertEqual(parsed["tools"], [READ_TOOL])

    def test_appends_when_other_tools_only(self) -> None:
        other = {"type": "function", "function": {"name": "bash", "parameters": {}}}
        parsed = json.loads(ensure_read_tool(json.dumps({"tools": [other]}).encode()))
        self.assertEqual(parsed["tools"], [other, READ_TOOL])

    def test_keeps_existing_read_tool(self) -> None:
        body = json.dumps({"tools": [READ_TOOL]}).encode()
        self.assertEqual(ensure_read_tool(body), body)

    def test_detects_flat_tool_name(self) -> None:
        body = json.dumps({"tools": [{"type": "function", "name": "read"}]}).encode()
        self.assertEqual(ensure_read_tool(body), body)

    def test_ignores_malformed_tools_field(self) -> None:
        parsed = json.loads(ensure_read_tool(json.dumps({"tools": "nope"}).encode()))
        self.assertEqual(parsed["tools"], [READ_TOOL])

    def test_passes_non_json_through(self) -> None:
        self.assertEqual(ensure_read_tool(b"\xff\xfe not json"), b"\xff\xfe not json")
        self.assertEqual(ensure_read_tool(b"[1, 2]"), b"[1, 2]")
        self.assertEqual(ensure_read_tool(b""), b"")


if __name__ == "__main__":
    unittest.main()
