"""Transparent Gemini OpenAI-compatibility proxy for Open WebUI native tool calling.

Fixes known incompatibilities between Open WebUI's Native function-calling loop and
Google's OpenAI-compatible endpoint (https://generativelanguage.googleapis.com/v1beta/openai):

1. Missing ``index`` on streamed ``tool_calls[]`` deltas:
   Gemini omits ``index`` on SSE tool-call chunks, causing Open WebUI's stream
   accumulator to discard the tool calls and return empty responses.
2. Stripped ``extra_content.google.thought_signature`` on follow-up turns:
   Gemini thinking models (2.5 / 3.x) require ``thought_signature`` to be echoed
   back on assistant ``tool_calls``. Open WebUI strips unknown fields, triggering
   HTTP 400 errors on turn 2. This proxy caches ``extra_content`` by tool call ID
   and re-injects it automatically (falling back to Google's documented
   ``skip_thought_signature_validator`` sentinel for uncached historical calls).
3. Strict JSON Schema validation for MCP tools:
   Flattens Pydantic/FastMCP ``anyOf: [{"type": "..."}, {"type": "null"}]`` unions
   and strips unsupported keywords (``default``, ``title``, ``additionalProperties``,
   ``$schema``) before forwarding tool declarations to Gemini.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from src import config

logger = logging.getLogger(__name__)

_MAX_CACHE_ENTRIES = 2000
_SKIP_THOUGHT_SIG_SENTINEL = "skip_thought_signature_validator"

_HOP_BY_HOP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "accept-encoding",
    }
)

_HOP_BY_HOP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
    }
)

_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "$defs",
        "definitions",
        "additionalProperties",
        "default",
        "title",
        "examples",
    }
)

_cache_lock = threading.Lock()
_thought_sig_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_seen_without_sig: OrderedDict[str, None] = OrderedDict()

_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client configured for long Gemini thinking responses."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0),
            follow_redirects=True,
        )
    return _http_client


async def close_http_client() -> None:
    """Close the shared async HTTP client during server shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


def clear_signature_cache() -> None:
    """Clear in-memory thought signature caches (mainly for tests)."""
    with _cache_lock:
        _thought_sig_cache.clear()
        _seen_without_sig.clear()


def cache_tool_call_metadata(tool_call_id: str, extra_content: dict[str, Any] | None) -> None:
    """Record ``extra_content`` (or absence thereof) for a tool_call ID."""
    if not tool_call_id:
        return
    with _cache_lock:
        if extra_content:
            _thought_sig_cache[tool_call_id] = copy.deepcopy(extra_content)
            _thought_sig_cache.move_to_end(tool_call_id)
            _seen_without_sig.pop(tool_call_id, None)
            while len(_thought_sig_cache) > _MAX_CACHE_ENTRIES:
                _thought_sig_cache.popitem(last=False)
        elif tool_call_id not in _thought_sig_cache:
            _seen_without_sig[tool_call_id] = None
            _seen_without_sig.move_to_end(tool_call_id)
            while len(_seen_without_sig) > _MAX_CACHE_ENTRIES:
                _seen_without_sig.popitem(last=False)


def lookup_tool_call_metadata(tool_call_id: str) -> tuple[dict[str, Any] | None, bool]:
    """
    Look up cached ``extra_content`` for ``tool_call_id``.
    Returns ``(extra_content_or_None, was_seen_without_signature)``.
    """
    if not tool_call_id:
        return None, False
    with _cache_lock:
        cached = _thought_sig_cache.get(tool_call_id)
        if cached is not None:
            _thought_sig_cache.move_to_end(tool_call_id)
            return copy.deepcopy(cached), False
        if tool_call_id in _seen_without_sig:
            _seen_without_sig.move_to_end(tool_call_id)
            return None, True
    return None, False


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local JSON Schema ``$ref`` like ``#/$defs/Foo``."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
        target = defs.get(parts[1])
        if isinstance(target, dict):
            return copy.deepcopy(target)
    return None


def _is_null_schema(branch: Any) -> bool:
    return isinstance(branch, dict) and branch.get("type") == "null"


def sanitize_json_schema(
    schema: Any,
    defs: dict[str, Any] | None = None,
    _seen_refs: frozenset[str] = frozenset(),
) -> Any:
    """
    Recursively simplify JSON Schema into the subset cleanly accepted by
    Gemini's OpenAI-compatible function-calling validator.
    """
    if not isinstance(schema, dict):
        return schema

    if defs is None:
        raw_defs = schema.get("$defs") or schema.get("definitions")
        defs = raw_defs if isinstance(raw_defs, dict) else {}

    node = dict(schema)

    # Inline local $ref if present
    ref = node.get("$ref")
    if isinstance(ref, str) and ref not in _seen_refs:
        resolved = _resolve_ref(ref, defs)
        if resolved is not None:
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            merged = {**resolved, **siblings}
            return sanitize_json_schema(merged, defs=defs, _seen_refs=_seen_refs | {ref})

    # Flatten nullable anyOf / oneOf unions (e.g. Pydantic's `str | None = None`)
    for union_key in ("anyOf", "oneOf"):
        branches = node.get(union_key)
        if isinstance(branches, list):
            non_null = [b for b in branches if not _is_null_schema(b)]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                chosen = sanitize_json_schema(non_null[0], defs=defs, _seen_refs=_seen_refs)
                if isinstance(chosen, dict):
                    merged = dict(chosen)
                    if "description" in node and "description" not in merged:
                        merged["description"] = node["description"]
                    return merged
            elif non_null:
                sanitized_branches = [
                    sanitize_json_schema(b, defs=defs, _seen_refs=_seen_refs) for b in non_null
                ]
                result: dict[str, Any] = {"anyOf": sanitized_branches}
                if "description" in node:
                    result["description"] = node["description"]
                return result

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if key == "type" and isinstance(value, list):
            non_null_types = [t for t in value if t != "null"]
            cleaned["type"] = non_null_types[0] if non_null_types else "string"
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                prop_name: sanitize_json_schema(prop_schema, defs=defs, _seen_refs=_seen_refs)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = sanitize_json_schema(value, defs=defs, _seen_refs=_seen_refs)
        elif key == "required" and isinstance(value, list):
            if value:
                cleaned["required"] = value
        else:
            cleaned[key] = value

    if cleaned.get("type") == "object" and "properties" not in cleaned:
        cleaned["properties"] = {}

    return cleaned


def prepare_outgoing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Patch an outgoing ``/chat/completions`` payload before forwarding to Gemini:
    - Sanitize ``tools[].function.parameters`` schemas.
    - Re-inject cached ``extra_content`` (``thought_signature``) onto assistant
      ``tool_calls`` stripped by Open WebUI, using Google's bypass sentinel only
      for uncached historical calls.
    """
    patched = copy.deepcopy(payload)

    tools = patched.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
                fn["parameters"] = sanitize_json_schema(fn["parameters"])

    messages = patched.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                continue

            has_any_extra = False
            all_known_without_sig = True
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                if tc.get("extra_content"):
                    has_any_extra = True
                    continue
                tc_id = tc.get("id")
                cached_extra, seen_no_sig = lookup_tool_call_metadata(str(tc_id or ""))
                if cached_extra is not None:
                    tc["extra_content"] = cached_extra
                    has_any_extra = True
                elif not seen_no_sig:
                    all_known_without_sig = False

            # If none of the calls in this turn carried a signature AND at least one
            # call predates this process (e.g. after container restart), inject
            # Google's documented sentinel on the first call to avoid HTTP 400.
            if not has_any_extra and not all_known_without_sig:
                first_tc = tool_calls[0]
                if isinstance(first_tc, dict):
                    first_tc["extra_content"] = {
                        "google": {"thought_signature": _SKIP_THOUGHT_SIG_SENTINEL}
                    }

    return patched


class SSEToolCallPatcher:
    """
    Stateful per-stream patcher for Gemini OpenAI-compatible SSE chunks.

    - Assigns valid sequential ``index`` values to ``delta.tool_calls[]`` items.
    - Caches ``extra_content`` (including ``thought_signature``) by ``tool_call.id``.
    - Ensures ``finish_reason`` is ``"tool_calls"`` when tool calls were emitted.
    """

    def __init__(self) -> None:
        self._id_to_index: dict[int, dict[str, int]] = {}
        self._index_to_id: dict[int, dict[int, str]] = {}
        self._next_index: dict[int, int] = {}
        self._last_index: dict[int, int | None] = {}
        self._saw_tool_calls: dict[int, bool] = {}

    def patch_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return chunk

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_idx = choice.get("index")
            if not isinstance(choice_idx, int) or isinstance(choice_idx, bool):
                choice_idx = 0
                choice["index"] = 0

            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
                self._patch_tool_calls(choice_idx, delta["tool_calls"])

            if (
                self._saw_tool_calls.get(choice_idx)
                and choice.get("finish_reason") == "stop"
            ):
                choice["finish_reason"] = "tool_calls"

        return chunk

    def _patch_tool_calls(self, choice_idx: int, tool_calls: list[Any]) -> None:
        id_map = self._id_to_index.setdefault(choice_idx, {})
        idx_to_id = self._index_to_id.setdefault(choice_idx, {})
        next_idx = self._next_index.get(choice_idx, 0)
        last_idx = self._last_index.get(choice_idx)

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            self._saw_tool_calls[choice_idx] = True

            raw_idx = tc.get("index")
            tc_id = tc.get("id")
            fn = tc.get("function")
            fn_name = fn.get("name") if isinstance(fn, dict) else None

            if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
                idx = raw_idx
                if tc_id:
                    id_map[str(tc_id)] = idx
                    idx_to_id[idx] = str(tc_id)
                next_idx = max(next_idx, idx + 1)
            else:
                if tc_id and str(tc_id) in id_map:
                    idx = id_map[str(tc_id)]
                elif tc_id:
                    idx = next_idx
                    id_map[str(tc_id)] = idx
                    idx_to_id[idx] = str(tc_id)
                    next_idx += 1
                elif fn_name and (last_idx is None or idx_to_id.get(last_idx) is not None):
                    idx = next_idx
                    next_idx += 1
                elif last_idx is not None:
                    idx = last_idx
                else:
                    idx = next_idx
                    next_idx += 1
                tc["index"] = idx

            if not tc_id and fn_name and idx not in idx_to_id:
                synth_id = f"call_{choice_idx}_{idx}"
                tc["id"] = synth_id
                tc_id = synth_id
                id_map[synth_id] = idx
                idx_to_id[idx] = synth_id

            if (tc_id or fn_name) and not tc.get("type"):
                tc["type"] = "function"

            effective_id = str(tc_id) if tc_id else idx_to_id.get(idx, "")
            if effective_id:
                extra = tc.get("extra_content")
                cache_tool_call_metadata(
                    effective_id,
                    extra if isinstance(extra, dict) else None,
                )

            last_idx = idx

        self._next_index[choice_idx] = next_idx
        self._last_index[choice_idx] = last_idx


def patch_non_streaming_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Capture ``extra_content`` and normalize tool calls on a non-streaming response."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return payload

    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue

        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            if not tc.get("id"):
                tc["id"] = f"call_{choice.get('index', 0)}_{i}"
            if not tc.get("type"):
                tc["type"] = "function"
            extra = tc.get("extra_content")
            cache_tool_call_metadata(
                str(tc["id"]),
                extra if isinstance(extra, dict) else None,
            )

        if choice.get("finish_reason") == "stop":
            choice["finish_reason"] = "tool_calls"

    return payload


def _build_upstream_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in _HOP_BY_HOP_REQUEST_HEADERS:
            continue
        headers[key] = value
    if "authorization" not in {k.lower() for k in headers} and config.GEMINI_API_KEY:
        headers["Authorization"] = f"Bearer {config.GEMINI_API_KEY}"
    return headers


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS
    }


async def _stream_patched_sse(
    upstream_resp: httpx.Response,
    is_chat_completions: bool,
) -> AsyncIterator[bytes]:
    patcher = SSEToolCallPatcher() if is_chat_completions else None
    try:
        async for line in upstream_resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data:"):
                raw_data = line[5:].strip()
                if raw_data == "[DONE]":
                    yield b"data: [DONE]\n\n"
                    continue
                if patcher is not None and raw_data:
                    try:
                        chunk = json.loads(raw_data)
                    except json.JSONDecodeError:
                        yield f"data: {raw_data}\n\n".encode("utf-8")
                    else:
                        if isinstance(chunk, dict):
                            chunk = patcher.patch_chunk(chunk)
                        yield (
                            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        ).encode("utf-8")
                else:
                    yield f"data: {raw_data}\n\n".encode("utf-8")
            elif line.startswith(":"):
                yield f"{line}\n\n".encode("utf-8")
            else:
                yield f"{line}\n".encode("utf-8")
    finally:
        await upstream_resp.aclose()


async def gemini_proxy_endpoint(request: Request) -> Response:
    """Forward ``/gemini/v1/{path}`` requests to Gemini's OpenAI-compatible API."""
    subpath = request.path_params.get("path", "").lstrip("/")
    base = config.GEMINI_OPENAI_BASE_URL.rstrip("/")
    target_url = f"{base}/{subpath}" if subpath else base
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    is_chat_completions = (
        request.method.upper() == "POST" and subpath.rstrip("/").endswith("chat/completions")
    )

    body = await request.body()
    headers = _build_upstream_headers(request)

    if is_chat_completions and body:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            patched = prepare_outgoing_payload(parsed)
            body = json.dumps(patched, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

    client = _get_http_client()
    upstream_req = client.build_request(
        method=request.method,
        url=target_url,
        headers=headers,
        content=body if body else None,
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        logger.error("Gemini proxy upstream error (%s %s): %s", request.method, target_url, exc)
        return Response(
            content=json.dumps({"error": {"message": f"Upstream Gemini error: {exc}"}}),
            status_code=502,
            media_type="application/json",
        )

    content_type = upstream_resp.headers.get("content-type", "")
    resp_headers = _filter_response_headers(upstream_resp.headers)

    if upstream_resp.status_code >= 400:
        err_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        logger.warning(
            "Gemini upstream returned HTTP %d for %s %s: %s",
            upstream_resp.status_code,
            request.method,
            subpath,
            err_body.decode("utf-8", errors="replace")[:1000],
        )
        return Response(
            content=err_body,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=content_type or "application/json",
        )

    if "text/event-stream" in content_type.lower():
        resp_headers["Cache-Control"] = "no-cache"
        resp_headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(
            _stream_patched_sse(upstream_resp, is_chat_completions=is_chat_completions),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type="text/event-stream",
        )

    raw_body = await upstream_resp.aread()
    await upstream_resp.aclose()

    if is_chat_completions and "application/json" in content_type.lower():
        try:
            resp_json = json.loads(raw_body)
        except json.JSONDecodeError:
            resp_json = None
        if isinstance(resp_json, dict):
            resp_json = patch_non_streaming_response(resp_json)
            raw_body = json.dumps(resp_json, ensure_ascii=False).encode("utf-8")

    return Response(
        content=raw_body,
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=content_type or None,
    )
