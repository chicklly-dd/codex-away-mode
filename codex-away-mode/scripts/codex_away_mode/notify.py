from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import cards
from .config import AppConfig, effective_notification_mode as _config_mode
from .config import ensure_runtime_state_writable, load_config, save_config
from .state import StateStore


DEFAULT_SUMMARY_MAX_AGE_SECONDS = 300
DEFAULT_PROMPT_MARKER_MAX_AGE_SECONDS = 300
DEFAULT_COMPLETION_MARKER_MAX_AGE_SECONDS = 24 * 60 * 60
MAX_DISPLAY_COMMAND_CHARS = 240
_SAFE_CAPTURE_VALUE_KEYS = {
    "approval_policy",
    "cwd",
    "event",
    "event_name",
    "goal_state",
    "goal_status",
    "hook_event_name",
    "model",
    "sandbox_mode",
    "session_id",
    "source",
    "status",
    "thread_id",
    "thread_source",
    "turn_id",
    "type",
}
_MAX_CAPTURE_DEPTH = 6
_MAX_CAPTURE_LIST_ITEMS = 20


@dataclass(frozen=True)
class NotifyResult:
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class PermissionRequestContext:
    hook_event_name: str
    tool_name: str
    session_id: str | None
    turn_id: str | None
    cwd: str | None
    command: str | None
    description: str | None
    raw_tool_input: dict
    command_hash: str
    dedupe_key: str


def _runtime_store(paths) -> StateStore:
    ensure_runtime_state_writable(paths)
    return StateStore(Path(paths.runtime_state_path))


def mark_prompt(
    paths,
    cwd: str,
    now: datetime,
    hook_stdin: str | bytes | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> str:
    store = _runtime_store(paths)
    route = resolve_completion_route(
        cwd=cwd,
        hook_stdin=hook_stdin,
        explicit_session_id=session_id,
        explicit_turn_id=turn_id,
    )
    expires_at = (
        _to_utc(now) + timedelta(seconds=DEFAULT_COMPLETION_MARKER_MAX_AGE_SECONDS)
    ).isoformat()
    route_key = store.mark_completion_prompt(
        route=route,
        marked_at=_to_utc(now).isoformat(),
        expires_at=expires_at,
    )
    store.mark_prompt_marker(
        cwd=cwd,
        marked_at=_to_utc(now).isoformat(),
        expires_at=expires_at,
    )
    return route_key


def start_live_completion_card(
    paths,
    lark,
    *,
    cwd: str,
    hook_stdin: str | bytes | None,
    now: datetime,
) -> NotifyResult:
    try:
        config = load_config(Path(paths.config_path))
        if _config_mode(config, now=now) == "off":
            return NotifyResult("skipped", "notification_mode_off")
        send_live = getattr(lark, "send_live_completion_card", None)
        if send_live is None:
            return NotifyResult("skipped", "live_card_client_unavailable")

        store = _runtime_store(paths)
        route = resolve_completion_route(cwd=cwd, hook_stdin=hook_stdin)
        started_at = _to_utc(now).isoformat()
        activities = [{"kind": "commentary", "text": "收到新任务，开始处理。"}]
        transcript_offset = _transcript_file_size_from_hook(hook_stdin)
        existing = store.get_live_completion_card(route)
        if existing:
            update_live = getattr(lark, "update_live_completion_card", None)
            if update_live is None:
                return NotifyResult("update_failed", "live_card_client_unavailable")
            update_live(
                message_id=existing["message_id"],
                status="working",
                activities=activities,
                started_at=started_at,
                now=started_at,
                cwd=cwd,
                answer=None,
            )
            store.create_live_completion_card(
                route=route,
                message_id=existing["message_id"],
                started_at=started_at,
                activities=activities,
                transcript_offset=transcript_offset,
            )
            return NotifyResult("started")
        result = send_live(
            status="working",
            activities=activities,
            started_at=started_at,
            now=started_at,
            cwd=cwd,
            answer=None,
        )
        message_id = getattr(result, "message_id", None)
        if not message_id:
            return NotifyResult("send_failed", "message_id_missing")
        store.create_live_completion_card(
            route=route,
            message_id=str(message_id),
            started_at=started_at,
            activities=activities,
            transcript_offset=transcript_offset,
        )
    except Exception:
        return NotifyResult("send_failed")
    return NotifyResult("started")


def update_live_completion_progress(
    paths,
    lark,
    *,
    cwd: str,
    hook_stdin: str | bytes | None,
    now: datetime,
) -> NotifyResult:
    config = load_config(Path(paths.config_path))
    if _config_mode(config, now=now) == "off":
        return NotifyResult("skipped", "notification_mode_off")
    update_live = getattr(lark, "update_live_completion_card", None)
    if update_live is None:
        return NotifyResult("skipped", "live_card_client_unavailable")
    hook_payload = _hook_payload_mapping(hook_stdin)
    tool_name = hook_payload.get("tool_name")
    if not tool_name:
        return NotifyResult("skipped", "tool_name_missing")

    store = _runtime_store(paths)
    route = resolve_completion_route(cwd=cwd, hook_stdin=hook_stdin)
    live = store.get_live_completion_card(route)
    if not live:
        return NotifyResult("skipped", "live_card_missing")
    transcript_path = hook_payload.get("transcript_path")
    commentary, transcript_offset = _transcript_commentary_delta(
        transcript_path,
        int(live.get("transcript_offset") or 0),
        cwd=cwd,
    )
    tool_activity = _tool_progress_entry(hook_payload, cwd=cwd)
    entries = [*commentary, tool_activity]
    row = store.append_live_completion_entries(
        route=route,
        entries=entries,
        updated_at=_to_utc(now).isoformat(),
        transcript_offset=transcript_offset,
    )
    if not row:
        return NotifyResult("skipped", "live_card_missing")
    if not row.get("appended_count"):
        return NotifyResult("updated", str(row.get("activity_count") or 0))
    payload = {
        "status": "working",
        "activities": row.get("activities") or [],
        "activity_count": row.get("activity_count"),
        "started_at": row["started_at"],
        "now": _to_utc(now).isoformat(),
        "cwd": cwd,
        "answer": None,
    }
    try:
        update_live(message_id=row["message_id"], **payload)
    except Exception:
        return NotifyResult("update_failed")
    return NotifyResult("updated", str(row.get("activity_count") or 0))


def _tool_progress_entry(payload: dict[str, Any], *, cwd: str) -> dict[str, Any]:
    tool_name = str(payload.get("tool_name") or "").strip()
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    command = tool_input.get("command")
    if not command and isinstance(tool_input.get("cmd"), str):
        command = tool_input["cmd"]
    if not command and isinstance(tool_input.get("code"), str):
        nested = re.search(r'\bcmd\s*:\s*("(?:\\.|[^"\\])*")', tool_input["code"])
        if nested:
            try:
                command = json.loads(nested.group(1))
            except json.JSONDecodeError:
                pass
    if isinstance(command, str) and command.strip() and tool_name != "apply_patch":
        display_command = _truncate_display(
            _redact_progress_content(command, cwd=cwd),
            MAX_DISPLAY_COMMAND_CHARS,
        )
        return {
            "kind": "command",
            "text": f"已运行命令：\n{display_command}",
            "tool_use_id": _optional_text(payload.get("tool_use_id")),
        }
    if tool_name == "apply_patch" and isinstance(command, str):
        return {
            "kind": "tool",
            "text": _summarize_patch(command),
            "tool_use_id": _optional_text(payload.get("tool_use_id")),
        }

    path = tool_input.get("file_path") or tool_input.get("path")
    basename = re.split(r"[\\/]", str(path).strip())[-1] if path else ""
    if basename and tool_name in {"Read", "Write", "Edit"}:
        verb = {"Read": "读取", "Write": "写入", "Edit": "编辑"}[tool_name]
        label = f"已{verb} {basename}"
    else:
        label = _progress_activity_label(tool_name)
    return {
        "kind": "tool",
        "text": label,
        "tool_use_id": _optional_text(payload.get("tool_use_id")),
    }


def _progress_activity_label(tool_name: str) -> str:
    name = str(tool_name).strip()
    labels = {
        "Bash": "已完成命令操作",
        "apply_patch": "已完成代码修改",
        "Read": "已读取项目内容",
        "Write": "已写入文件",
        "Edit": "已编辑文件",
        "Agent": "已完成子任务",
    }
    if name in labels:
        return labels[name]
    if name.startswith("mcp__"):
        return f"已完成外部工具操作（{name}）"
    return f"已完成工具操作（{name or '未知工具'}）"


def _summarize_patch(patch: str) -> str:
    files = []
    for line in patch.splitlines():
        for operation in ("Update", "Add", "Delete", "Move"):
            prefix = f"*** {operation} File:"
            if line.startswith(prefix):
                files.append(line[len(prefix) :].strip())
                break
    added = sum(1 for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines() if line.startswith("-") and not line.startswith("---"))
    if files:
        names = [re.split(r"[\\/]", item.strip())[-1] for item in files[:8]]
        suffix = f"（+{added} -{removed}）" if added or removed else ""
        extra = f" 等 {len(files)} 个文件" if len(files) > len(names) else ""
        return f"已编辑 {', '.join(names)}{extra}{suffix}"
    return "已完成代码修改"


def _hook_payload_mapping(hook_stdin: str | bytes | None) -> dict[str, Any]:
    if isinstance(hook_stdin, bytes):
        hook_text = hook_stdin.decode("utf-8", errors="replace")
    else:
        hook_text = hook_stdin or ""
    try:
        payload = json.loads(hook_text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _transcript_file_size_from_hook(hook_stdin: str | bytes | None) -> int:
    transcript_path = _hook_payload_mapping(hook_stdin).get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return 0
    try:
        return Path(transcript_path).stat().st_size
    except OSError:
        return 0


def _transcript_commentary_delta(
    transcript_path: Any,
    offset: int,
    *,
    cwd: str,
) -> tuple[list[dict[str, Any]], int | None]:
    """Read only visible assistant commentary; transcript structure is best-effort."""
    if not isinstance(transcript_path, str) or not transcript_path:
        return [], None
    path = Path(transcript_path)
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            if offset < 0 or offset > size:
                # Avoid replaying earlier turns if Codex compacts or replaces the transcript.
                return [], size
            handle.seek(offset)
            chunk = handle.read()
    except OSError:
        return [], None

    last_newline = chunk.rfind(bytes((10,)))
    if last_newline < 0:
        return [], offset
    complete = chunk[: last_newline + 1]
    cursor = offset
    entries: list[dict[str, Any]] = []
    for raw_line in complete.splitlines(keepends=True):
        cursor += len(raw_line)
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        # Do not read analysis/reasoning or encrypted summary records.
        if (
            payload.get("type") != "message"
            or payload.get("role") != "assistant"
            or payload.get("phase") != "commentary"
        ):
            continue
        content = payload.get("content")
        if not isinstance(content, list):
            continue
        text = "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "output_text"
        ).strip()
        if text:
            entries.append(
                {
                    "kind": "commentary",
                    "text": _redact_progress_content(text, cwd=cwd),
                    "cursor_end": cursor,
                }
            )
    return entries, offset + len(complete)


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _flush_live_commentary(
    store: StateStore,
    route,
    live: dict[str, Any],
    *,
    hook_stdin: str | bytes | None,
    cwd: str,
    now: datetime,
) -> dict[str, Any]:
    transcript_path = _hook_payload_mapping(hook_stdin).get("transcript_path")
    commentary, transcript_offset = _transcript_commentary_delta(
        transcript_path,
        int(live.get("transcript_offset") or 0),
        cwd=cwd,
    )
    if commentary or transcript_offset is not None:
        store.append_live_completion_entries(
            route=route,
            entries=commentary,
            updated_at=_to_utc(now).isoformat(),
            transcript_offset=transcript_offset,
        )
        return store.get_live_completion_card(route) or live
    return live


def _redact_progress_content(value: str, *, cwd: str) -> str:
    text = _redact_for_display(value) or ""
    local_paths = {cwd, str(Path.home())}
    for env_name in ("USERPROFILE", "LOCALAPPDATA", "APPDATA"):
        env_value = os.environ.get(env_name)
        if env_value:
            local_paths.add(env_value)
    for local_path in sorted(local_paths, key=len, reverse=True):
        if local_path:
            text = re.sub(re.escape(local_path), "[本地路径]", text, flags=re.IGNORECASE)
    return text


def stage_summary(
    paths,
    *,
    cwd: str,
    summary_markdown: str,
    now: datetime,
    max_age_seconds: int = DEFAULT_SUMMARY_MAX_AGE_SECONDS,
    session_id: str | None = None,
    turn_id: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    store = _runtime_store(paths)
    route = resolve_completion_route(
        cwd=cwd,
        explicit_session_id=session_id,
        explicit_turn_id=turn_id,
        env=env,
    )
    route_key = store.stage_completion_summary(
        route=route,
        summary_markdown=summary_markdown,
        staged_at=_to_utc(now).isoformat(),
        expires_at=(_to_utc(now) + timedelta(seconds=max_age_seconds)).isoformat(),
    )
    store.stage_summary(
        cwd=cwd,
        summary_markdown=summary_markdown,
        staged_at=_to_utc(now).isoformat(),
        expires_at=(_to_utc(now) + timedelta(seconds=max_age_seconds)).isoformat(),
    )
    return route_key


def send_permission_request(
    paths,
    lark,
    *,
    hook_stdin: str | bytes | None,
    now: datetime,
) -> NotifyResult:
    config = load_config(Path(paths.config_path))
    if not getattr(config, "approval_notifications_enabled", True):
        return NotifyResult("skipped", "approval_notifications_disabled")

    context = permission_request_context(hook_stdin)
    if context is None:
        return NotifyResult("skipped", "invalid_permission_request_payload")

    store = _runtime_store(paths)
    seen_at = _to_utc(now).isoformat()
    reserved = store.reserve_approval_notification(
        dedupe_key=context.dedupe_key,
        session_id=context.session_id,
        turn_id=context.turn_id,
        cwd=context.cwd,
        tool_name=context.tool_name,
        command_hash=context.command_hash,
        seen_at=seen_at,
        throttle_seconds=int(getattr(config, "approval_notifications_throttle_seconds", 300) or 300),
    )
    if reserved["status"] == "suppressed":
        return NotifyResult("suppressed", context.dedupe_key)

    try:
        send_result = lark.send_permission_request_card(
            {
                "project": cards.project_from_cwd(context.cwd),
                "cwd": context.cwd,
                "tool_name": context.tool_name,
                "description": context.description,
                "command": context.command,
                "now": now,
                "session_id": context.session_id,
                "turn_id": context.turn_id,
            }
        )
    except Exception:
        store.mark_approval_notification_result(
            context.dedupe_key,
            status="send_failed",
        )
        return NotifyResult("send_failed", context.dedupe_key)

    store.mark_approval_notification_result(
        context.dedupe_key,
        status="sent",
        sent_at=seen_at,
    )
    _send_permission_request_urgent_if_enabled(
        store,
        lark,
        config=config,
        context=context,
        message_id=getattr(send_result, "message_id", None),
        sent_at=seen_at,
    )
    return NotifyResult("sent", context.dedupe_key)


def _send_permission_request_urgent_if_enabled(
    store: StateStore,
    lark,
    *,
    config: AppConfig,
    context: PermissionRequestContext,
    message_id: str | None,
    sent_at: str,
) -> None:
    if not getattr(config, "approval_notifications_urgent_app_enabled", True):
        store.mark_approval_notification_urgent_result(
            context.dedupe_key,
            urgent_status="skipped_disabled",
        )
        return
    if not message_id:
        store.mark_approval_notification_urgent_result(
            context.dedupe_key,
            urgent_status="skipped_missing_message_id",
            urgent_error_code="message_id_missing",
        )
        return
    urgent_method = getattr(lark, "send_permission_request_urgent", None)
    if urgent_method is None:
        store.mark_approval_notification_urgent_result(
            context.dedupe_key,
            urgent_status="skipped_unavailable",
            urgent_error_code="client_missing_urgent_method",
        )
        return
    try:
        result = urgent_method(message_id=message_id)
    except Exception as exc:
        store.mark_approval_notification_urgent_result(
            context.dedupe_key,
            urgent_status="failed",
            urgent_error_code=_approval_urgent_error_code(exc),
            urgent_error_detail=_truncate_display(_redact_for_display(str(exc)), 300),
        )
        return
    invalid_ids = _extract_invalid_user_ids(result)
    invalid_hashes = [_short_sensitive_hash(value) for value in invalid_ids]
    if invalid_ids:
        store.mark_approval_notification_urgent_result(
            context.dedupe_key,
            urgent_status="failed",
            urgent_error_code="approval_urgent_invalid_user",
            urgent_error_detail="Feishu urgent_app returned invalid_user_id_list.",
            urgent_invalid_user_count=len(invalid_ids),
            urgent_invalid_user_hashes=invalid_hashes,
        )
        return
    store.mark_approval_notification_urgent_result(
        context.dedupe_key,
        urgent_status="sent",
        urgent_sent_at=sent_at,
        urgent_invalid_user_count=len(invalid_ids),
        urgent_invalid_user_hashes=invalid_hashes,
    )


def _approval_urgent_error_code(exc: Exception) -> str:
    text = str(exc).lower()
    if "im:message.urgent" in text or "permission" in text or "scope" in text:
        return "approval_urgent_permission_missing"
    if "feishu_user_id" in text or "user" in text and "missing" in text:
        return "feishu_user_id_missing"
    return "approval_urgent_failed"


def _extract_invalid_user_ids(payload: Any) -> list[str]:
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "invalid_user_id_list" and isinstance(child, list):
                    found.extend(str(item) for item in child if item)
                else:
                    walk(child)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _short_sensitive_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def permission_request_context(hook_stdin: str | bytes | None) -> PermissionRequestContext | None:
    payload = _json_payload(hook_stdin)
    if not isinstance(payload, dict):
        return None
    if payload.get("hook_event_name") != "PermissionRequest":
        return None
    raw_tool_input = payload.get("tool_input")
    if not isinstance(raw_tool_input, dict):
        raw_tool_input = {}
    tool_name = _string_or_default(payload.get("tool_name"), "未知工具")
    cwd = _optional_string(payload.get("cwd"))
    session_id = _optional_string(payload.get("session_id"))
    turn_id = _optional_string(payload.get("turn_id"))
    command = _permission_command(tool_name, raw_tool_input)
    description = _permission_description(raw_tool_input)
    display_command = _truncate_display(_redact_for_display(command), 800)
    display_description = _truncate_display(_redact_for_display(description), 300)
    fingerprint_source = display_command or display_description or tool_name
    command_hash = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    if session_id or turn_id:
        route_part = f"{session_id or 'no-session'}:{turn_id or 'no-turn'}"
    else:
        route_part = f"cwd:{StateStore.cwd_hash(cwd or '')}"
    dedupe_key = f"approval:{route_part}:{tool_name}:{command_hash}"
    return PermissionRequestContext(
        hook_event_name="PermissionRequest",
        tool_name=tool_name,
        session_id=session_id,
        turn_id=turn_id,
        cwd=_truncate_display(_redact_for_display(cwd), 120),
        command=display_command,
        description=display_description,
        raw_tool_input=raw_tool_input,
        command_hash=command_hash,
        dedupe_key=dedupe_key,
    )


def record_hook_invocation(
    paths,
    *,
    hook_event_name: str,
    cwd: str,
    now: datetime,
    hooks_fingerprint: str,
    hook_stdin: str | bytes | None,
) -> str | None:
    if not hook_stdin:
        return None
    try:
        store = _runtime_store(paths)
        return store.record_diagnostic_event(
            event_kind="codex_hook_invocation",
            severity="info",
            message=f"{hook_event_name} hook executed.",
            detail={
                "hook_event_name": hook_event_name,
                "cwd_hash": StateStore.cwd_hash(cwd),
                "hooks_fingerprint": hooks_fingerprint,
            },
            created_at=_to_utc(now).isoformat(),
        )
    except Exception:
        return None


def _record_completion_decision(
    store: StateStore,
    *,
    decision: str,
    reason: str | None,
    route,
    cwd: str | None,
    goal_status: str,
    summary_present: bool,
    marker_present: bool,
    now: datetime,
) -> None:
    try:
        store.record_diagnostic_event(
            event_kind="completion_notification_decision",
            severity="info" if decision != "skipped" else "warning",
            message="Completion notification decision.",
            detail={
                "decision": decision,
                "reason": reason,
                "route_key_hash": route.route_key_hash,
                "route_kind": route.route_kind,
                "cwd_hash": route.cwd_hash or (StateStore.cwd_hash(cwd) if cwd else None),
                "session_id_present": route.session_id_hash is not None,
                "turn_id_present": route.turn_id_hash is not None,
                "goal_status": goal_status,
                "summary_present": summary_present,
                "marker_present": marker_present,
            },
            created_at=_to_utc(now).isoformat(),
        )
    except Exception:
        return


def resolve_notify_cwd(
    explicit_cwd: str | None,
    hook_stdin: str | bytes | None,
    process_cwd: str | None,
) -> str:
    if explicit_cwd:
        return explicit_cwd
    stdin_cwd = _extract_stdin_cwd(hook_stdin)
    if stdin_cwd and os.path.isabs(stdin_cwd):
        return stdin_cwd
    return process_cwd or os.getcwd()


def resolve_completion_route(
    *,
    cwd: str | None,
    hook_stdin: str | bytes | None = None,
    explicit_session_id: str | None = None,
    explicit_turn_id: str | None = None,
    env: dict[str, str] | None = None,
):
    env_map = env if env is not None else {}
    session_id = (
        _extract_stdin_string_field(hook_stdin, "session_id")
        or _extract_stdin_string_field(hook_stdin, "thread_id")
        or _optional_string(explicit_session_id)
        or _optional_string(env_map.get("CODEX_THREAD_ID"))
    )
    turn_id = (
        _extract_stdin_string_field(hook_stdin, "turn_id")
        or _optional_string(explicit_turn_id)
    )
    return StateStore.build_completion_route(
        cwd=cwd,
        session_id=session_id,
        turn_id=turn_id,
    )


def send_completion_from_summary(
    paths,
    lark,
    cwd: str,
    now: datetime,
    hook_stdin: str | bytes | None = None,
    max_age_seconds: int = DEFAULT_SUMMARY_MAX_AGE_SECONDS,
    prompt_marker_max_age_seconds: int = DEFAULT_COMPLETION_MARKER_MAX_AGE_SECONDS,
) -> NotifyResult:
    store = _runtime_store(paths)
    route = resolve_completion_route(cwd=cwd, hook_stdin=hook_stdin)
    goal_status = goal_status_from_hook_stdin(hook_stdin)
    skip_reason = skip_cwd_reason(paths, cwd)
    if skip_reason:
        try:
            store.delete_completion_prompt_marker(route)
            store.delete_prompt_marker(cwd)
        except Exception:
            pass
        _record_completion_decision(
            store,
            decision="skipped",
            reason=skip_reason,
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=False,
            marker_present=False,
            now=now,
        )
        return NotifyResult("skipped", skip_reason)

    summary = store.get_completion_summary(route)
    legacy_summary = None if summary else store.get_staged_summary(cwd)
    active_summary = summary or legacy_summary
    marker = store.get_completion_prompt_marker(route)
    legacy_marker = None if marker else store.get_prompt_marker(cwd)
    active_marker = marker or legacy_marker

    if goal_status == "active":
        live = store.get_live_completion_card(route)
        update_live = getattr(lark, "update_live_completion_card", None)
        if live and update_live:
            live = _flush_live_commentary(
                store,
                route,
                live,
                hook_stdin=hook_stdin,
                cwd=cwd,
                now=now,
            )
            activities = list(live.get("activities") or [])
            if not activities or activities[-1] != "目标任务仍在继续。":
                activities.append("目标任务仍在继续。")
            try:
                update_live(
                    message_id=live["message_id"],
                    status="working",
                    activities=activities,
                    activity_count=live.get("activity_count"),
                    started_at=live["started_at"],
                    now=_to_utc(now).isoformat(),
                    cwd=cwd,
                    answer=None,
                )
            except Exception:
                pass
        store.delete_completion_summary(route)
        if active_summary:
            store.delete_staged_summary_by_hash(active_summary.get("cwd_hash"))
        _record_completion_decision(
            store,
            decision="skipped",
            reason="goal_active",
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=active_summary is not None,
            marker_present=active_marker is not None,
            now=now,
        )
        return NotifyResult("skipped", "goal_active")

    summary = active_summary
    live = store.get_live_completion_card(route)
    update_live = getattr(lark, "update_live_completion_card", None)
    if live and update_live:
        live = _flush_live_commentary(
            store,
            route,
            live,
            hook_stdin=hook_stdin,
            cwd=cwd,
            now=now,
        )
        last_assistant_message = _extract_stdin_string_field(
            hook_stdin,
            "last_assistant_message",
        )
        fresh_summary = (
            summary
            if summary
            and _runtime_record_is_fresh(summary, "staged_at", max_age_seconds, now)
            else None
        )
        answer = last_assistant_message or (
            fresh_summary.get("summary_markdown") if fresh_summary else None
        )
        answer = answer or "本轮已结束，但没有可用的最终答复摘要。"
        try:
            update_live(
                message_id=live["message_id"],
                status="completed",
                activities=live.get("activities") or [],
                activity_count=live.get("activity_count"),
                started_at=live["started_at"],
                now=_to_utc(now).isoformat(),
                cwd=cwd,
                answer=answer,
            )
        except Exception:
            # Keep the existing card and its state for a later retry. Sending a
            # second completion message here leaves a stale working card behind.
            return NotifyResult("update_failed", "live_card_update_failed")
        else:
            store.delete_live_completion_card(route)
            store.delete_completion_summary(route)
            store.delete_completion_prompt_marker(route)
            if summary:
                store.delete_staged_summary_by_hash(summary.get("cwd_hash"))
            if active_marker:
                store.delete_prompt_marker_by_hash(active_marker.get("cwd_hash"))
            _record_completion_decision(
                store,
                decision="live_card_updated",
                reason=None,
                route=route,
                cwd=cwd,
                goal_status=goal_status,
                summary_present=summary is not None,
                marker_present=active_marker is not None,
                now=now,
            )
            return NotifyResult("live_card_updated")

    if summary and _runtime_record_is_fresh(summary, "staged_at", max_age_seconds, now):
        send_live = getattr(lark, "send_live_completion_card", None)
        if send_live:
            timestamp = _to_utc(now).isoformat()
            send_live(
                status="completed",
                activities=[],
                started_at=timestamp,
                now=timestamp,
                cwd=cwd,
                answer=summary["summary_markdown"],
            )
        else:
            lark.send_summary_card(summary["summary_markdown"], cwd=cwd)
        store.delete_completion_summary(route)
        store.delete_completion_prompt_marker(route)
        store.delete_staged_summary_by_hash(summary.get("cwd_hash"))
        if active_marker:
            store.delete_prompt_marker_by_hash(active_marker.get("cwd_hash"))
        _record_completion_decision(
            store,
            decision="summary_sent",
            reason=None,
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=True,
            marker_present=active_marker is not None,
            now=now,
        )
        return NotifyResult("summary_sent")
    if summary:
        store.delete_completion_summary(route)
        store.delete_staged_summary_by_hash(summary.get("cwd_hash"))
        _record_completion_decision(
            store,
            decision="skipped",
            reason="summary_stale",
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=True,
            marker_present=active_marker is not None,
            now=now,
        )

    marker = active_marker
    if marker is None:
        _record_completion_decision(
            store,
            decision="skipped",
            reason="summary_missing",
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=False,
            marker_present=False,
            now=now,
        )
        return NotifyResult("skipped", "summary_missing")
    if not _runtime_record_is_fresh(marker, "marked_at", prompt_marker_max_age_seconds, now):
        store.delete_completion_prompt_marker(route)
        store.delete_prompt_marker_by_hash(marker.get("cwd_hash"))
        _record_completion_decision(
            store,
            decision="skipped",
            reason="prompt_marker_stale",
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=False,
            marker_present=True,
            now=now,
        )
        return NotifyResult("skipped", "prompt_marker_stale")

    if goal_status == "unknown":
        store.delete_completion_prompt_marker(route)
        store.delete_prompt_marker_by_hash(marker.get("cwd_hash"))
        _record_completion_decision(
            store,
            decision="skipped",
            reason="summary_missing_goal_unknown",
            route=route,
            cwd=cwd,
            goal_status=goal_status,
            summary_present=False,
            marker_present=True,
            now=now,
        )
        return NotifyResult("skipped", "summary_missing_goal_unknown")

    lark.send_fallback_card(cwd)
    store.delete_completion_prompt_marker(route)
    store.delete_prompt_marker_by_hash(marker.get("cwd_hash"))
    _record_completion_decision(
        store,
        decision="fallback_sent",
        reason="summary_missing",
        route=route,
        cwd=cwd,
        goal_status=goal_status,
        summary_present=False,
        marker_present=True,
        now=now,
    )
    return NotifyResult("fallback_sent", "summary_missing")


def _fresh_completion_summary_for_stop(
    store: StateStore,
    *,
    cwd: str,
    hook_stdin: str | bytes | None,
    now: datetime,
) -> dict | None:
    route = resolve_completion_route(cwd=cwd, hook_stdin=hook_stdin)
    summary = store.get_completion_summary(route)
    if summary and _runtime_record_is_fresh(summary, "staged_at", DEFAULT_SUMMARY_MAX_AGE_SECONDS, now):
        return summary
    legacy_summary = store.get_staged_summary(cwd)
    if legacy_summary and _runtime_record_is_fresh(
        legacy_summary,
        "staged_at",
        DEFAULT_SUMMARY_MAX_AGE_SECONDS,
        now,
    ):
        return legacy_summary
    return None


def send_away_early_exit_if_needed(
    paths,
    lark,
    cwd: str,
    now: datetime,
    hook_stdin: str | bytes | None = None,
) -> NotifyResult | None:
    store = _runtime_store(paths)
    codex_session_id = (
        _extract_stdin_string_field(hook_stdin, "session_id")
        or _extract_stdin_string_field(hook_stdin, "thread_id")
        or _extract_stdin_string_field(hook_stdin, "turn_id")
    )
    sessions = store.find_active_away_sessions(
        cwd=cwd,
        codex_session_id=codex_session_id,
    )
    if not sessions:
        return None

    for session in sessions:
        window = store.get_window(str(session.get("active_window_id") or ""))
        if not window:
            _record_away_stop_ignored(
                store,
                session=session,
                reason="active_window_missing",
                now=now,
            )
            return NotifyResult("away_active_stop_ignored", "active_window_missing")

        deadline = _parse_iso_time(session.get("deadline_at") or window.get("deadline_at"))
        lease = store.get_waiter_lease(str(session["session_id"]))
        if _lease_is_alive(lease, now):
            _record_away_stop_ignored(
                store,
                session=session,
                reason="waiter_alive",
                now=now,
            )
            return NotifyResult("away_active_stop_ignored", "waiter_alive")

        if deadline and deadline <= _to_utc(now):
            summary = _fresh_completion_summary_for_stop(
                store,
                cwd=cwd,
                hook_stdin=hook_stdin,
                now=now,
            )
            if summary:
                _close_away_for_stop(
                    store,
                    session=session,
                    window=window,
                    reason="stale_timeout",
                    status="timed_out",
                    closed_at=now,
                )
                store.record_diagnostic_event(
                    event_kind="away_deadline_closed_completion_allowed",
                    severity="warning",
                    message="Stop hook closed a stale Away Session and allowed completion notification to continue.",
                    detail={
                        "session_id": session.get("session_id"),
                        "window_id": window.get("window_id"),
                        "reason": "stale_timeout",
                    },
                    created_at=_to_utc(now).isoformat(),
                )
                return None
            if hasattr(lark, "send_away_timeout_card"):
                lark.send_away_timeout_card(
                    {
                        "project": session.get("project") or "Codex Away Mode",
                        "cwd": session.get("cwd") or cwd,
                        "codex_session_id": session.get("codex_session_id") or codex_session_id,
                        "deadline": deadline,
                    }
                )
            _close_away_for_stop(
                store,
                session=session,
                window=window,
                reason="stale_timeout",
                status="timed_out",
                closed_at=now,
            )
            store.record_diagnostic_event(
                event_kind="away_deadline_closed",
                severity="warning",
                message="Stop hook closed an Away Session whose deadline had passed.",
                detail={
                    "session_id": session.get("session_id"),
                    "window_id": window.get("window_id"),
                    "reason": "stale_timeout",
                },
                created_at=_to_utc(now).isoformat(),
            )
            return NotifyResult("away_deadline_closed", "stale_timeout")

        _record_away_stop_ignored(
            store,
            session=session,
            reason="insufficient_evidence",
            now=now,
        )
        return NotifyResult("away_active_stop_ignored", "insufficient_evidence")
    return None


def capture_hook_payload(
    paths,
    *,
    event_kind: str,
    hook_stdin: str | bytes | None,
    cwd: str,
    now: datetime,
) -> None:
    if not hook_stdin:
        return
    try:
        config = load_config(Path(paths.config_path))
        if not config.capture_hook_payloads:
            return
        record = {
            "captured_at": _to_utc(now).isoformat(),
            "event_kind": event_kind,
            "resolved_cwd": cwd,
            "payload": _redacted_hook_payload(hook_stdin),
        }
        _append_private_jsonl(Path(paths.log_dir) / "hook-payload-samples.jsonl", record)
    except Exception:
        return


def goal_status_from_hook_stdin(hook_stdin: str | bytes | None) -> str:
    transcript_path = _extract_stdin_string_field(hook_stdin, "transcript_path")
    if not transcript_path:
        return "unknown"
    return goal_status_from_transcript(Path(transcript_path))


def goal_status_from_transcript(path) -> str:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return "unknown"

    call_names: dict[str, str] = {}
    latest_status: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict) or record.get("type") != "response_item":
                    continue
                item_type = payload.get("type")
                if item_type == "function_call" and payload.get("name") in {
                    "create_goal",
                    "get_goal",
                    "update_goal",
                }:
                    call_id = payload.get("call_id")
                    if isinstance(call_id, str):
                        call_names[call_id] = payload["name"]
                    continue
                if item_type != "function_call_output":
                    continue
                call_id = payload.get("call_id")
                if not isinstance(call_id, str) or call_id not in call_names:
                    continue
                status = _goal_status_from_tool_output(payload.get("output"))
                if status in {"active", "blocked", "complete", "none"}:
                    latest_status = status
    except OSError:
        return "unknown"
    return latest_status or "none"


def skip_cwd_reason(paths, cwd: str | None) -> str | None:
    if not cwd or not os.path.isabs(cwd):
        return "non_user_workspace"
    path = _resolve_path(cwd)
    if path == Path("/"):
        return "non_user_workspace"

    codex_home = _resolve_path(getattr(paths, "codex_home", Path.home() / ".codex"))
    data_dir = _resolve_path(getattr(paths, "data_dir", codex_home / "codex-away-mode"))
    tmp_paths = {
        _resolve_path("/tmp"),
        _resolve_path("/private/tmp"),
        _resolve_path(tempfile.gettempdir()),
    }
    if _is_within(path, codex_home) or _is_within(path, data_dir):
        return "non_user_workspace"
    if any(path == tmp_path or _is_within(path, tmp_path) for tmp_path in tmp_paths):
        return "non_user_workspace"
    return None


def send_test_notification(paths, lark):
    if hasattr(lark, "send_test_notification"):
        result = lark.send_test_notification()
    else:
        config = load_config(Path(paths.config_path))
        card = cards.completion_card(
            title="Codex Away Mode 测试通知",
            fields={"完成": "测试通知已发送。"},
            footer_mode_text=cards.notification_mode_footer_text(config.notification_mode),
            now=datetime.now(timezone.utc),
        )
        if config.feishu_chat_id:
            result = lark.send_interactive_card(chat_id=config.feishu_chat_id, card=card)
        elif config.feishu_user_id:
            result = lark.send_interactive_card(user_id=config.feishu_user_id, card=card)
        else:
            raise RuntimeError("Missing feishu_chat_id or feishu_user_id; run setup feishu first.")
    chat_id = getattr(result, "chat_id", None)
    if chat_id:
        config = load_config(Path(paths.config_path))
        config.feishu_chat_id = chat_id
        save_config(Path(paths.config_path), config)
    return result


def set_notification_mode(
    paths,
    mode: str,
    *,
    until: datetime | None = None,
) -> AppConfig:
    if mode not in {"all", "off", "snooze"}:
        raise ValueError(f"unsupported notification mode: {mode}")

    config_path = Path(paths.config_path)
    config = load_config(config_path)
    if mode == "snooze":
        if until is None:
            raise ValueError("snooze requires until")
        config.notification_mode = "all"
        config.snooze_until = _to_utc(until).isoformat()
    else:
        config.notification_mode = mode
        config.snooze_until = None
    save_config(config_path, config)
    return config


def effective_notification_mode(paths, now: datetime | None = None) -> str:
    return _config_mode(load_config(Path(paths.config_path)), now=now)


def _close_away_for_stop(
    store: StateStore,
    *,
    session: dict,
    window: dict,
    reason: str,
    status: str,
    closed_at: datetime,
) -> None:
    closed_at_text = _to_utc(closed_at).isoformat()
    store.close_away_session(
        session["session_id"],
        status=status,
        reason=reason,
        closed_at=closed_at_text,
    )
    store.close_active_card(window["window_id"], closed_at=closed_at_text)
    store.close_window(
        window["window_id"],
        status=status,
        reason=reason,
        closed_at=closed_at_text,
    )


def _record_away_stop_ignored(
    store: StateStore,
    *,
    session: dict,
    reason: str,
    now: datetime,
) -> None:
    store.record_diagnostic_event(
        event_kind="away_active_stop_ignored",
        severity="info",
        message="Stop hook ignored an active Away Session.",
        detail={
            "session_id": session.get("session_id"),
            "status": session.get("status"),
            "reason": reason,
        },
        created_at=_to_utc(now).isoformat(),
    )


def _lease_is_alive(lease: dict | None, now: datetime) -> bool:
    if not lease:
        return False
    expires_at = _parse_iso_time(lease.get("expires_at"))
    return bool(expires_at and expires_at > _to_utc(now))


def _parse_iso_time(value) -> datetime | None:
    if not value:
        return None
    try:
        return _to_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError:
        return None


def _runtime_record_is_fresh(
    record: dict,
    timestamp_key: str,
    max_age_seconds: int,
    now: datetime,
) -> bool:
    expires_at = _parse_iso_time(record.get("expires_at"))
    now_utc = _to_utc(now)
    if expires_at is not None:
        return expires_at >= now_utc
    timestamp = _parse_iso_time(record.get(timestamp_key))
    if timestamp is None:
        return False
    age = now_utc.timestamp() - timestamp.timestamp()
    return 0 <= age <= max_age_seconds


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _extract_stdin_cwd(hook_stdin: str | bytes | None) -> str | None:
    return _extract_stdin_string_field(hook_stdin, "cwd")


def _json_payload(hook_stdin: str | bytes | None):
    if not hook_stdin:
        return None
    if isinstance(hook_stdin, bytes):
        hook_stdin = hook_stdin.decode("utf-8", errors="replace")
    try:
        return json.loads(hook_stdin)
    except json.JSONDecodeError:
        return None


def _extract_stdin_string_field(hook_stdin: str | bytes | None, field: str) -> str | None:
    payload = _json_payload(hook_stdin)
    if not isinstance(payload, dict):
        return None
    value = payload.get(field)
    return value if isinstance(value, str) else None


def _optional_string(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_or_default(value, default: str) -> str:
    return _optional_string(value) or default


def _permission_command(tool_name: str, tool_input: dict) -> str | None:
    for key in ("command", "cmd", "path", "file_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if tool_input:
        compact: dict[str, str] = {}
        for key in ("description", "explanation", "intent", "path", "file_path"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                compact[key] = value.strip()
        if compact:
            return json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return f"Codex 请求使用 {tool_name} 执行一项需要审批的操作"


def _permission_description(tool_input: dict) -> str | None:
    for key in ("description", "explanation", "intent"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _redact_for_display(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    patterns = [
        r"sk-[A-Za-z0-9_\-]{8,}",
        r"xox[baprs]-[A-Za-z0-9_\-]{8,}",
        r"Bearer\s+[A-Za-z0-9._\-]+",
        r"Authorization:\s*[^\s]+",
        r"(OPENAI_API_KEY|FEISHU_[A-Z0-9_]*|LARK_[A-Z0-9_]*|app_secret|tenant_access_token|user_access_token|refresh_token)=([^\s]+)",
        r"\b(?:ou|oc)_[A-Za-z0-9_\-]{8,}\b",
    ]
    for pattern in patterns:
        text = re.sub(pattern, _redaction_replacement, text, flags=re.IGNORECASE)
    return text


def _redaction_replacement(match: re.Match) -> str:
    if match.lastindex and match.lastindex >= 1 and "=" in match.group(0):
        return f"{match.group(1)}=[REDACTED]"
    return "[REDACTED]"


def _truncate_display(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _goal_status_from_tool_output(output) -> str | None:
    if not isinstance(output, str):
        return None
    start = output.find("{")
    if start == -1:
        return None
    try:
        payload = json.loads(output[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "goal" not in payload:
        return None
    goal = payload.get("goal")
    if goal is None:
        return "none"
    if not isinstance(goal, dict):
        return None
    status = goal.get("status")
    return status if status in {"active", "blocked", "complete"} else None


def _redacted_hook_payload(hook_stdin: str | bytes):
    if isinstance(hook_stdin, bytes):
        hook_text = hook_stdin.decode("utf-8", errors="replace")
    else:
        hook_text = hook_stdin
    try:
        payload = json.loads(hook_text)
    except json.JSONDecodeError:
        return _redacted_string(None, hook_text)
    return _redact_value(payload, key=None, depth=0)


def _redact_value(value, *, key: str | None, depth: int):
    if depth > _MAX_CAPTURE_DEPTH:
        return {"type": type(value).__name__, "truncated": "depth"}
    if isinstance(value, dict):
        return {
            str(child_key): _redact_value(child_value, key=str(child_key), depth=depth + 1)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "items": [
                _redact_value(item, key=key, depth=depth + 1)
                for item in value[:_MAX_CAPTURE_LIST_ITEMS]
            ],
        }
    if isinstance(value, str):
        return _redacted_string(key, value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return {"type": type(value).__name__}


def _redacted_string(key: str | None, value: str):
    if key in _SAFE_CAPTURE_VALUE_KEYS and len(value) <= 240 and "\n" not in value:
        return value
    return {
        "type": "string",
        "length": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
    }


def _append_private_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        os.chmod(path, 0o600)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _resolve_path(value) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
