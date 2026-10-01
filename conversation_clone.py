from __future__ import annotations

import copy
import asyncio
import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable


SCOPES = {"all", "group", "private", "auto"}


@dataclass(frozen=True)
class UMO:
    raw: str
    platform_id: str
    message_type: str
    session_id: str

    @property
    def detected_scope(self) -> str:
        value = self.message_type.casefold()
        if value == "groupmessage":
            return "group"
        if value == "friendmessage":
            return "private"
        return "unknown"


@dataclass(frozen=True)
class CloneMapping:
    source_umo: UMO
    target_umos: tuple[UMO, ...]
    scope: str = "auto"
    revision: int = 1
    mapping_key: str = ""

    def job_key(self, target: UMO) -> str:
        return stable_hash(
            {
                "mapping_key": self.mapping_key,
                "source_umo": self.source_umo.raw,
                "target_umo": target.raw,
                "revision": self.revision,
            }
        )


@dataclass
class CloneResult:
    job_key: str
    target_conversation_id: str
    conversation: Any
    status: str


class CloneError(RuntimeError):
    """Raised when an AstrBot API operation prevents a complete clone."""


def parse_umo(value: str) -> UMO:
    if not isinstance(value, str):
        raise ValueError("UMO 必须是字符串")
    raw = str(value or "").strip()
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise ValueError(f"UMO 格式无效：{raw!r}")
    if any(part != part.strip() for part in parts):
        raise ValueError("UMO 字段两侧不能包含空格")
    return UMO(raw, *(part.strip() for part in parts))


def parse_json(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value or ""))
    except (TypeError, ValueError):
        return default


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_scope(source: UMO, scope: str) -> str:
    normalized = str(scope or "auto").strip().casefold()
    if normalized not in SCOPES:
        raise ValueError(f"scope 必须是 all、group、private 或 auto：{scope!r}")
    detected = source.detected_scope
    if normalized == "all":
        return detected
    if detected == "unknown":
        raise ValueError(f"无法从 A UMO 自动识别群聊或私聊类型：{source.raw}")
    if normalized in {"group", "private"} and normalized != detected:
        raise ValueError(f"A UMO {source.raw} 的类型是 {detected}，与 scope={normalized} 不匹配")
    return detected


def parse_clone_mappings(value: Any) -> list[CloneMapping]:
    try:
        payload = value if isinstance(value, list) else json.loads(value or "[]")
    except (ValueError, TypeError) as exc:
        raise ValueError("clone_mappings_json 不是有效的 JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("clone_mappings_json 必须是 JSON 数组")

    mappings: list[CloneMapping] = []
    source_owner: set[str] = set()
    target_owner: dict[str, str] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index + 1} 个映射必须是对象")
        try:
            source = parse_umo(item.get("source_umo", ""))
        except ValueError as exc:
            raise ValueError(f"第 {index + 1} 个映射的 source_umo 无效：{exc}") from exc
        targets = item.get("target_umos", [])
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"第 {index + 1} 个映射必须包含非空 target_umos")
        scope = str(item.get("scope", "auto") or "auto").strip().casefold()
        validate_scope(source, scope)
        revision = item.get("revision", 1)
        if isinstance(revision, bool) or not isinstance(revision, int) or not 1 <= revision <= 2147483647:
            raise ValueError(f"第 {index + 1} 个映射的 revision 必须是正整数")
        if source.raw in source_owner:
            raise ValueError(f"source_umo 重复配置：{source.raw}")
        source_owner.add(source.raw)

        normalized_targets: list[UMO] = []
        for raw_target in targets:
            target = parse_umo(str(raw_target))
            if target.raw == source.raw:
                raise ValueError(f"映射不能把 UMO 复制给自身：{source.raw}")
            if target.raw in target_owner:
                raise ValueError(f"目标 UMO 同时属于多个映射：{target.raw}")
            target_owner[target.raw] = source.raw
            normalized_targets.append(target)

        identity = {
            "source_umo": source.raw,
            "scope": scope,
        }
        mappings.append(
            CloneMapping(
                source_umo=source,
                target_umos=tuple(normalized_targets),
                scope=scope,
                revision=revision,
                mapping_key=stable_hash(identity),
            )
        )
    return mappings


def mapping_for_target(mappings: Iterable[CloneMapping], target_umo: str) -> CloneMapping | None:
    for mapping in mappings:
        if any(target.raw == target_umo for target in mapping.target_umos):
            return mapping
    return None


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _signature(method: Callable[..., Any]) -> inspect.Signature | None:
    try:
        return inspect.signature(method)
    except (TypeError, ValueError):
        return None


def _call_with_umo(method: Callable[..., Any], umo: str, values: dict[str, Any]) -> Any:
    """Call AstrBot methods while tolerating common parameter-name changes."""
    signature = _signature(method)
    if signature is None:
        return method(umo, **values)
    params = signature.parameters
    kwargs: dict[str, Any] = {}
    positional: list[Any] = []
    umo_names = ("unified_msg_origin", "umo", "source_umo")
    selected_umo_name = next((name for name in umo_names if name in params), None)
    if selected_umo_name:
        # Some AstrBot versions expose the UMO as positional-only, while
        # decorated methods may only expose a generic *args/**kwargs wrapper.
        # Passing the UMO positionally covers both cases.
        umo_parameter = params[selected_umo_name]
        if umo_parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional.append(umo)
        else:
            kwargs[selected_umo_name] = umo
    aliases = {
        "conversation_id": ("conversation_id", "cid"),
        "platform_id": ("platform_id", "platform"),
        "user_id": ("user_id", "session_id"),
        "history": ("history",),
        "title": ("title",),
        "persona_id": ("persona_id",),
        "token_usage": ("token_usage",),
        "page": ("page", "page_number"),
        "page_size": ("page_size", "limit", "size"),
        "offset": ("offset",),
        "cursor": ("cursor",),
        "message": ("message", "record", "item"),
    }
    if not selected_umo_name and params:
        first = next(iter(params.values()))
        first_is_value = first.name in {alias for names in aliases.values() for alias in names} or first.name in values
        if not first_is_value and first.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            positional.append(umo)
    has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    for key, value in values.items():
        name = next((candidate for candidate in aliases.get(key, (key,)) if candidate in params), None)
        if name is not None:
            kwargs[name] = value
        elif has_var_kwargs:
            kwargs[key] = value
    return method(*positional, **kwargs)


async def get_conversations(manager: Any, umo: str) -> list[Any]:
    method = getattr(manager, "get_conversations", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 get_conversations")
    result = await _maybe_await(_call_with_umo(method, umo, {}))
    return list(result or [])


async def get_current_conversation_id(manager: Any, umo: str) -> str | None:
    method = getattr(manager, "get_curr_conversation_id", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 get_curr_conversation_id")
    result = await _maybe_await(_call_with_umo(method, umo, {}))
    return str(result) if result else None


async def get_conversation(manager: Any, umo: str, conversation_id_value: str) -> Any:
    method = getattr(manager, "get_conversation", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 get_conversation")
    return await _maybe_await(
        _call_with_umo(method, umo, {"conversation_id": conversation_id_value})
    )


async def new_conversation(manager: Any, umo: UMO, title: str) -> str:
    method = getattr(manager, "new_conversation", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 new_conversation")
    result = await _maybe_await(
        _call_with_umo(method, umo.raw, {"platform_id": umo.platform_id, "title": title})
    )
    cid = getattr(result, "cid", None) or getattr(result, "conversation_id", None) or result
    if not cid:
        raise CloneError("new_conversation 未返回有效 conversation_id")
    return str(cid)


async def update_conversation(manager: Any, umo: UMO, target_cid: str, source: Any) -> None:
    method = getattr(manager, "update_conversation", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 update_conversation")
    values = {
        "conversation_id": target_cid,
        "history": conversation_history(source),
        "title": str(getattr(source, "title", "") or ""),
        "persona_id": getattr(source, "persona_id", None),
        "token_usage": getattr(source, "token_usage", 0) or 0,
    }
    await _maybe_await(_call_with_umo(method, umo.raw, values))


async def switch_conversation(manager: Any, umo: str, conversation_id_value: str) -> None:
    method = getattr(manager, "switch_conversation", None)
    if not callable(method):
        raise CloneError("AstrBot ConversationManager 缺少 switch_conversation")
    await _maybe_await(
        _call_with_umo(method, umo, {"conversation_id": conversation_id_value})
    )


def conversation_id(conversation: Any) -> str:
    value = getattr(conversation, "cid", None) or getattr(conversation, "conversation_id", None)
    return str(value or "")


def conversation_history(conversation: Any) -> list[Any]:
    raw = getattr(conversation, "history", "[]")
    if isinstance(raw, list):
        value = raw
    else:
        try:
            value = json.loads(raw or "[]")
        except (TypeError, ValueError) as exc:
            raise CloneError("对话历史不是有效 JSON，已停止以避免生成空副本") from exc
    if not isinstance(value, list):
        raise CloneError("对话历史必须是数组")
    return copy.deepcopy(value)


def conversation_sort_key(conversation: Any) -> tuple[float, float, str]:
    def timestamp(value):
        if hasattr(value, "timestamp"):
            return value.timestamp()
        return float(value or 0)
    created = timestamp(getattr(conversation, "created_at", 0))
    updated = timestamp(getattr(conversation, "updated_at", created))
    return created, updated, conversation_id(conversation)


def _record_as_json(record: Any) -> Any:
    if isinstance(record, dict):
        return record
    for method_name in ("model_dump", "dict"):
        method = getattr(record, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    if hasattr(record, "__dict__"):
        return vars(record)
    return str(record)


def platform_record_fingerprint(source_umo: str, record: Any) -> str:
    return stable_hash({"source_umo": source_umo, "record": _record_as_json(record)})


def _page_records(value: Any) -> tuple[list[Any], Any, bool | None]:
    if value is None:
        return [], None, False
    if isinstance(value, (list, tuple)):
        return list(value), None, None
    if isinstance(value, dict):
        records = value.get("items", value.get("messages", value.get("data", [])))
        if isinstance(records, dict):
            records = records.get("items", records.get("messages", []))
        if not isinstance(records, (list, tuple)):
            records = []
        return list(records), value.get("next_cursor"), value.get("has_more")
    for name in ("items", "messages", "records", "data"):
        records = getattr(value, name, None)
        if isinstance(records, (list, tuple)):
            return list(records), getattr(value, "next_cursor", None), getattr(value, "has_more", None)
    return [], None, False


async def get_platform_history(manager: Any, umo: UMO, page_size: int = 100) -> list[Any]:
    method = getattr(manager, "get", None)
    if not callable(method):
        raise CloneError("AstrBot PlatformMessageHistoryManager 缺少 get")
    all_records: list[Any] = []
    page = 1
    cursor: Any = None
    seen_pages = set()
    while True:
        values: dict[str, Any] = {
            "platform_id": umo.platform_id,
            "user_id": umo.raw,
            "page": page,
            "page_size": page_size,
        }
        if cursor is not None:
            values["cursor"] = cursor
        result = await _maybe_await(_call_with_umo(method, umo.raw, values))
        records, next_cursor, has_more = _page_records(result)
        if records:
            page_fingerprint = stable_hash([_record_as_json(record) for record in records])
            if page_fingerprint in seen_pages:
                raise CloneError("平台历史接口重复返回同一页，无法继续分页")
            seen_pages.add(page_fingerprint)
        all_records.extend(records)
        if not records:
            break
        if next_cursor is not None and next_cursor != cursor:
            cursor = next_cursor
        elif has_more is True or len(records) >= page_size:
            page += 1
        else:
            break
        if len(seen_pages) > 100000:
            raise CloneError("PlatformMessageHistory 分页超过安全上限")
    return all_records


def adapt_platform_record(record: Any, target: UMO) -> Any:
    result = copy.copy(record)
    fields = {
        "platform_id": target.platform_id,
        "user_id": target.raw,
        "session_id": target.session_id,
        "unified_msg_origin": target.raw,
        "llm_checkpoint_id": None,
        "id": None,
        "pk": None,
    }
    if isinstance(result, dict):
        result = dict(result)
        for key, value in fields.items():
            if key in result:
                result[key] = value
        for key in ("id", "pk", "message_id", "llm_checkpoint_id"):
            result.pop(key, None)
        return result
    for key, value in fields.items():
        if hasattr(result, key):
            try:
                setattr(result, key, value)
            except Exception:
                pass
    return result


def _platform_insert_values(record: Any, target: UMO) -> dict[str, Any]:
    raw = _record_as_json(record)
    if isinstance(raw, dict):
        content = copy.deepcopy(raw.get("content", {}))
        sender_id = raw.get("sender_id")
        sender_name = raw.get("sender_name")
    else:
        content = {}
        sender_id = None
        sender_name = None
    return {
        "platform_id": target.platform_id,
        "user_id": target.raw,
        "content": content,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "llm_checkpoint_id": None,
        "message": adapt_platform_record(record, target),
    }


async def insert_platform_record(manager: Any, target: UMO, record: Any) -> None:
    method = getattr(manager, "insert", None)
    if not callable(method):
        raise CloneError("AstrBot PlatformMessageHistoryManager 缺少 insert")
    values = _platform_insert_values(record, target)
    signature = _signature(method)
    if signature is not None:
        names = set(signature.parameters)
        if {"platform_id", "user_id", "content"}.issubset(names):
            kwargs = {key: value for key, value in values.items() if key in names}
            result = method(**kwargs)
        elif "unified_msg_origin" in names or "umo" in names:
            result = _call_with_umo(method, target.raw, {"message": values["message"]})
        elif {"platform_id", "user_id"}.issubset(names):
            result = method(target.platform_id, target.session_id, values["message"])
        else:
            result = method(values["message"])
    else:
        result = method(
            target.platform_id,
            target.raw,
            values["content"],
            values["sender_id"],
            values["sender_name"],
            None,
        )
    await _maybe_await(result)


class ConversationCloner:
    def __init__(self, context: Any, storage: Any, page_size: int = 100):
        self.context = context
        self.storage = storage
        self.page_size = max(1, int(page_size))

    @property
    def conversation_manager(self) -> Any:
        return getattr(self.context, "conversation_manager", None)

    @property
    def history_manager(self) -> Any:
        return getattr(self.context, "message_history_manager", None)

    async def ensure_clone(self, mapping: CloneMapping, target: UMO) -> CloneResult:
        validate_scope(mapping.source_umo, mapping.scope)
        manager = self.conversation_manager
        history_manager = self.history_manager
        job = await self.storage.get_or_create_job(
            mapping.job_key(target),
            mapping.mapping_key,
            mapping.source_umo.raw,
            target.raw,
            mapping.scope,
            mapping.revision,
        )
        job_key = job["job_key"]
        if job.get("status") == "complete" and job.get("target_conversation_id"):
            # B may have switched, created or deleted conversations since cloning.
            # A completed job must not pin B to the original copied conversation.
            cid = await get_current_conversation_id(manager, target.raw)
            conversation = await get_conversation(manager, target.raw, cid) if cid else None
            return CloneResult(job_key, cid, conversation, "already_complete")

        await self.storage.begin_attempt(job_key)
        original_target_cid = None
        created_cids = set()
        try:
            required = ("get_conversations", "get_conversation", "get_curr_conversation_id",
                        "new_conversation", "update_conversation", "switch_conversation")
            if any(not callable(getattr(manager, name, None)) for name in required):
                raise CloneError("AstrBot ConversationManager API 不完整，无法复制")
            if any(not callable(getattr(history_manager, name, None)) for name in ("get", "insert")):
                raise CloneError("AstrBot PlatformMessageHistory API 不可用，无法复制")
            original_target_cid = await get_current_conversation_id(manager, target.raw)
            source_conversations = sorted(
                await get_conversations(manager, mapping.source_umo.raw),
                key=conversation_sort_key,
            )
            if not source_conversations:
                raise CloneError("A UMO 没有可复制的 conversation，请检查完整 UMO 和来源历史")
            source_current = await get_current_conversation_id(manager, mapping.source_umo.raw)
            await self.storage.set_source_current(job_key, source_current)
            await self.storage.mark_job(job_key, "running", phase="conversations",
                                        conversation_total=len(source_conversations))

            source_to_target: dict[str, str] = {}
            for source_conversation in source_conversations:
                source_cid = conversation_id(source_conversation)
                if not source_cid:
                    continue
                saved = await self.storage.get_conversation_map(job_key, source_cid)
                if saved and saved["status"] == "complete":
                    source_to_target[source_cid] = str(saved["target_cid"])
                    continue
                title = str(getattr(source_conversation, "title", "") or "")
                if saved:
                    target_cid = str(saved["target_cid"])
                else:
                    # Validate history before allocating an empty target conversation.
                    conversation_history(source_conversation)
                    target_cid = await new_conversation(manager, target, title or f"复制对话 {source_cid}")
                    created_cids.add(target_cid)
                    await self.storage.save_conversation_map(
                        job_key, source_cid, target_cid, title, status="pending")
                await update_conversation(manager, target, target_cid, source_conversation)
                await self.storage.save_conversation_map(
                    job_key, source_cid, target_cid, title,
                    message_count=len(conversation_history(source_conversation)))
                source_to_target[source_cid] = target_cid

            await self.storage.mark_job(job_key, "running", phase="platform_history")
            records = await get_platform_history(history_manager, mapping.source_umo, self.page_size)
            await self.storage.mark_job(job_key, "running", platform_total=len(records))
            for record in records:
                fingerprint = platform_record_fingerprint(mapping.source_umo.raw, record)
                if await self.storage.has_platform_record(job_key, fingerprint):
                    continue
                await insert_platform_record(history_manager, target, record)
                await self.storage.save_platform_record(job_key, fingerprint)

            target_current = source_to_target.get(source_current or "")
            if target_current is None and source_to_target:
                target_current = next(reversed(source_to_target.values()))
            if not target_current:
                raise CloneError("A UMO 没有可复制的 conversation")
            await self.storage.mark_job(job_key, "running", phase="switching")
            await switch_conversation(manager, target.raw, target_current)
            target_conversation = await get_conversation(manager, target.raw, target_current)
            await self.storage.mark_job(
                job_key,
                "complete",
                phase="complete",
                target_conversation_id=target_current,
                error=None,
            )
            return CloneResult(job_key, target_current, target_conversation, "complete")
        except asyncio.CancelledError:
            progress = await self.storage.progress(job_key)
            status = "partial" if progress["allocated_conversations"] or progress["platform_done"] else "pending"
            await self.storage.mark_job(job_key, status, error="插件已停止，重载后可继续")
            raise
        except Exception as exc:
            if original_target_cid and created_cids:
                current = await get_current_conversation_id(manager, target.raw)
                if current in created_cids:
                    await switch_conversation(manager, target.raw, original_target_cid)
            progress = await self.storage.progress(job_key)
            status = "partial" if progress["allocated_conversations"] or progress["platform_done"] else "failed"
            await self.storage.mark_job(job_key, status, error=str(exc))
            if isinstance(exc, CloneError):
                raise
            raise CloneError(f"UMO 复制失败：{exc}") from exc
