from __future__ import annotations

import asyncio
import inspect
import json
import os
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api.web import error_response, json_response, request

try:
    from .conversation_clone import (
        CloneError, ConversationCloner, conversation_history, conversation_id,
        get_conversation, mapping_for_target, parse_clone_mappings,
    )
    from .storage import PluginStorage
except ImportError:  # AstrBot versions that load main.py as a standalone module
    from conversation_clone import (  # type: ignore[no-redef]
        CloneError,
        ConversationCloner,
        conversation_history,
        conversation_id,
        get_conversation,
        mapping_for_target,
        parse_clone_mappings,
    )
    from storage import PluginStorage  # type: ignore[no-redef]


PLUGIN_ID = "astrbot_plugin_cross_platform_share"
PLUGIN_VERSION = "2.3.0"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))


@register(
    PLUGIN_ID,
    "AstrBot plugin community",
    "按 UMO 将一个用户的 AstrBot 对话独立复制给另一个用户",
    PLUGIN_VERSION,
)
class CrossPlatformShare(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.storage: PluginStorage | None = None
        self._storage_lock = asyncio.Lock()
        self._target_locks: dict[str, asyncio.Lock] = {}
        self._clone_tasks: dict[str, asyncio.Task] = {}
        self._config_error: str | None = None
        self._config_lock = asyncio.Lock()
        self._schedule_lock = asyncio.Lock()

        # Dashboard Page API. The Dashboard supplies authentication before these
        # handlers are called, so page users do not need a second token.
        context.register_web_api(
            f"/{PLUGIN_ID}/settings",
            self.page_settings,
            ["GET"],
            "读取 UMO 复制映射和任务状态",
        )
        context.register_web_api(
            f"/{PLUGIN_ID}/settings/save",
            self.page_settings_save,
            ["POST"],
            "保存 UMO 复制映射",
        )
        for endpoint, handler, description in (
            ("history", self.page_history, "分页查询复制记录"),
            ("history/detail", self.page_detail, "查询任务阶段和对话映射"),
            ("history/preview", self.page_preview, "只读查看目标对话历史"),
            ("jobs/retry", self.page_retry, "继续未完成任务"),
            ("jobs/recopy", self.page_recopy, "增加版本并创建新的独立副本"),
        ):
            context.register_web_api(f"/{PLUGIN_ID}/{endpoint}", handler, ["POST"], description)

    def _cfg(self, key: str, default: Any = None) -> Any:
        if hasattr(self.config, "get"):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    async def _ensure_storage(self) -> PluginStorage:
        if self.storage is not None:
            return self.storage
        async with self._storage_lock:
            if self.storage is None:
                directory = str(self._cfg("storage_dir", "") or "").strip()
                storage = PluginStorage(directory or os.path.join(PLUGIN_DIR, "data"))
                await storage.open()
                self.storage = storage
        return self.storage

    async def initialize(self) -> None:
        """Restore configured clone mappings and start unfinished jobs.

        The dashboard save endpoint schedules jobs immediately.  This startup
        pass covers mappings that were already saved before the plugin was
        loaded or that were interrupted by a restart.
        """
        try:
            mappings = parse_clone_mappings(self._cfg("clone_mappings_json", "[]"))
            scheduled = await self._schedule_clones(mappings)
            if scheduled:
                logger.info("[跨平台共享] 已从配置恢复并启动 %d 个复制任务", scheduled)
        except ValueError as exc:
            self._config_error = str(exc)
            logger.error("[跨平台共享] 配置无效，无法自动启动复制：%s", exc)
        except Exception:
            logger.exception("[跨平台共享] 启动时恢复复制任务失败")

    async def terminate(self) -> None:
        tasks = list(self._clone_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._clone_tasks.clear()
        if self.storage is not None:
            await self.storage.close()
            self.storage = None

    def _target_lock(self, umo: str) -> asyncio.Lock:
        lock = self._target_locks.get(umo)
        if lock is None:
            lock = asyncio.Lock()
            self._target_locks[umo] = lock
        return lock

    async def _clone_target(self, mapping, target) -> None:
        storage = await self._ensure_storage()
        async with self._target_lock(target.raw):
            try:
                configured = mapping_for_target(
                    parse_clone_mappings(self._cfg("clone_mappings_json", "[]")), target.raw)
                if configured is None or (configured.source_umo, configured.scope, configured.revision) != (
                        mapping.source_umo, mapping.scope, mapping.revision):
                    job = await storage.find_job(mapping.source_umo.raw, target.raw, mapping.scope, mapping.revision)
                    if job:
                        await storage.mark_job(job["job_key"], "failed", error="映射已变更或删除，排队任务未执行")
                    return
                await ConversationCloner(self.context, storage).ensure_clone(mapping, target)
            except CloneError as exc:
                logger.error("[跨平台共享] 保存映射后复制 %s 失败：%s", target.raw, exc)
            except Exception:
                logger.exception("[跨平台共享] 保存映射后复制 %s 时发生未预期错误", target.raw)

    async def _schedule_clones(self, mappings, only_target=None) -> int:
        async with self._schedule_lock:
            return await self._enqueue_clones(mappings, only_target)

    async def _enqueue_clones(self, mappings, only_target=None) -> int:
        """Create visible pending jobs, then run each target clone in the background."""
        storage = await self._ensure_storage()
        scheduled = 0
        for mapping in mappings:
            for target in mapping.target_umos:
                if only_target is not None and target.raw != only_target:
                    continue
                job_key = mapping.job_key(target)
                job = await storage.get_or_create_job(
                    job_key,
                    mapping.mapping_key,
                    mapping.source_umo.raw,
                    target.raw,
                    mapping.scope,
                    mapping.revision,
                )
                job_key = job["job_key"]
                if job.get("status") == "complete":
                    continue
                existing = self._clone_tasks.get(job_key)
                if existing is not None and not existing.done():
                    continue
                await storage.mark_job(job_key, "pending", error=None)
                task = asyncio.create_task(self._clone_target(mapping, target))
                self._clone_tasks[job_key] = task
                task.add_done_callback(self._task_done_callback(job_key))
                scheduled += 1
        return scheduled

    def _task_done_callback(self, target_umo: str):
        def callback(finished: asyncio.Task) -> None:
            # A retry may replace the entry before an older callback runs.
            # Only remove the task that is still registered for this target.
            if self._clone_tasks.get(target_umo) is finished:
                self._clone_tasks.pop(target_umo, None)

        return callback

    @staticmethod
    def _event_umo(event: AstrMessageEvent) -> str:
        return str(getattr(event, "unified_msg_origin", "") or "").strip()

    def _mapping_dicts(self) -> list[dict[str, Any]]:
        raw = self._cfg("clone_mappings_json", "[]")
        mappings = parse_clone_mappings(raw)
        return [
            {
                "source_umo": mapping.source_umo.raw,
                "target_umos": [target.raw for target in mapping.target_umos],
                "scope": mapping.scope,
                "revision": mapping.revision,
            }
            for mapping in mappings
        ]

    async def page_settings(self):
        try:
            mappings = self._mapping_dicts()
            storage = await self._ensure_storage()
            jobs = []
            for mapping in mappings:
                for target in mapping["target_umos"]:
                    job = await storage.find_job(mapping["source_umo"], target, mapping["scope"], mapping["revision"])
                    if job:
                        job.update(await storage.progress(job["job_key"]))
                        jobs.append(job)
            return json_response({"mappings": mappings, "jobs": jobs, "version": PLUGIN_VERSION})
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:
            logger.exception("[跨平台共享] 管理页读取失败")
            return error_response(f"读取失败：{exc}", status_code=500)

    async def page_settings_save(self):
        async with self._config_lock:
            return await self._page_settings_save()

    async def _save_mappings(self, normalized):
        serialized = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        old = self._cfg("clone_mappings_json", "[]")
        self.config["clone_mappings_json"] = serialized
        try:
            save = getattr(self.config, "save_config_async", None) or getattr(self.config, "save_config", None)
            if not callable(save):
                raise RuntimeError("AstrBot 配置对象不支持保存")
            result = save()
            if inspect.isawaitable(result):
                await result
        except Exception:
            self.config["clone_mappings_json"] = old
            raise

    async def _page_settings_save(self):
        payload = await request.json(default={})
        if not isinstance(payload, dict):
            return error_response("请求内容必须是 JSON 对象")
        if "mappings" not in payload and "clone_mappings_json" not in payload:
            return error_response("缺少 mappings 参数")
        value = payload.get("mappings", payload.get("clone_mappings_json"))
        try:
            mappings = parse_clone_mappings(value)
            normalized = [
                {
                    "source_umo": mapping.source_umo.raw,
                    "target_umos": [target.raw for target in mapping.target_umos],
                    "scope": mapping.scope,
                    "revision": mapping.revision,
                }
                for mapping in mappings
            ]
            await self._save_mappings(normalized)
            self._config_error = None
            scheduled = await self._schedule_clones(mappings)
            return json_response({"saved": True, "mappings": normalized, "scheduled": scheduled})
        except ValueError as exc:
            return error_response(str(exc), status_code=400)
        except Exception as exc:
            logger.exception("[跨平台共享] 管理页保存失败")
            return error_response(f"保存失败：{exc}", status_code=500)

    @staticmethod
    def _page_number(value):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000000:
            raise ValueError("页码必须是正整数")
        return value

    def _job_mapping(self, job):
        mapping = mapping_for_target(parse_clone_mappings(self._cfg("clone_mappings_json", "[]")), job["target_umo"])
        if mapping and (mapping.source_umo.raw, mapping.scope, mapping.revision) == (
                job["source_umo"], job["scope"], job["revision"]):
            return mapping
        return None

    async def page_history(self):
        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            page = self._page_number(payload.get("page", 1))
            storage = await self._ensure_storage()
            jobs = await storage.list_jobs(page, 20)
            for job in jobs:
                job["can_retry"] = self._job_mapping(job) is not None and job["status"] in {"partial", "failed", "pending"}
            return json_response({"jobs": jobs, "page": page, "page_size": 20, "total": await storage.count_jobs()})
        except (ValueError, TypeError) as exc:
            return error_response(str(exc))
        except Exception:
            logger.exception("[跨平台共享] 读取复制记录失败")
            return error_response("读取复制记录失败，请查看日志", status_code=500)

    async def page_detail(self):
        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            storage = await self._ensure_storage()
            job = await storage.get_job(str(payload.get("job_key", "")))
            if not job:
                return error_response("任务不存在", status_code=404)
            page = self._page_number(payload.get("page", 1))
            job.update(await storage.progress(job["job_key"]))
            detail = await storage.job_detail(job["job_key"], page)
            return json_response({"job": job, **detail, "page": page, "page_size": 50})
        except (ValueError, TypeError) as exc:
            return error_response(str(exc))
        except Exception:
            logger.exception("[跨平台共享] 读取任务详情失败")
            return error_response("读取任务详情失败，请查看日志", status_code=500)

    async def page_preview(self):
        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            storage = await self._ensure_storage()
            job = await storage.get_job(str(payload.get("job_key", "")))
            if not job:
                return error_response("任务不存在", status_code=404)
            record = await storage.get_conversation_map(job["job_key"], str(payload.get("source_cid", "")))
            if not record or record["status"] != "complete":
                return error_response("该对话尚未复制完成", status_code=404)
            conversation = await get_conversation(self.context.conversation_manager,
                                                   job["target_umo"], record["target_cid"])
            if conversation is None:
                return error_response("副本已在 AstrBot 中删除；任务记录仍保留", status_code=404)
            owner = getattr(conversation, "user_id", job["target_umo"])
            if owner != job["target_umo"]:
                return error_response("对话不属于该任务的目标 UMO", status_code=403)
            messages = conversation_history(conversation)
            page = self._page_number(payload.get("page", 1))
            start = (page - 1) * 20
            preview = []
            for index, message in enumerate(messages[start:start + 20], start):
                role = message.get("role", "unknown") if isinstance(message, dict) else "unknown"
                # Read only; render as escaped text. Media data is never sent to the page.
                text = self._preview_text(message)
                preview.append({"index": index + 1, "role": role, "text": text[:4000], "truncated": len(text) > 4000})
            return json_response({"messages": preview, "total": len(messages), "page": page, "page_size": 20,
                                  "title": getattr(conversation, "title", ""), "target_cid": record["target_cid"]})
        except (ValueError, TypeError) as exc:
            return error_response(str(exc))
        except Exception:
            logger.exception("[跨平台共享] 读取历史预览失败")
            return error_response("读取历史预览失败，请查看日志", status_code=500)

    @staticmethod
    def _preview_text(message):
        if not isinstance(message, dict):
            return str(message)
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(str(part.get("text", "")) if part.get("type") in {"text", "plain"}
                             else f"[{part.get('type', '消息组件')}]"
                             for part in content if isinstance(part, dict))
        else:
            text = ""
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            text += "\n[工具调用] " + ", ".join(str(call.get("function", {}).get("name", "未知工具"))
                                               for call in calls if isinstance(call, dict))
        return text or "[空消息或结构化内容]"

    async def page_retry(self):
        async with self._config_lock:
            try:
                payload = await request.json(default={})
                if not isinstance(payload, dict):
                    raise ValueError("请求内容必须是 JSON 对象")
                storage = await self._ensure_storage()
                job = await storage.get_job(str(payload.get("job_key", "")))
                if not job:
                    return error_response("任务不存在", status_code=404)
                mapping = self._job_mapping(job)
                if not mapping:
                    return error_response("该记录已不属于当前映射，请保存新的映射", status_code=409)
                scheduled = await self._schedule_clones([mapping], only_target=job["target_umo"])
                return json_response({"scheduled": scheduled})
            except (ValueError, TypeError) as exc:
                return error_response(str(exc))
            except Exception:
                logger.exception("[跨平台共享] 重试失败")
                return error_response("重试失败，请查看日志", status_code=500)

    async def page_recopy(self):
        async with self._config_lock:
            try:
                payload = await request.json(default={})
                if not isinstance(payload, dict):
                    raise ValueError("请求内容必须是 JSON 对象")
                normalized = self._mapping_dicts()
                selected = next((m for m in normalized if m["source_umo"] == payload.get("source_umo")), None)
                if not selected or selected["revision"] != payload.get("revision"):
                    return error_response("映射已变更，请刷新后重试", status_code=409)
                storage = await self._ensure_storage()
                selected["revision"] = max(selected["revision"], await storage.max_revision(selected["source_umo"])) + 1
                mappings = parse_clone_mappings(normalized)
                await self._save_mappings(normalized)
                mapping = next(m for m in mappings if m.source_umo.raw == selected["source_umo"])
                scheduled = await self._schedule_clones([mapping])
                return json_response({"mappings": normalized, "revision": selected["revision"], "scheduled": scheduled})
            except (ValueError, TypeError) as exc:
                return error_response(str(exc))
            except Exception:
                logger.exception("[跨平台共享] 创建新批次失败")
                return error_response("创建新批次失败，请查看日志", status_code=500)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any) -> None:
        target_umo = self._event_umo(event)
        if not target_umo:
            return
        try:
            mappings = parse_clone_mappings(self._cfg("clone_mappings_json", "[]"))
            mapping = mapping_for_target(mappings, target_umo)
            if mapping is None:
                return
            storage = await self._ensure_storage()
            target = next(target for target in mapping.target_umos if target.raw == target_umo)
            async with self._target_lock(target_umo):
                result = await ConversationCloner(self.context, storage).ensure_clone(mapping, target)
            # The target should immediately use the independent copy, including
            # when the copy was started by the dashboard save request.
            if result.conversation is not None and (
                    result.status != "already_complete" or
                    conversation_id(getattr(req, "conversation", None)) != result.target_conversation_id):
                req.conversation = result.conversation
                if hasattr(req, "contexts"):
                    req.contexts = conversation_history(result.conversation)
        except ValueError as exc:
            message = str(exc)
            if message != self._config_error:
                self._config_error = message
                logger.error("[跨平台共享] 配置无效：%s", message)
        except CloneError as exc:
            logger.error("[跨平台共享] UMO 复制失败：%s", exc)
        except Exception:
            logger.exception("[跨平台共享] 处理 LLM 请求时发生未预期错误")
