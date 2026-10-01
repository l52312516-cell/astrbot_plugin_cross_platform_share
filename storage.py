from __future__ import annotations

import os
import time
import asyncio
from typing import Any

try:
    import aiosqlite
except ImportError:  # pragma: no cover - dependency is declared by the plugin
    aiosqlite = None  # type: ignore[assignment]


class PluginStorage:
    """Plugin-owned metadata store; AstrBot conversation rows stay untouched."""

    def __init__(self, directory: str):
        self.directory = directory
        self.path = os.path.join(directory, "cross_platform_share.sqlite3")
        self._conn = None
        self._job_lock = asyncio.Lock()

    async def open(self) -> None:
        if aiosqlite is None:
            raise RuntimeError("缺少 aiosqlite，请安装插件 requirements.txt")
        os.makedirs(self.directory, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS clone_jobs (
                job_key TEXT PRIMARY KEY,
                mapping_key TEXT NOT NULL,
                source_umo TEXT NOT NULL,
                target_umo TEXT NOT NULL,
                scope TEXT NOT NULL,
                revision INTEGER NOT NULL,
                status TEXT NOT NULL,
                source_current_cid TEXT,
                target_conversation_id TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS conversation_maps (
                job_key TEXT NOT NULL,
                source_cid TEXT NOT NULL,
                target_cid TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (job_key, source_cid)
            );
            CREATE TABLE IF NOT EXISTS platform_history_maps (
                job_key TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (job_key, fingerprint)
            );
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_key TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT,
                message TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_key, id);
            CREATE INDEX IF NOT EXISTS idx_clone_identity
                ON clone_jobs(source_umo, target_umo, scope, revision);
            """
        )
        # Additive migration: preserve metadata and completed copies from 2.2.x.
        for table, columns in {
            "clone_jobs": {
                "phase": "TEXT", "attempt_count": "INTEGER NOT NULL DEFAULT 0",
                "started_at": "REAL", "finished_at": "REAL",
                "conversation_total": "INTEGER", "platform_total": "INTEGER",
            },
            "conversation_maps": {
                "status": "TEXT NOT NULL DEFAULT 'complete'", "message_count": "INTEGER",
            },
        }.items():
            async with self._conn.execute(f"PRAGMA table_info({table})") as cur:
                existing = {row[1] for row in await cur.fetchall()}
            for column, definition in columns.items():
                if column not in existing:
                    await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require(self):
        if self._conn is None:
            raise RuntimeError("插件存储尚未初始化")
        return self._conn

    async def get_or_create_job(
        self,
        job_key: str,
        mapping_key: str,
        source_umo: str,
        target_umo: str,
        scope: str,
        revision: int,
    ) -> dict[str, Any]:
        async with self._job_lock:
            # Old job hashes depended on all B targets. Reuse the same A/B/revision
            # when upgrading or adding another B instead of copying existing Bs again.
            existing = await self.find_job(source_umo, target_umo, scope, revision)
            if existing:
                return existing
            conn = self._require()
            now = time.time()
            await conn.execute(
                """INSERT INTO clone_jobs
                  (job_key, mapping_key, source_umo, target_umo, scope, revision,
                   status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (job_key, mapping_key, source_umo, target_umo, scope, revision, now, now),
            )
            await conn.commit()
            await self.add_event(job_key, "pending", None, "任务已创建")
            return await self.get_job(job_key)

    async def find_job(self, source: str, target: str, scope: str, revision: int):
        async with self._require().execute(
            """SELECT * FROM clone_jobs WHERE source_umo=? AND target_umo=?
               AND scope=? AND revision=?
               ORDER BY (status='complete') DESC, created_at DESC LIMIT 1""",
            (source, target, scope, revision),
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def add_event(self, job_key, status, phase, message=None):
        conn = self._require()
        await conn.execute(
            "INSERT INTO job_events(job_key,status,phase,message,created_at) VALUES (?,?,?,?,?)",
            (job_key, status, phase, message, time.time()),
        )
        await conn.commit()

    async def begin_attempt(self, job_key):
        conn = self._require()
        await conn.execute(
            "UPDATE clone_jobs SET attempt_count=attempt_count+1 WHERE job_key=?", (job_key,)
        )
        await self.mark_job(job_key, "running", phase="checking", error=None,
                            started_at=time.time(), finished_at=None)

    async def mark_job(self, job_key: str, status: str, **fields: Any) -> None:
        allowed = {
            "source_current_cid",
            "target_conversation_id",
            "error",
            "completed_at",
            "phase", "started_at", "finished_at", "conversation_total", "platform_total",
        }
        updates: dict[str, Any] = {key: value for key, value in fields.items() if key in allowed}
        updates["status"] = status
        updates["updated_at"] = time.time()
        if status == "complete":
            updates.setdefault("completed_at", time.time())
        if status in {"complete", "failed", "partial"}:
            updates.setdefault("finished_at", time.time())
        previous = await self.get_job(job_key)
        columns = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [job_key]
        conn = self._require()
        await conn.execute(f"UPDATE clone_jobs SET {columns} WHERE job_key = ?", values)
        await conn.commit()
        if previous and (previous["status"] != status or
                         previous.get("phase") != updates.get("phase", previous.get("phase"))):
            await self.add_event(job_key, status, updates.get("phase", previous.get("phase")),
                                 updates.get("error"))

    async def set_source_current(self, job_key: str, source_cid: str | None) -> None:
        await self.mark_job(job_key, "running", source_current_cid=source_cid)

    async def get_job(self, job_key: str) -> dict[str, Any] | None:
        conn = self._require()
        cur = await conn.execute("SELECT * FROM clone_jobs WHERE job_key = ?", (job_key,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_jobs(self, page: int = 1, page_size: int = 20) -> list[dict[str, Any]]:
        conn = self._require()
        cur = await conn.execute(
            "SELECT * FROM clone_jobs ORDER BY created_at DESC, job_key LIMIT ? OFFSET ?",
            (page_size, (page - 1) * page_size),
        )
        jobs = [dict(row) for row in await cur.fetchall()]
        for job in jobs:
            job.update(await self.progress(job["job_key"]))
        return jobs

    async def count_jobs(self) -> int:
        async with self._require().execute("SELECT COUNT(*) FROM clone_jobs") as cur:
            return (await cur.fetchone())[0]

    async def max_revision(self, source):
        async with self._require().execute(
            "SELECT COALESCE(MAX(revision),0) FROM clone_jobs WHERE source_umo=?", (source,)
        ) as cur:
            return (await cur.fetchone())[0]

    async def progress(self, job_key):
        async with self._require().execute(
            """SELECT
               (SELECT COUNT(*) FROM conversation_maps WHERE job_key=?) AS allocated_conversations,
               (SELECT COUNT(*) FROM conversation_maps WHERE job_key=? AND status='complete') AS conversations_done,
               (SELECT SUM(message_count) FROM conversation_maps WHERE job_key=? AND status='complete') AS messages_done,
               (SELECT COUNT(*) FROM platform_history_maps WHERE job_key=?) AS platform_done""",
            (job_key,) * 4,
        ) as cur:
            return dict(await cur.fetchone())

    async def job_detail(self, job_key, page=1, page_size=50):
        conn = self._require()
        async with conn.execute(
            "SELECT * FROM conversation_maps WHERE job_key=? ORDER BY created_at,source_cid LIMIT ? OFFSET ?",
            (job_key, page_size, (page - 1) * page_size),
        ) as cur:
            conversations = [dict(row) for row in await cur.fetchall()]
        async with conn.execute(
            "SELECT * FROM job_events WHERE job_key=? ORDER BY id DESC LIMIT 50", (job_key,)
        ) as cur:
            events = [dict(row) for row in await cur.fetchall()]
        return {"conversations": conversations, "events": events}

    async def save_conversation_map(self, job_key: str, source_cid: str, target_cid: str, title: str,
                                    status="complete", message_count=None) -> None:
        conn = self._require()
        await conn.execute(
            """
            INSERT INTO conversation_maps
              (job_key, source_cid, target_cid, title, created_at, status, message_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_key, source_cid) DO UPDATE SET
              target_cid=excluded.target_cid, title=excluded.title,
              status=excluded.status, message_count=excluded.message_count
            """,
            (job_key, source_cid, target_cid, title, time.time(), status, message_count),
        )
        await conn.commit()

    async def get_conversation_map(self, job_key: str, source_cid: str) -> dict[str, Any] | None:
        conn = self._require()
        cur = await conn.execute(
            "SELECT * FROM conversation_maps WHERE job_key = ? AND source_cid = ?",
            (job_key, source_cid),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def has_platform_record(self, job_key: str, fingerprint: str) -> bool:
        conn = self._require()
        cur = await conn.execute(
            "SELECT 1 FROM platform_history_maps WHERE job_key = ? AND fingerprint = ?",
            (job_key, fingerprint),
        )
        return await cur.fetchone() is not None

    async def save_platform_record(self, job_key: str, fingerprint: str) -> None:
        conn = self._require()
        await conn.execute(
            "INSERT OR IGNORE INTO platform_history_maps (job_key, fingerprint, created_at) VALUES (?, ?, ?)",
            (job_key, fingerprint, time.time()),
        )
        await conn.commit()
