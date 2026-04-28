import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg
from psycopg_pool import AsyncConnectionPool
from engine.logging_config import get_logger

logger = get_logger(__name__)

class ConversationStore:
    """
    会话持久化存储类，负责管理会话元数据（标题、角色、统计信息等）。
    具体的状态 Checkpoint 由 LangGraph PostgresSaver 处理，共用同一个数据库。
    """
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = AsyncConnectionPool(conninfo=db_url, open=False)

    async def open(self):
        """开启连接池。"""
        await self.pool.open()

    async def close(self):
        """关闭连接池。"""
        await self.pool.close()

    async def init_db(self):
        """初始化元数据表。"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                # 创建会话元数据表
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        session_id   TEXT PRIMARY KEY,
                        title        TEXT NOT NULL,
                        role         TEXT DEFAULT '',
                        created_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        message_count INTEGER DEFAULT 0
                    );
                """)
                # 确保索引存在
                await cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_updated_at ON conversation_metadata (updated_at DESC);")
                await conn.commit()
        logger.info("Database conversation_metadata table initialized.")

    async def upsert_conversation(self, session_id: str, title: str, role: str, message_count: int):
        """新建或更新会话元数据。"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO conversation_metadata (session_id, title, role, message_count, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        role = EXCLUDED.role,
                        message_count = EXCLUDED.message_count,
                        updated_at = CURRENT_TIMESTAMP;
                """, (session_id, title, role, message_count))
                await conn.commit()

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """按更新时间倒序列出所有会话。"""
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute("""
                    SELECT session_id, title, role, created_at, updated_at, message_count
                    FROM conversation_metadata
                    ORDER BY updated_at DESC;
                """)
                rows = await cur.fetchall()
                # 序列化 datetime
                for row in rows:
                    if row["created_at"]:
                        row["created_at"] = row["created_at"].isoformat()
                    if row["updated_at"]:
                        row["updated_at"] = row["updated_at"].isoformat()
                return rows

    async def delete_metadata(self, session_id: str):
        """删除会话元数据记录。"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM conversation_metadata WHERE session_id = %s;", (session_id,))
                await conn.commit()
        logger.info(f"Conversation metadata {session_id} deleted.")

