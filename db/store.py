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
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # 创建会话元数据表
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS conversation_metadata (
                            session_id   TEXT PRIMARY KEY,
                            title        TEXT NOT NULL,
                            role         TEXT DEFAULT '',
                            last_reply   TEXT DEFAULT '',
                            created_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            message_count INTEGER DEFAULT 0
                        );
                    """)
                    logger.debug("Created conversation_metadata table")
                    
                    # 确保索引存在
                    await cur.execute("CREATE INDEX IF NOT EXISTS idx_cm_updated_at ON conversation_metadata (updated_at DESC);")
                    
                    # 迁移：为现有表添加 last_reply 列（如果不存在）
                    await cur.execute("""
                        ALTER TABLE conversation_metadata
                        ADD COLUMN IF NOT EXISTS last_reply TEXT DEFAULT '';
                    """)
                    
                    # 创建暂停会话上下文表
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS paused_context (
                            id              SERIAL PRIMARY KEY,
                            session_id      TEXT NOT NULL,
                            paused_state    JSONB NOT NULL,
                            created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            archived_at     TIMESTAMP WITH TIME ZONE,
                            new_session_id  TEXT
                        );
                    """)
                    logger.debug("Created paused_context table")
                    await cur.execute("CREATE INDEX IF NOT EXISTS idx_pc_session_id ON paused_context (session_id);")
                    await cur.execute("CREATE INDEX IF NOT EXISTS idx_pc_created_at ON paused_context (created_at DESC);")
                    
                    # 创建 session_id -> conversation_id 映射表（用于 Dify 多轮对话）
                    await cur.execute("""
                        CREATE TABLE IF NOT EXISTS session_conversation_mapping (
                            session_id      TEXT PRIMARY KEY,
                            conversation_id TEXT NOT NULL UNIQUE,
                            source          TEXT NOT NULL DEFAULT 'dify',
                            created_at      TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    logger.info("Created session_conversation_mapping table")
                    await cur.execute("""
                        ALTER TABLE session_conversation_mapping
                        ADD COLUMN IF NOT EXISTS source TEXT;
                    """)
                    await cur.execute("""
                        UPDATE session_conversation_mapping
                        SET source = 'legacy'
                        WHERE source IS NULL;
                    """)
                    await cur.execute("""
                        ALTER TABLE session_conversation_mapping
                        ALTER COLUMN source SET DEFAULT 'dify';
                    """)
                    await cur.execute("""
                        ALTER TABLE session_conversation_mapping
                        ALTER COLUMN source SET NOT NULL;
                    """)
                    await cur.execute("CREATE INDEX IF NOT EXISTS idx_scm_conversation_id ON session_conversation_mapping (conversation_id);")
                    
                    await conn.commit()
            logger.info("Database tables initialization complete: conversation_metadata, paused_context, session_conversation_mapping")
        except Exception as e:
            logger.error(f"Failed to initialize database tables: {e}", exc_info=True)
            raise

    async def upsert_conversation(self, session_id: str, title: str, role: str, message_count: int, last_reply: str = ""):
        """新建或更新会话元数据。"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO conversation_metadata (session_id, title, role, message_count, last_reply, updated_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        role = EXCLUDED.role,
                        message_count = EXCLUDED.message_count,
                        last_reply = CASE
                            WHEN EXCLUDED.last_reply <> '' THEN EXCLUDED.last_reply
                            ELSE conversation_metadata.last_reply
                        END,
                        updated_at = CURRENT_TIMESTAMP;
                """, (session_id, title, role, message_count, last_reply))
                await conn.commit()

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """按更新时间倒序列出所有会话。"""
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute("""
                    SELECT session_id, title, role, created_at, updated_at, message_count, last_reply
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

    # ─── 暂停上下文管理 ───
    async def save_paused_context(self, session_id: str, paused_state: Dict[str, Any]) -> int:
        """
        保存暂停的会话上下文。
        
        Args:
            session_id: 原会话ID
            paused_state: 暂停时的状态 (包含思维链、当前答案等)
            
        Returns:
            保存的记录ID
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO paused_context (session_id, paused_state)
                    VALUES (%s, %s)
                    RETURNING id;
                """, (session_id, json.dumps(paused_state)))
                result = await cur.fetchone()
                await conn.commit()
                context_id = result[0] if result else None
        logger.info(f"Paused context saved: session={session_id}, id={context_id}")
        return context_id

    async def get_paused_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取最新的暂停会话上下文（未被归档的）。"""
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                await cur.execute("""
                    SELECT paused_state FROM paused_context
                    WHERE session_id = %s AND archived_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT 1;
                """, (session_id,))
                result = await cur.fetchone()
                if result:
                    return json.loads(result["paused_state"])
        return None

    async def archive_paused_context(self, session_id: str, new_session_id: Optional[str] = None):
        """
        归档所有暂停上下文（标记为已使用）。
        
        Args:
            session_id: 原会话ID
            new_session_id: 新会话ID（如果开启了新会话）
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    UPDATE paused_context
                    SET archived_at = CURRENT_TIMESTAMP, new_session_id = %s
                    WHERE session_id = %s AND archived_at IS NULL;
                """, (new_session_id, session_id))
                await conn.commit()
        logger.info(f"Paused context archived: session={session_id}, new_session={new_session_id}")

    # ─── Conversation ID 映射管理 ───
    async def get_or_create_conversation_id(self, session_id: str) -> str:
        """
        获取或创建 session_id 对应的 Dify conversation_id。
        
        如果该 session_id 已有映射，返回现有的 conversation_id。
        否则生成新的 conversation_id 并保存映射。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            Dify conversation_id
        """
        import uuid
        
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # 先尝试查询现有映射
                    logger.debug(f"[get_or_create_conversation_id] Querying for session_id={session_id}")
                    await cur.execute(
                        "SELECT conversation_id FROM session_conversation_mapping WHERE session_id = %s;",
                        (session_id,)
                    )
                    result = await cur.fetchone()
                    
                    if result:
                        logger.info(f"[get_or_create_conversation_id] Found existing mapping: session={session_id}, conversation_id={result[0]}")
                        return result[0]
                    
                    # 生成新的 conversation_id
                    conversation_id = str(uuid.uuid4())
                    logger.debug(f"[get_or_create_conversation_id] Generating new conversation_id={conversation_id} for session={session_id}")
                    
                    try:
                        await cur.execute(
                            """
                            INSERT INTO session_conversation_mapping (session_id, conversation_id, source)
                            VALUES (%s, %s, 'local');
                            """,
                            (session_id, conversation_id)
                        )
                        await conn.commit()
                        logger.info(f"[get_or_create_conversation_id] Created mapping: session={session_id}, conversation_id={conversation_id}")
                        return conversation_id
                    except Exception as insert_err:
                        # 如果并发冲突，重新查询
                        logger.warning(f"[get_or_create_conversation_id] Insert failed ({type(insert_err).__name__}): {insert_err}, retrying with new connection...")
                        
                        # 创建新连接进行重试
                        async with self.pool.connection() as retry_conn:
                            async with retry_conn.cursor() as retry_cur:
                                await retry_cur.execute(
                                    "SELECT conversation_id FROM session_conversation_mapping WHERE session_id = %s;",
                                    (session_id,)
                                )
                                result = await retry_cur.fetchone()
                                if result:
                                    logger.info(f"[get_or_create_conversation_id] Found mapping on retry: session={session_id}, conversation_id={result[0]}")
                                    return result[0]
                        
                        # 重试失败，记录错误并返回空字符串而不是抛出异常
                        logger.error(f"[get_or_create_conversation_id] Failed to get or create mapping for session={session_id}: {insert_err}")
                        return ""
        except Exception as e:
            logger.error(f"[get_or_create_conversation_id] Unexpected error for session={session_id}: {e}", exc_info=True)
            return ""
    
    async def get_conversation_id(self, session_id: str) -> Optional[str]:
        """
        获取 session_id 对应的 conversation_id。
        
        Args:
            session_id: 会话 ID
            
        Returns:
            conversation_id 或 None（如果不存在）
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT conversation_id FROM session_conversation_mapping
                    WHERE session_id = %s AND source = 'dify';
                    """,
                    (session_id,)
                )
                result = await cur.fetchone()
                return result[0] if result else None

    async def set_conversation_id(self, session_id: str, conversation_id: str):
        """保存或更新 session_id 对应的真实 Dify conversation_id。"""
        if not session_id or not conversation_id:
            return

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO session_conversation_mapping (session_id, conversation_id, source)
                    VALUES (%s, %s, 'dify')
                    ON CONFLICT (session_id) DO UPDATE SET
                        conversation_id = EXCLUDED.conversation_id,
                        source = 'dify';
                    """,
                    (session_id, conversation_id),
                )
                await conn.commit()
        logger.info(
            f"[set_conversation_id] Saved mapping: session={session_id}, "
            f"conversation_id={conversation_id}"
        )
