#!/usr/bin/env python3
"""
诊断脚本：检查 conversation_id 映射是否正常工作。

运行方式：
    python test_conversation_mapping.py
"""

import asyncio
import uuid
import os
from pathlib import Path
from db.store import ConversationStore
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")


async def test_conversation_mapping():
    """测试 conversation_id 映射功能。"""
    
    # 从 .env 获取数据库 URL
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://intent_user:intent_password@localhost:5432/intent_recognition"
    )
    
    print(f"[Test] Database URL: {DATABASE_URL}")
    print()
    
    try:
        # 初始化连接池
        pool = AsyncConnectionPool(conninfo=DATABASE_URL, max_size=5, open=False)
        await pool.open()
        print("[✓] 数据库连接成功")
        
        # 初始化存储
        store = ConversationStore(DATABASE_URL)
        store.pool = pool
        await store.init_db()
        print("[✓] 数据库表初始化成功")
        print()
        
        # 测试 1: 创建新的 session_id 映射
        print("=" * 60)
        print("测试 1: 创建新的 session_id -> conversation_id 映射")
        print("=" * 60)
        
        test_session_id_1 = str(uuid.uuid4())
        print(f"Test session_id: {test_session_id_1}")
        
        conv_id_1 = await store.get_or_create_conversation_id(test_session_id_1)
        print(f"Generated conversation_id: {conv_id_1}")
        
        if conv_id_1:
            print("[✓] 映射创建成功")
        else:
            print("[✗] 映射创建失败")
        print()
        
        # 测试 2: 复用现有的 session_id 映射
        print("=" * 60)
        print("测试 2: 复用现有的 session_id -> conversation_id 映射")
        print("=" * 60)
        
        conv_id_1_again = await store.get_or_create_conversation_id(test_session_id_1)
        print(f"Queried conversation_id: {conv_id_1_again}")
        
        if conv_id_1 == conv_id_1_again:
            print("[✓] 映射复用成功，ID 保持一致")
        else:
            print(f"[✗] 映射复用失败！")
            print(f"   第一次: {conv_id_1}")
            print(f"   第二次: {conv_id_1_again}")
        print()
        
        # 测试 3: 多个 session_id 对应不同的 conversation_id
        print("=" * 60)
        print("测试 3: 不同的 session_id 应对应不同的 conversation_id")
        print("=" * 60)
        
        test_session_id_2 = str(uuid.uuid4())
        conv_id_2 = await store.get_or_create_conversation_id(test_session_id_2)
        
        print(f"Session 1: {test_session_id_1}")
        print(f"  → Conversation: {conv_id_1}")
        print()
        print(f"Session 2: {test_session_id_2}")
        print(f"  → Conversation: {conv_id_2}")
        
        if conv_id_1 != conv_id_2:
            print("[✓] 不同 session 的 conversation_id 不同")
        else:
            print("[✗] 不同 session 的 conversation_id 相同（错误）")
        print()
        
        # 测试 4: 直接查询数据库
        print("=" * 60)
        print("测试 4: 查询数据库中的映射记录")
        print("=" * 60)
        
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM session_conversation_mapping;")
                count = await cur.fetchone()
                print(f"数据库中的映射记录数: {count[0]}")
                
                await cur.execute("""
                    SELECT session_id, conversation_id, created_at 
                    FROM session_conversation_mapping 
                    ORDER BY created_at DESC 
                    LIMIT 5;
                """)
                rows = await cur.fetchall()
                
                if rows:
                    print("\n最近 5 条记录:")
                    for i, (sid, cid, created_at) in enumerate(rows, 1):
                        is_test = " (test)" if sid in [test_session_id_1, test_session_id_2] else ""
                        print(f"  {i}. {sid[:8]}...{sid[-4:]} → {cid[:8]}...{cid[-4:]}{is_test}")
                else:
                    print("[!] 数据库中没有映射记录")
        
        print()
        print("=" * 60)
        print("✓ 所有测试完成")
        print("=" * 60)
        
        await pool.close()
        
    except Exception as e:
        print(f"[✗] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_conversation_mapping())
