"""
Agent Cards 管理 — 加载、缓存和管理 Agent Card 元数据。

提供统一的 Agent Card 加载接口，支持缓存机制，避免重复的远程发现。
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from functools import lru_cache

# 添加项目根目录和 engine 到 Python 路径
# 这样可以支持 `from engine...` 和 `from agent_card...` 两种导入方式
_engine_path = Path(__file__).parent
_project_root = Path(__file__).parent.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_engine_path) not in sys.path:
    sys.path.insert(0, str(_engine_path))

# 动态导入，支持不同的项目结构
try:
    from agent_card import AgentCard
    from a2a import discover_a2a_agent_cards
except ImportError as e:
    print(f"[AgentCardManager] 导入失败: {e}")
    print(f"  _engine_path: {_engine_path}")
    print(f"  _project_root: {_project_root}")
    print(f"  sys.path: {sys.path[:3]}")
    AgentCard = None
    discover_a2a_agent_cards = None


class AgentCardManager:
    """
    管理 Agent Card 的加载、缓存和发现。
    
    支持功能:
    - 从远程 A2A 端点发现 Agent Cards
    - 本地内存缓存（可选）
    - 强制刷新缓存
    """

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 3600):
        """
        初始化 Agent Card 管理器。
        
        Args:
            enable_cache: 是否启用缓存
            cache_ttl: 缓存过期时间（秒）
        """
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self._cards_cache: Optional[List[AgentCard]] = None
        self._cache_timestamp: Optional[float] = None

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效。"""
        if not self.enable_cache or self._cards_cache is None:
            return False
        if self._cache_timestamp is None:
            return False
        import time
        elapsed = time.time() - self._cache_timestamp
        return elapsed < self.cache_ttl

    async def load_cards_async(self, force_refresh: bool = False) -> List[AgentCard]:
        """
        异步加载 Agent Cards。
        
        Args:
            force_refresh: 强制刷新缓存
        
        Returns:
            Agent Card 列表
        """
        if not force_refresh and self._is_cache_valid():
            return self._cards_cache or []

        if discover_a2a_agent_cards is None:
            print("[AgentCardManager] Warning: discover_a2a_agent_cards 不可用")
            return []

        try:
            # 在线程池中执行同步发现函数，避免阻塞事件循环
            cards = await asyncio.to_thread(discover_a2a_agent_cards)
            
            if self.enable_cache:
                import time
                self._cards_cache = cards
                self._cache_timestamp = time.time()
            
            return cards
        except Exception as e:
            print(f"[AgentCardManager] 加载 Agent Cards 失败: {e}")
            return self._cards_cache or []

    def load_cards(self, force_refresh: bool = False) -> List[AgentCard]:
        """
        同步加载 Agent Cards。
        
        Args:
            force_refresh: 强制刷新缓存
        
        Returns:
            Agent Card 列表
        """
        if not force_refresh and self._is_cache_valid():
            return self._cards_cache or []

        if discover_a2a_agent_cards is None:
            print("[AgentCardManager] Warning: discover_a2a_agent_cards 不可用")
            return []

        try:
            cards = discover_a2a_agent_cards()
            
            if self.enable_cache:
                import time
                self._cards_cache = cards
                self._cache_timestamp = time.time()
            
            return cards
        except Exception as e:
            print(f"[AgentCardManager] 加载 Agent Cards 失败: {e}")
            return self._cards_cache or []

    def refresh_cache(self) -> List[AgentCard]:
        """
        强制刷新缓存并返回最新的 Agent Cards。
        
        Returns:
            Agent Card 列表
        """
        return self.load_cards(force_refresh=True)

    def get_card_by_id(self, agent_id: str, cards: Optional[List[AgentCard]] = None) -> Optional[AgentCard]:
        """
        根据 agent_id 获取 Agent Card。
        
        Args:
            agent_id: Agent 的标识符
            cards: 可选的 Agent Card 列表（如不提供，则使用当前缓存）
        
        Returns:
            Agent Card 或 None
        """
        if cards is None:
            cards = self._cards_cache or []
        
        for card in cards:
            if hasattr(card, 'metadata') and hasattr(card.metadata, 'agent_id'):
                if card.metadata.agent_id == agent_id:
                    return card
        return None

    def index_cards_by_id(self, cards: List[AgentCard]) -> Dict[str, AgentCard]:
        """
        按 agent_id 索引 Agent Cards。
        
        Args:
            cards: Agent Card 列表
        
        Returns:
            {agent_id -> AgentCard} 映射
        """
        result = {}
        for card in cards:
            if hasattr(card, 'metadata') and hasattr(card.metadata, 'agent_id'):
                result[card.metadata.agent_id] = card
        return result


# 全局单例管理器
_global_manager: Optional[AgentCardManager] = None


def get_agent_card_manager(enable_cache: bool = True) -> AgentCardManager:
    """
    获取全局 Agent Card 管理器实例。
    
    Args:
        enable_cache: 是否启用缓存（首次创建时有效）
    
    Returns:
        AgentCardManager 实例
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = AgentCardManager(enable_cache=enable_cache)
    return _global_manager


# 便捷函数
async def load_cards_async(force_refresh: bool = False) -> List[AgentCard]:
    """异步加载 Agent Cards（使用全局管理器）。"""
    manager = get_agent_card_manager()
    return await manager.load_cards_async(force_refresh=force_refresh)


def load_cards(force_refresh: bool = False) -> List[AgentCard]:
    """同步加载 Agent Cards（使用全局管理器）。"""
    manager = get_agent_card_manager()
    return manager.load_cards(force_refresh=force_refresh)


def refresh_cards() -> List[AgentCard]:
    """强制刷新 Agent Cards 缓存（使用全局管理器）。"""
    manager = get_agent_card_manager()
    return manager.refresh_cache()
