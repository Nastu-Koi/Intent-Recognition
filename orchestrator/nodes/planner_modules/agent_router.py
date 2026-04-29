"""
Agent Router — 智能体路由和发现逻辑。

提供基于关键词和意图模式的智能体匹配和路由功能。
"""

import re
from typing import List, Tuple, Iterable

# 动态导入
try:
    import sys
    from pathlib import Path
    _loader_path = Path(__file__).parent.parent.parent.parent / "engine"
    if str(_loader_path) not in sys.path:
        sys.path.insert(0, str(_loader_path))
    from agent_card import AgentCard
except ImportError:
    AgentCard = None


# 弱关键词集合 — 这些关键词经常出现但信息量较少
WEAK_CONTEXT_KEYWORDS = {"入职", "离职"}


def find_matches(query: str, candidates: Iterable[str]) -> List[str]:
    """
    在查询中查找与候选词匹配的词汇。
    
    支持两种匹配模式：
    1. ASCII 词（如 "HR"）使用词边界匹配
    2. 中文词使用子串匹配
    
    Args:
        query: 用户查询文本
        candidates: 候选词列表（关键词或意图模式）
    
    Returns:
        匹配到的候选词列表
    """
    query_lower = query.lower()
    matches = []
    
    for item in candidates:
        item_lower = item.lower()
        
        # 对于 ASCII 词（如 "HR", "AI"），使用词边界匹配
        if item_lower.isascii() and item_lower.replace(" ", "").isalpha():
            if re.search(rf"\b{re.escape(item_lower)}\b", query_lower):
                matches.append(item)
        # 对于其他词（主要是中文），使用子串匹配
        elif item_lower in query_lower:
            matches.append(item)
    
    return matches


def route_query_multi(
    query: str,
    cards: List[AgentCard],
    min_matches: int = 1,
) -> List[Tuple[AgentCard, List[str]]]:
    """
    基于关键词和意图模式的多 Agent 路由。
    
    路由策略:
    1. 对每个 Agent Card 计算意图匹配和关键词匹配
    2. 过滤掉弱关键词（如"入职"）
    3. 按匹配数量和优先级排序
    4. 返回排序后的 [(Agent Card, 匹配词列表)]
    
    Args:
        query: 用户查询
        cards: 可用的 Agent Card 列表
        min_matches: 最少匹配数（默认1）
    
    Returns:
        [(AgentCard, matched_terms), ...] 按匹配质量排序
    """
    scored = []
    
    for card in cards:
        # 提取意图模式和关键词
        intent_patterns = card.capabilities.intent_patterns or []
        keywords = card.capabilities.keywords or []
        
        # 分别计算意图和关键词匹配
        intent_matches = find_matches(query, intent_patterns)
        keyword_matches = find_matches(query, keywords)
        
        # 过滤掉弱关键词
        strong_keyword_matches = [
            kw for kw in keyword_matches
            if kw.lower() not in WEAK_CONTEXT_KEYWORDS
        ]
        
        # 合并（去重）
        all_matches = []
        all_matches.extend(intent_matches)
        all_matches.extend(strong_keyword_matches)
        all_matches = list(dict.fromkeys(all_matches))  # 去重
        
        # 记录有匹配的 Agent
        if len(all_matches) >= min_matches:
            # (匹配数, 优先级, AgentCard, 匹配词)
            priority = card.capabilities.priority or 5
            scored.append((len(all_matches), priority, card, all_matches))
    
    # 按匹配数和优先级排序（降序）
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    
    # 返回 (AgentCard, 匹配词)
    return [(card, matches) for _, _, card, matches in scored]


def filter_agents_by_role(
    cards: List[AgentCard],
    accessible_agent_ids: List[str],
) -> List[AgentCard]:
    """
    根据 RBAC 角色过滤 Agent Cards。
    
    Args:
        cards: 所有可用的 Agent Card
        accessible_agent_ids: 该角色可访问的 agent_id 列表
    
    Returns:
        过滤后的 Agent Card 列表
    """
    return [
        card for card in cards
        if hasattr(card, 'metadata') and card.metadata.agent_id in accessible_agent_ids
    ]


def get_agent_metadata(card: 'AgentCard') -> dict:
    """
    提取 Agent Card 的关键元数据。
    
    Args:
        card: Agent Card 对象
    
    Returns:
        {agent_id, name, description, keywords, intent_patterns, priority}
    """
    return {
        "agent_id": card.metadata.agent_id if hasattr(card, 'metadata') else "unknown",
        "name": card.metadata.name if hasattr(card, 'metadata') else "Unknown",
        "description": card.metadata.description if hasattr(card, 'metadata') else "",
        "keywords": card.capabilities.keywords if hasattr(card, 'capabilities') else [],
        "intent_patterns": card.capabilities.intent_patterns if hasattr(card, 'capabilities') else [],
        "priority": card.capabilities.priority if hasattr(card, 'capabilities') else 5,
    }
