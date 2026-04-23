"""
Agent Card加载器 - 自动发现和管理Agent Cards
"""

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from engine.agent_card import AgentCard


@dataclass
class LoadStatistics:
    """加载统计信息"""
    total: int = 0
    success: int = 0
    failed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class AgentCardRegistry:
    """Agent Card注册表"""
    
    def __init__(self):
        self._cards: Dict[str, AgentCard] = {}
        self._intent_to_agents: Dict[str, List[str]] = {}
        self._keyword_to_agents: Dict[str, List[str]] = {}
        self._subagents: Dict[str, object] = {}
    
    def register(self, card: AgentCard) -> None:
        """注册一个Agent Card"""
        agent_id = card.metadata.agent_id
        self._cards[agent_id] = card
        
        # 索引意图模式
        for pattern in card.capabilities.intent_patterns:
            if pattern not in self._intent_to_agents:
                self._intent_to_agents[pattern] = []
            self._intent_to_agents[pattern].append(agent_id)
        
        # 索引关键词
        for keyword in card.capabilities.keywords:
            if keyword not in self._keyword_to_agents:
                self._keyword_to_agents[keyword] = []
            self._keyword_to_agents[keyword].append(agent_id)
    
    def get_by_id(self, agent_id: str) -> Optional[AgentCard]:
        """按ID获取Agent Card"""
        return self._cards.get(agent_id)
    
    def get_by_intent(self, intent: str) -> List[AgentCard]:
        """按意图获取Agent Cards（降序排列，按优先级）"""
        agent_ids = self._intent_to_agents.get(intent, [])
        cards = [self._cards[aid] for aid in agent_ids if aid in self._cards]
        # 按优先级排序（从高到低）
        cards.sort(key=lambda c: c.capabilities.priority, reverse=True)
        return cards
    
    def get_by_keyword(self, keyword: str) -> List[AgentCard]:
        """按关键词获取Agent Cards"""
        agent_ids = self._keyword_to_agents.get(keyword, [])
        cards = [self._cards[aid] for aid in agent_ids if aid in self._cards]
        cards.sort(key=lambda c: c.capabilities.priority, reverse=True)
        return cards
    
    def list_all(self) -> List[AgentCard]:
        """列出所有Agent Cards"""
        return list(self._cards.values())
    
    def register_subagent(self, agent_id: str, subagent: object) -> None:
        """注册执行器"""
        self._subagents[agent_id] = subagent
    
    def get_subagent(self, agent_id: str) -> Optional[object]:
        """获取执行器"""
        return self._subagents.get(agent_id)
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "total_agents": len(self._cards),
            "total_intents": len(self._intent_to_agents),
            "total_keywords": len(self._keyword_to_agents),
            "total_subagents": len(self._subagents),
        }


class AgentCardLoader:
    """Agent Card加载器"""
    
    def __init__(self, skills_root: str = None):
        """
        初始化加载器
        
        Args:
            skills_root: skills目录的根路径，默认为项目中的agents
        """
        if skills_root is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent
            skills_root = project_root / "agents"
        else:
            skills_root = Path(skills_root)
        
        self.skills_root = skills_root
        self.registry = AgentCardRegistry()
        self.stats = LoadStatistics()
    
    def discover_agent_cards(self) -> List[Tuple[str, Path]]:
        """
        自动发现所有的agent_card.yaml文件
        
        Returns:
            [(agent_id, yaml_path), ...]
        """
        cards = []
        
        if not self.skills_root.exists():
            return cards
        
        # 遍历agents下的每个skill目录
        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            
            # 查找agent_card.yaml
            card_path = skill_dir / "agent_card.yaml"
            if card_path.exists():
                agent_id = skill_dir.name
                cards.append((agent_id, str(card_path)))
        
        return cards
    
    def load_card(self, yaml_path: str) -> Optional[AgentCard]:
        """加载单个Agent Card"""
        try:
            card = AgentCard.from_yaml(yaml_path)
            return card
        except Exception as e:
            error_msg = f"Failed to load {yaml_path}: {str(e)}"
            self.stats.errors.append(error_msg)
            return None
    
    def load_all_cards(self) -> AgentCardRegistry:
        """
        加载所有发现的Agent Cards并注册
        
        Returns:
            AgentCardRegistry实例
        """
        self.stats = LoadStatistics()
        cards = self.discover_agent_cards()
        self.stats.total = len(cards)
        
        for agent_id, yaml_path in cards:
            card = self.load_card(yaml_path)
            if card:
                self.registry.register(card)
                self._load_subagent(card)
                self.stats.success += 1
            else:
                self.stats.failed += 1
        
        return self.registry
    
    def _load_subagent(self, card: AgentCard) -> bool:
        """
        动态加载subagent
        
        Args:
            card: AgentCard实例
        
        Returns:
            是否加载成功
        """
        if card.execution is None:
            return False
        
        try:
            # 动态导入模块
            module = importlib.import_module(card.execution.module)
            
            # 获取类
            subagent_class = getattr(module, card.execution.class_name)
            
            # 创建实例（不带参数）
            subagent = subagent_class()
            
            # 注册到registry
            self.registry.register_subagent(card.metadata.agent_id, subagent)
            
            return True
        except Exception as e:
            error_msg = f"Failed to load subagent for {card.metadata.agent_id}: {str(e)}"
            self.stats.errors.append(error_msg)
            return False
    
    def print_statistics(self) -> None:
        """打印加载统计"""
        print(f"""
========== Agent Card 加载统计 ==========
总计发现: {self.stats.total}
加载成功: {self.stats.success}
加载失败: {self.stats.failed}
Registry统计: {self.registry.get_statistics()}

错误列表:
{chr(10).join('  - ' + err for err in self.stats.errors) if self.stats.errors else '  无'}
============================================
""")


def print_available_agents(registry: AgentCardRegistry) -> None:
    """打印可用的Agent列表"""
    print("\n========== 可用的Agent Cards ==========")
    cards = registry.list_all()
    if not cards:
        print("没有发现任何Agent Cards")
        return
    
    for card in sorted(cards, key=lambda c: c.capabilities.priority, reverse=True):
        meta = card.metadata
        cap = card.capabilities
        print(f"""
  Agent ID: {meta.agent_id}
  名称: {meta.name}
  描述: {meta.description}
  版本: {meta.version}
  优先级: {cap.priority}
  技能: {', '.join(cap.skills)}
  意图模式: {', '.join(cap.intent_patterns)}
  关键词: {', '.join(cap.keywords[:3])}{'...' if len(cap.keywords) > 3 else ''}
""")
    print("=" * 40)


def query_agent_for_intent(registry: AgentCardRegistry, intent: str) -> None:
    """查询特定意图对应的Agent"""
    print(f"\n查询意图: {intent}")
    cards = registry.get_by_intent(intent)
    
    if not cards:
        print(f"  没有找到匹配的Agent")
        return
    
    print(f"  找到 {len(cards)} 个Agent:")
    for i, card in enumerate(cards, 1):
        print(f"    {i}. {card.metadata.name} (id: {card.metadata.agent_id}, priority: {card.capabilities.priority})")


def get_agent_capabilities_summary(registry: AgentCardRegistry) -> str:
    """获取Agent能力总结"""
    lines = ["========== Agent系统能力总结 =========="]
    
    stats = registry.get_statistics()
    lines.append(f"Agent总数: {stats['total_agents']}")
    lines.append(f"意图总数: {stats['total_intents']}")
    lines.append(f"关键词总数: {stats['total_keywords']}")
    
    lines.append("\n支持的意图模式:")
    all_cards = registry.list_all()
    all_intents = set()
    for card in all_cards:
        all_intents.update(card.capabilities.intent_patterns)
    
    for intent in sorted(all_intents):
        matching_agents = registry.get_by_intent(intent)
        agent_names = [c.metadata.name for c in matching_agents]
        lines.append(f"  - {intent}: {', '.join(agent_names)}")
    
    lines.append("\n支持的关键词 (前20个):")
    all_keywords = set()
    for card in all_cards:
        all_keywords.update(card.capabilities.keywords)
    
    for keyword in sorted(all_keywords)[:20]:
        matching_agents = registry.get_by_keyword(keyword)
        agent_names = [c.metadata.name for c in matching_agents]
        lines.append(f"  - {keyword}: {', '.join(agent_names)}")
    
    lines.append("=" * 40)
    return "\n".join(lines)


# 便利函数：一键初始化Agent Card系统
def init_agent_card_system(skills_root: str = None) -> AgentCardRegistry:
    """
    初始化Agent Card系统
    
    Args:
        skills_root: skills目录的根路径
    
    Returns:
        AgentCardRegistry实例
    """
    loader = AgentCardLoader(skills_root)
    registry = loader.load_all_cards()
    return registry


# 调试函数：打印完整的系统信息
def print_system_info(registry: AgentCardRegistry) -> None:
    """打印Agent Card系统的完整信息"""
    print_available_agents(registry)
    print(get_agent_capabilities_summary(registry))
