"""
Task Builder — 任务构建辅助函数。

支持将路由结果和智能体元数据转换为标准化的任务项（TaskItem）。
"""

from typing import List, Dict, Any, Optional, Tuple

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


def build_task_item(
    executor: str,
    executor_type: str,
    instruction: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建标准化的任务项。
    
    Args:
        executor: 执行器标识符（agent_id 或 worker 名称）
        executor_type: 执行器类型（'dify' 或 'a2a'）
        instruction: 具体执行指令
        metadata: 扩展元数据
    
    Returns:
        标准化的 TaskItem 字典
    """
    return {
        "executor": executor,
        "executor_type": executor_type,
        "instruction": instruction,
        "metadata": metadata or {},
    }


def build_a2a_task(
    executor: str,
    agent_name: str,
    instruction: str,
    matched_terms: Optional[List[str]] = None,
    original_query: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建 A2A 任务项。
    
    Args:
        executor: Remote Agent ID
        agent_name: Agent 名称
        instruction: 执行指令
        matched_terms: 匹配的关键词
        original_query: 原始用户查询
        metadata: 附加元数据
    
    Returns:
        A2A 任务字典
    """
    full_metadata = {
        "agent_name": agent_name,
        "matched_terms": matched_terms or [],
        "original_query": original_query or "",
    }
    if metadata:
        full_metadata.update(metadata)
    
    return build_task_item(
        executor=executor,
        executor_type="a2a",
        instruction=instruction,
        metadata=full_metadata,
    )


def build_tasks_from_routes(
    query: str,
    routes: List[Tuple['AgentCard', List[str]]],
    original_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    从路由结果构建任务列表。
    
    Args:
        query: 当前查询（可能是处理过的）
        routes: [(AgentCard, matched_terms), ...] 路由结果
        original_query: 原始用户查询
    
    Returns:
        任务列表 [TaskItem, ...]
    """
    tasks = []
    
    for card, matched_terms in routes:
        agent_id = card.metadata.agent_id if hasattr(card, 'metadata') else "unknown"
        agent_name = card.metadata.name if hasattr(card, 'metadata') else "Unknown"
        
        # 任务指令就是当前查询
        task = build_a2a_task(
            executor=agent_id,
            agent_name=agent_name,
            instruction=query,
            matched_terms=matched_terms,
            original_query=original_query,
        )
        tasks.append(task)
    
    return tasks


def validate_task_item(task: Dict[str, Any]) -> bool:
    """
    验证任务项的格式是否有效。
    
    支持两种格式：
    1. 新格式（推荐）：target + instruction + metadata（可选）
    2. 旧格式（兼容）：executor + instruction + metadata（可选），executor_type 可选
    
    Args:
        task: 任务字典
    
    Returns:
        True 如果有效，False 否则
    """
    # 检查是新格式（target）还是旧格式（executor）
    has_target = "target" in task
    has_executor = "executor" in task
    
    if not (has_target or has_executor):
        return False
    
    # 必须有 instruction
    if "instruction" not in task:
        return False
    
    # 验证字段值不为空
    required_field = "target" if has_target else "executor"
    if task[required_field] is None or (isinstance(task[required_field], str) and not task[required_field].strip()):
        return False
    if task["instruction"] is None or (isinstance(task["instruction"], str) and not task["instruction"].strip()):
        return False
    
    # 如果有 executor_type，则只接受 "a2a"（所有任务都通过 A2A 调用）
    if "executor_type" in task and task["executor_type"] != "a2a":
        return False
    
    return True


def filter_valid_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤出格式有效的任务。
    
    Args:
        tasks: 任务列表
    
    Returns:
        有效的任务列表
    """
    return [task for task in tasks if validate_task_item(task)]


def merge_task_lists(*task_lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并多个任务列表。
    
    Args:
        task_lists: 多个任务列表
    
    Returns:
        合并后的任务列表
    """
    result = []
    for tasks in task_lists:
        result.extend(tasks)
    return result
