"""
Agent Card - 统一的Agent声明配置文件
支持YAML加载和序列化
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class MetadataCard:
    """Agent元数据卡"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    author: str = "Agent"


@dataclass
class CapabilitiesCard:
    """能力卡"""
    skills: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    intent_patterns: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.5
    priority: int = 5


@dataclass
class ConfigurationCard:
    """配置卡"""
    max_iterations: int = 30
    timeout: int = 300
    max_input_length: int = 10000
    temperature: float = 0.7


@dataclass
class ExecutionCard:
    """执行卡"""
    module: str  # e.g., "agents.dify_knowledge_qa.subagent"
    class_name: str  # e.g., "DifyKnowledgeQAAgent"
    mode: str = "sync"  # sync or async


@dataclass
class ToolCard:
    """工具卡"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass
class DependencyCard:
    """依赖卡"""
    python_packages: List[str] = field(default_factory=list)
    external_services: List[str] = field(default_factory=list)
    knowledge_bases: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class MonitoringCard:
    """监控和指标卡"""
    enable_logging: bool = True
    log_level: str = "INFO"
    metrics: List[str] = field(default_factory=lambda: ["response_time", "success_rate"])


@dataclass
class AgentCard:
    """完整的Agent Card - 一个Agent的完整声明"""
    metadata: MetadataCard
    capabilities: CapabilitiesCard
    configuration: ConfigurationCard = field(default_factory=ConfigurationCard)
    execution: Optional[ExecutionCard] = None
    tools: List[ToolCard] = field(default_factory=list)
    dependencies: DependencyCard = field(default_factory=DependencyCard)
    monitoring: MonitoringCard = field(default_factory=MonitoringCard)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentCard":
        """从YAML文件加载Agent Card"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent Card YAML文件不存在: {yaml_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """从字典加载Agent Card"""
        # 解析metadata
        metadata_data = data.get('metadata', {})
        metadata = MetadataCard(**metadata_data)
        
        # 解析capabilities
        capabilities_data = data.get('capabilities', {})
        capabilities = CapabilitiesCard(**capabilities_data)
        
        # 解析configuration
        config_data = data.get('configuration', {})
        configuration = ConfigurationCard(**config_data)
        
        # 解析execution
        execution = None
        execution_data = data.get('execution')
        if execution_data:
            execution = ExecutionCard(**execution_data)
        
        # 解析tools
        tools = []
        tools_data = data.get('tools', [])
        for tool_data in tools_data:
            tools.append(ToolCard(**tool_data))
        
        # 解析dependencies
        dependencies_data = data.get('dependencies', {})
        dependencies = DependencyCard(**dependencies_data)
        
        # 解析monitoring
        monitoring_data = data.get('monitoring', {})
        monitoring = MonitoringCard(**monitoring_data)
        
        # 自定义属性（其他所有字段）
        custom_keys = {'metadata', 'capabilities', 'configuration', 'execution', 'tools', 'dependencies', 'monitoring'}
        custom_attributes = {k: v for k, v in data.items() if k not in custom_keys}
        
        return cls(
            metadata=metadata,
            capabilities=capabilities,
            configuration=configuration,
            execution=execution,
            tools=tools,
            dependencies=dependencies,
            monitoring=monitoring,
            custom_attributes=custom_attributes
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        
        # metadata
        result['metadata'] = asdict(self.metadata)
        
        # capabilities
        result['capabilities'] = asdict(self.capabilities)
        
        # configuration
        result['configuration'] = asdict(self.configuration)
        
        # execution
        if self.execution:
            result['execution'] = asdict(self.execution)
        
        # tools
        if self.tools:
            result['tools'] = [asdict(tool) for tool in self.tools]
        
        # dependencies
        result['dependencies'] = asdict(self.dependencies)
        
        # monitoring
        result['monitoring'] = asdict(self.monitoring)
        
        # custom attributes
        result.update(self.custom_attributes)
        
        return result

    def to_yaml(self, output_path: str) -> None:
        """保存为YAML文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)

    def __str__(self) -> str:
        """字符串表示"""
        return f"AgentCard({self.metadata.agent_id}@{self.metadata.version})"

    def __repr__(self) -> str:
        """调试表示"""
        return f"AgentCard(id={self.metadata.agent_id}, name={self.metadata.name}, priority={self.capabilities.priority})"
