"""
Minimal A2A protocol helpers for remote agent discovery and JSON-RPC calls.

The main service uses only remote Agent Cards and message/send calls. Local
SubAgent imports stay inside each standalone A2A agent service.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

import requests
import yaml

from engine.agent_card import (
    AgentCard,
    CapabilitiesCard,
    ConfigurationCard,
    MetadataCard,
)
from engine.logging_config import get_logger


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_A2A_CONFIG = PROJECT_ROOT / ".config" / "a2a_agents.yaml"

# ─── 默认 A2A 配置常量 (从 YAML 加载时会覆盖) ───
DEFAULT_AGENT_ID_PREFIX = "dify_"
DEFAULT_DOCKER_NAME_PREFIX = "agent-"
DEFAULT_WELL_KNOWN_PATH = "/.well-known/agent-card.json"
DEFAULT_BASE_PORT = 8101


@dataclass
class A2AAgentEndpoint:
    agent_id: str
    card_url: str


def _service_url_for_card_url(card_url: str) -> str:
    if "/.well-known/" in card_url:
        return card_url.split("/.well-known/", 1)[0]
    return card_url.rstrip("/")


def _text_parts(text: str) -> list[dict[str, Any]]:
    return [{"kind": "text", "text": text}]


def local_card_to_a2a(
    card: AgentCard,
    *,
    base_url: str,
) -> dict[str, Any]:
    """
    Convert the project's local AgentCard into an A2A Agent Card.

    Routing-only fields such as keywords and priority are kept under metadata
    so the A2A-native main router can still make the same decisions after
    discovering the remote card over HTTP.
    """
    cap = card.capabilities
    scope = card.custom_attributes.get("scope", [])
    examples = card.custom_attributes.get("examples", [])
    skills = []
    for skill in cap.skills:
        skills.append(
            {
                "id": skill,
                "name": skill,
                "description": skill,
                "tags": list(dict.fromkeys([*cap.keywords, *cap.intent_patterns])),
                "examples": examples,
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"],
            }
        )

    return {
        "protocolVersion": "0.2.6",
        "name": card.metadata.name,
        "description": card.metadata.description,
        "url": f"{base_url.rstrip('/')}/a2a/{card.metadata.agent_id}",
        "version": card.metadata.version,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
        "metadata": {
            "agent_id": card.metadata.agent_id,
            "category": card.metadata.category,
            "author": card.metadata.author,
            "keywords": cap.keywords,
            "intent_patterns": cap.intent_patterns,
            "priority": cap.priority,
            "confidence_threshold": cap.confidence_threshold,
            "scope": scope,
            "examples": examples,
        },
    }


def a2a_card_to_local(card_payload: dict[str, Any]) -> AgentCard:
    """Adapt a discovered A2A Agent Card to the router's AgentCard shape."""
    metadata = card_payload.get("metadata") or {}
    skills_payload = card_payload.get("skills") or []

    agent_id = (
        str(metadata.get("agent_id") or "").strip()
        or str(card_payload.get("name") or "").strip().lower().replace(" ", "_")
    )
    keywords = list(metadata.get("keywords") or [])
    intent_patterns = list(metadata.get("intent_patterns") or [])
    skills = [
        str(skill.get("id") or skill.get("name"))
        for skill in skills_payload
        if isinstance(skill, dict) and (skill.get("id") or skill.get("name"))
    ]

    for skill in skills_payload:
        if not isinstance(skill, dict):
            continue
        for tag in skill.get("tags") or []:
            if tag not in keywords and tag not in intent_patterns:
                keywords.append(str(tag))

    return AgentCard(
        metadata=MetadataCard(
            agent_id=agent_id,
            name=str(card_payload.get("name") or agent_id),
            description=str(card_payload.get("description") or ""),
            version=str(card_payload.get("version") or "1.0.0"),
            category=str(metadata.get("category") or "remote"),
            author=str(metadata.get("author") or "A2A"),
        ),
        capabilities=CapabilitiesCard(
            skills=skills,
            keywords=keywords,
            intent_patterns=intent_patterns,
            confidence_threshold=float(metadata.get("confidence_threshold") or 0.5),
            priority=int(metadata.get("priority") or 5),
        ),
        configuration=ConfigurationCard(),
        custom_attributes={
            "scope": metadata.get("scope", []),
            "examples": metadata.get("examples", []),
            "a2a": {
                "url": card_payload.get("url"),
                "card_url": card_payload.get("card_url"),
                "protocol_version": card_payload.get("protocolVersion"),
                "capabilities": card_payload.get("capabilities", {}),
                "input_modes": card_payload.get("defaultInputModes", []),
                "output_modes": card_payload.get("defaultOutputModes", []),
            },
        },
    )


def _is_in_docker() -> bool:
    """检查是否在 Docker 容器中运行"""
    import os
    return os.path.exists("/.dockerenv")


def _get_a2a_host() -> str:
    """
    根据环境获取 A2A 主机前缀
    
    优先级：
    1. 显式的环境变量 A2A_HOST
    2. 显式的环境变量 A2A_HOST_PREFIX (用于容器名前缀)
    3. 自动检测环境：Docker 返回容器前缀, 本地返回 127.0.0.1
    """
    import os
    
    # 优先使用明确的 A2A_HOST
    if os.getenv("A2A_HOST"):
        host = os.getenv("A2A_HOST")
        logger.info(f"🔧 A2A_HOST from env: {host}")
        return host
    
    # 其次使用 A2A_HOST_PREFIX（Docker 环境）
    if os.getenv("A2A_HOST_PREFIX"):
        host = os.getenv("A2A_HOST_PREFIX")
        logger.info(f"🔧 A2A_HOST_PREFIX from env: {host}")
        return host
    
    # 自动检测环境
    if _is_in_docker():
        logger.info("✅ Detected Docker environment, using container names for A2A agents")
        return DEFAULT_DOCKER_NAME_PREFIX
    else:
        logger.info("✅ Detected local environment, using 127.0.0.1 for A2A agents")
        return "127.0.0.1"


def _load_agents_from_yaml(config_path: Path, agent_ids: list[str]) -> list[AgentCard]:
    """
    从本地 YAML 配置文件加载特定的 agent 配置作为备用。
    
    这用于当远程 HTTP 发现失败时，确保系统至少知道这些 agents 存在。
    """
    from pathlib import Path
    
    cards = []
    agents_dir = PROJECT_ROOT / "agents"
    
    for agent_id in agent_ids:
        # agent_card.yaml 位置: agents/{agent_id}/agent_card.yaml
        card_file = agents_dir / agent_id / "agent_card.yaml"
        
        if not card_file.exists():
            logger.warning(f"⚠️  Local agent card not found: {card_file}")
            continue
        
        try:
            with open(card_file, 'r', encoding='utf-8') as f:
                card_config = yaml.safe_load(f)
            
            # 从 YAML 构建 AgentCard
            metadata = card_config.get('metadata', {})
            capabilities = card_config.get('capabilities', {})
            custom_attrs = {
                'scope': card_config.get('scope', []),
                'examples': card_config.get('examples', []),
            }
            
            card = AgentCard(
                metadata=MetadataCard(
                    agent_id=metadata.get('agent_id', agent_id),
                    name=metadata.get('name', agent_id),
                    description=metadata.get('description', ''),
                    version=metadata.get('version', '1.0.0'),
                    category=metadata.get('category', 'unknown'),
                    author=metadata.get('author', 'Unknown'),
                ),
                capabilities=CapabilitiesCard(
                    skills=capabilities.get('skills', []),
                    keywords=capabilities.get('keywords', []),
                    intent_patterns=capabilities.get('intent_patterns', []),
                    confidence_threshold=float(capabilities.get('confidence_threshold', 0.5)),
                    priority=int(capabilities.get('priority', 5)),
                ),
                custom_attributes=custom_attrs,
            )
            
            cards.append(card)
            logger.info(f"✅ Loaded local agent card for {agent_id} from {card_file}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load local agent card from {card_file}: {e}")
    
    return cards


def load_a2a_endpoints(config_path: str | Path = DEFAULT_A2A_CONFIG) -> list[A2AAgentEndpoint]:
    """
    加载 A2A 代理配置，自动适配 Docker 和本地环境
    
    配置优先级：
    1. 读取 YAML 配置文件中的 config 部分获取常量
    2. 对每个 agent，如果指定了 card_url 则直接使用
    3. 否则根据 agent.port + docker/local 模式自动生成 URL
    """
    import os
    
    path = Path(config_path)
    if not path.exists():
        logger.warning("A2A agent config not found: %s", path)
        return []

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    # 从配置文件读取 A2A 常量（如果存在）
    config = data.get("config", {})
    agent_id_prefix = config.get("agent_id_prefix", DEFAULT_AGENT_ID_PREFIX)
    docker_name_prefix = config.get("docker_name_prefix", DEFAULT_DOCKER_NAME_PREFIX)
    well_known_path = config.get("well_known_path", DEFAULT_WELL_KNOWN_PATH)
    base_port = config.get("base_port", DEFAULT_BASE_PORT)
    local_host = config.get("local_host", "127.0.0.1")

    # 获取环境相关的主机配置
    a2a_host_prefix = _get_a2a_host()
    is_docker_env = a2a_host_prefix == docker_name_prefix

    endpoints = []
    for item in data.get("agents", []):
        if not isinstance(item, dict):
            continue
        
        agent_id = str(item.get("id") or "").strip()
        if not agent_id:
            continue
        
        # 优先使用 YAML 中指定的 card_url
        card_url = str(item.get("card_url") or "").strip()
        
        if card_url:
            # 如果显式指定了 card_url，直接使用
            logger.debug(f"Using explicit card_url for {agent_id}: {card_url}")
        else:
            # 否则根据模式自动生成
            port = item.get("port")
            if not port:
                # 使用索引计算默认端口
                agent_idx = list(data.get("agents", [])).index(item)
                port = base_port + agent_idx
            
            if is_docker_env:
                # Docker 环境：使用容器名 + 端口
                # agent_id 格式：dify_file_uploader, dify_doc_summary 等
                # 容器名格式：agent-file-uploader, agent-doc-summary 等
                if agent_id.startswith(agent_id_prefix):
                    # 提取前缀后的部分并替换下划线
                    agent_name = agent_id[len(agent_id_prefix):].replace("_", "-")
                else:
                    # 如果不是预期前缀，直接替换下划线
                    agent_name = agent_id.replace("_", "-")
                
                card_url = f"http://{a2a_host_prefix}{agent_name}:{port}{well_known_path}"
                logger.debug(f"Docker mode - Agent {agent_id}: {card_url}")
            else:
                # 本地环境：使用 localhost + port
                card_url = f"http://{local_host}:{port}{well_known_path}"
                logger.debug(f"Local mode - Agent {agent_id}: {card_url}")
        
        endpoints.append(A2AAgentEndpoint(agent_id=agent_id, card_url=card_url))
    
    logger.info(f"Loaded {len(endpoints)} A2A agent endpoints (mode: {'docker' if is_docker_env else 'local'})")
    return endpoints


def discover_a2a_agent_cards(
    config_path: str | Path = DEFAULT_A2A_CONFIG,
    *,
    timeout: int = 5,
) -> list[AgentCard]:
    """
    Fetch remote A2A Agent Cards and adapt them for routing.
    
    如果 HTTP 发现失败（例如容器网络问题），尝试从本地 YAML 配置加载备用卡。
    这确保即使远程发现失败，系统仍然知道这些 agents 存在。
    """
    cards = []
    config_path = Path(config_path)
    endpoints = load_a2a_endpoints(config_path)
    failed_agents = []
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint.card_url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            payload["card_url"] = endpoint.card_url
            card = a2a_card_to_local(payload)
            if endpoint.agent_id and card.metadata.agent_id != endpoint.agent_id:
                logger.warning(
                    "A2A card id mismatch for %s: discovered %s",
                    endpoint.card_url,
                    card.metadata.agent_id,
                )
            cards.append(card)
            logger.debug(f"✅ Successfully discovered A2A agent: {endpoint.agent_id} from {endpoint.card_url}")
        except Exception as exc:
            logger.error(f"❌ Failed to discover A2A agent from {endpoint.card_url}: {exc}")
            failed_agents.append(endpoint.agent_id)
    
    # ─── 备用方案：从本地 YAML 配置加载失败的 agents ───
    if failed_agents:
        logger.warning(f"⚠️  {len(failed_agents)} agents failed remote discovery: {failed_agents}. Attempting local YAML fallback...")
        try:
            fallback_cards = _load_agents_from_yaml(config_path, failed_agents)
            cards.extend(fallback_cards)
            logger.info(f"✅ Loaded {len(fallback_cards)} agents from local YAML fallback")
        except Exception as e:
            logger.warning(f"⚠️  Local YAML fallback also failed: {e}")
    
    if not cards:
        logger.error("❌ No A2A agents discovered! Check network connectivity and agent service status.")
    
    return sorted(cards, key=lambda card: card.capabilities.priority, reverse=True)


class A2AClient:
    """Small blocking A2A JSON-RPC client for message/send."""

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def send_message(
        self,
        agent_card: AgentCard,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        a2a_meta = agent_card.custom_attributes.get("a2a", {})
        url = a2a_meta.get("url")
        if not url:
            raise ValueError(f"A2A url is missing for agent {agent_card.metadata.agent_id}")

        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": _text_parts(text),
                    "messageId": message_id,
                },
                "metadata": metadata or {},
            },
        }

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(data["error"])
        return data.get("result") or {}


def extract_text_from_a2a_result(result: dict[str, Any]) -> str:
    """Return the most useful text from an A2A Message or completed Task."""
    if not isinstance(result, dict):
        return str(result)

    for artifact in result.get("artifacts") or []:
        if not isinstance(artifact, dict):
            continue
        text = _extract_text_from_parts(artifact.get("parts") or [])
        if text:
            return text

    status = result.get("status") or {}
    if isinstance(status, dict):
        message = status.get("message") or {}
        text = _extract_text_from_parts(message.get("parts") or [])
        if text:
            return text

    text = _extract_text_from_parts(result.get("parts") or [])
    if text:
        return text

    return str(result)


def _extract_text_from_parts(parts: list[dict[str, Any]]) -> str:
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("kind") == "text" and part.get("text"):
            texts.append(str(part["text"]))
    return "\n".join(texts)
