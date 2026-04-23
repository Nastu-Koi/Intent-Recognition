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


def load_a2a_endpoints(config_path: str | Path = DEFAULT_A2A_CONFIG) -> list[A2AAgentEndpoint]:
    path = Path(config_path)
    if not path.exists():
        logger.warning("A2A agent config not found: %s", path)
        return []

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    endpoints = []
    for item in data.get("agents", []):
        if not isinstance(item, dict):
            continue
        card_url = str(item.get("card_url") or "").strip()
        agent_id = str(item.get("id") or "").strip()
        if card_url:
            endpoints.append(A2AAgentEndpoint(agent_id=agent_id, card_url=card_url))
    return endpoints


def discover_a2a_agent_cards(
    config_path: str | Path = DEFAULT_A2A_CONFIG,
    *,
    timeout: int = 5,
) -> list[AgentCard]:
    """Fetch remote A2A Agent Cards and adapt them for routing."""
    cards = []
    for endpoint in load_a2a_endpoints(config_path):
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
        except Exception as exc:
            logger.error("Failed to discover A2A agent from %s: %s", endpoint.card_url, exc)

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
