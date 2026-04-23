"""
Role-Based Access Control (RBAC) module for Agent Dispatch Service.

This module manages role permissions and filters accessible agents based on user roles.
Configuration is loaded from .config/role_permissions.yaml
"""

from dataclasses import dataclass
from typing import Optional, Set
import yaml

from engine.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RolePermission:
    """Represents a role and its agent access permissions."""
    name: str
    description: str
    accessible_agents: Set[str]  # empty set means all agents are accessible


class RoleBasedAccessControl:
    """
    Manages role-based access control for agent dispatch.
    
    Features:
    - Load role permissions from YAML configuration
    - Filter agents based on user roles
    - Support for unrestricted roles (admin)
    - Default role fallback
    """
    
    def __init__(self, config_path: str = ".config/role_permissions.yaml"):
        """
        Initialize RBAC with configuration file.
        
        Args:
            config_path: Path to role permissions YAML configuration
        """
        self.config_path = config_path
        self.roles: dict[str, RolePermission] = {}
        self.default_role: Optional[str] = None
        self.load_config()
    
    def load_config(self):
        """
        Load role permissions from YAML config file.
        
        YAML structure:
        roles:
          role_id:
            name: "Display Name"
            description: "Description"
            accessible_agents: [agent1, agent2]  # empty list = all agents
        default_role: role_id (optional)
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                logger.warning(f"Role config file {self.config_path} is empty")
                return
            
            for role_id, role_data in config.get("roles", {}).items():
                agents = role_data.get("accessible_agents", [])
                # Empty list means all agents are accessible
                agent_set = set(agents) if agents else set()
                
                self.roles[role_id] = RolePermission(
                    name=role_data.get("name", role_id),
                    description=role_data.get("description", ""),
                    accessible_agents=agent_set
                )
            
            self.default_role = config.get("default_role")
            logger.info(
                f"Loaded {len(self.roles)} roles from {self.config_path}. "
                f"Default role: {self.default_role}"
            )
        
        except FileNotFoundError:
            logger.warning(f"Role config not found at {self.config_path}, RBAC disabled")
        except Exception as e:
            logger.error(f"Failed to load role config: {e}")
    
    def get_accessible_agents(self, role_id: Optional[str]) -> Optional[Set[str]]:
        """
        Get accessible agent IDs for a role.
        
        Args:
            role_id: The role identifier
            
        Returns:
            Set[str]: agent IDs that this role can access
            None: if role not found
            empty set: if role has no restrictions (admin)
        """
        if role_id is None:
            role_id = self.default_role
        
        if role_id is None:
            return None
        
        if role_id not in self.roles:
            logger.warning(f"Unknown role: {role_id}")
            return None
        
        permission = self.roles[role_id]
        # Empty set means access to all agents
        return permission.accessible_agents if permission.accessible_agents else set()
    
    def is_agent_accessible(self, agent_id: str, role_id: Optional[str]) -> bool:
        """
        Check if a specific agent is accessible by a role.
        
        Args:
            agent_id: The agent identifier
            role_id: The role identifier
            
        Returns:
            True if agent is accessible, False otherwise
        """
        accessible = self.get_accessible_agents(role_id)
        
        if accessible is None:
            # No role specified and no default, deny access
            return False
        
        if not accessible:
            # Empty set means all agents are accessible
            return True
        
        return agent_id in accessible
    
    def filter_cards(self, cards: list, role_id: Optional[str]) -> list:
        """
        Filter agent cards based on role permissions.
        
        Args:
            cards: List of AgentCard objects
            role_id: The role identifier
            
        Returns:
            Filtered list of cards accessible by the role
        """
        accessible = self.get_accessible_agents(role_id)
        
        if accessible is None:
            logger.warning(f"Attempt to access agents with no valid role: {role_id}")
            return []
        
        if not accessible:
            # Empty set means all agents are accessible
            return cards
        
        filtered = [
            card for card in cards 
            if card.metadata.agent_id in accessible
        ]
        
        logger.info(
            f"Role '{role_id}' filtered {len(cards)} cards to {len(filtered)} "
            f"accessible agents: {[card.metadata.agent_id for card in filtered]}"
        )
        return filtered
    
    def get_role_info(self, role_id: str) -> Optional[dict]:
        """
        Get information about a specific role.
        
        Args:
            role_id: The role identifier
            
        Returns:
            Dict with role info or None if role not found
        """
        if role_id not in self.roles:
            return None
        
        perm = self.roles[role_id]
        return {
            "id": role_id,
            "name": perm.name,
            "description": perm.description,
            "accessible_agents": list(perm.accessible_agents) if perm.accessible_agents else "all",
        }
    
    def list_all_roles(self) -> list[dict]:
        """
        Get information about all configured roles.
        
        Returns:
            List of role information dicts
        """
        return [
            {
                "id": role_id,
                "name": perm.name,
                "description": perm.description,
                "accessible_agents": list(perm.accessible_agents) if perm.accessible_agents else "all",
            }
            for role_id, perm in self.roles.items()
        ]
