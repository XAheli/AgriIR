from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTool(ABC):
    """Base class for all tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for Ollama function calling"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for Ollama"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Ollama function parameters schema"""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    def to_ollama_schema(self) -> Dict[str, Any]:
        """Convert tool to Ollama function schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }