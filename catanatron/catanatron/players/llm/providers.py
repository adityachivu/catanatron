"""
LLM Provider Abstraction - Unified interface for multiple LLM providers.

This module provides a thin abstraction layer over PydanticAI to support:
- OpenAI (gpt-4, gpt-3.5-turbo, etc.)
- Anthropic (claude-3-opus, claude-3-sonnet, etc.)
- Google Gemini (gemini-1.5-pro, gemini-1.5-flash, etc.)
- Ollama (local models like llama3, mistral, etc.)

The abstraction allows easy switching between providers without changing
the player implementation.
"""

from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Generic
from enum import Enum
import asyncio
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"  # Google's Gemini models
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None  # Can be set via env var
    base_url: Optional[str] = None  # For Ollama or custom endpoints
    retries: int = 3
    timeout: float = 30.0
    
    @classmethod
    def openai(
        cls, 
        model: str = "gpt-4", 
        temperature: float = 0.7,
        **kwargs
    ) -> "LLMConfig":
        """Create an OpenAI configuration."""
        return cls(
            provider=LLMProvider.OPENAI,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    @classmethod
    def anthropic(
        cls, 
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        **kwargs
    ) -> "LLMConfig":
        """Create an Anthropic configuration."""
        return cls(
            provider=LLMProvider.ANTHROPIC,
            model=model,
            temperature=temperature,
            **kwargs
        )
    
    @classmethod
    def ollama(
        cls,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        **kwargs
    ) -> "LLMConfig":
        """Create an Ollama configuration."""
        return cls(
            provider=LLMProvider.OLLAMA,
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
    
    @classmethod
    def gemini(
        cls,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        **kwargs
    ) -> "LLMConfig":
        """Create a Google Gemini configuration.
        
        Args:
            model: Gemini model name (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
            temperature: Sampling temperature
            api_key: Optional API key (can also be set via GOOGLE_API_KEY env var)
            **kwargs: Additional configuration options
            
        Returns:
            LLMConfig for Gemini
        """
        return cls(
            provider=LLMProvider.GEMINI,
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs
        )
    
    def get_pydantic_ai_model_string(self) -> str:
        """Get the model string format expected by PydanticAI."""
        if self.provider == LLMProvider.OPENAI:
            return f"openai:{self.model}"
        elif self.provider == LLMProvider.ANTHROPIC:
            return f"anthropic:{self.model}"
        elif self.provider == LLMProvider.GEMINI:
            # PydanticAI uses "google-gla:" for Google Generative Language API (standard API)
            return f"google-gla:{self.model}"
        elif self.provider == LLMProvider.OLLAMA:
            return f"ollama:{self.model}"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


T = TypeVar('T')


class LLMClient(Generic[T]):
    """
    Generic LLM client that wraps PydanticAI Agent.
    
    This provides a consistent interface for making LLM calls with
    structured output, regardless of the underlying provider.
    
    Usage:
        from pydantic import BaseModel
        
        class MyResponse(BaseModel):
            answer: str
            confidence: float
        
        client = LLMClient(config, MyResponse)
        result = await client.run("What is 2+2?")
        print(result.answer)  # "4"
    """
    
    def __init__(
        self, 
        config: LLMConfig, 
        result_type: Type[T],
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            config: LLM provider configuration
            result_type: Pydantic model class for structured output
            system_prompt: Optional system prompt to use
        """
        self.config = config
        self.result_type = result_type
        self.system_prompt = system_prompt
        self._agent = None
        self._initialized = False
    
    def _validate_api_key(self):
        """Validate that required API key is set."""
        key_map = {
            LLMProvider.GEMINI: "GOOGLE_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY", 
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OLLAMA: None,  # No key needed
        }
        
        required_key = key_map.get(self.config.provider)
        if required_key and not os.getenv(required_key):
            raise ValueError(
                f"{required_key} environment variable not set! "
                f"Set it with: export {required_key}='your-api-key-here'"
            )
    
    def _ensure_initialized(self):
        """Lazily initialize the PydanticAI agent."""
        if self._initialized:
            return
        
        # Validate API keys are set
        logger.info(f"Initializing LLM client: {self.config.provider.value}:{self.config.model}")
        self._validate_api_key()
        
        try:
            from pydantic_ai import Agent
        except ImportError:
            raise ImportError(
                "pydantic-ai is required for LLM features. "
                "Install with: pip install pydantic-ai"
            )
        
        model_string = self.config.get_pydantic_ai_model_string()
        logger.debug(f"Creating PydanticAI agent with model string: {model_string}")
        
        self._agent = Agent(
            model_string,
            output_type=self.result_type,  # Changed from result_type to output_type
            system_prompt=self.system_prompt or "",
            retries=self.config.retries,
        )
        
        self._initialized = True
        logger.info(f"LLM client initialized successfully: {self.config.provider.value}:{self.config.model}")
    
    async def run(self, prompt: str, **kwargs) -> T:
        """
        Run the LLM with the given prompt.
        
        Args:
            prompt: The user prompt to send
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            Parsed response of type T
        """
        self._ensure_initialized()
        
        logger.debug(f"Sending prompt to LLM ({self.config.provider.value}:{self.config.model})")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        result = await self._agent.run(prompt, **kwargs)
        
        logger.debug(f"Received response from LLM ({self.config.provider.value}:{self.config.model})")
        return result.output  # Changed from result.data to result.output per pydantic-ai API
    
    def run_sync(self, prompt: str, **kwargs) -> T:
        """
        Synchronous version of run() for non-async contexts.
        
        Args:
            prompt: The user prompt to send
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            Parsed response of type T
        """
        self._ensure_initialized()
        
        logger.debug(f"Sending prompt to LLM ({self.config.provider.value}:{self.config.model}) [sync]")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Use the agent's run_sync if available, otherwise use asyncio.run
        if hasattr(self._agent, 'run_sync'):
            result = self._agent.run_sync(prompt, **kwargs)
        else:
            result = asyncio.run(self.run(prompt, **kwargs))
        
        logger.debug(f"Received response from LLM ({self.config.provider.value}:{self.config.model}) [sync]")
        return result.output if hasattr(result, 'output') else result


class MockLLMClient(Generic[T]):
    """
    Mock LLM client for testing without actual LLM calls.
    
    This is useful for:
    - Unit testing without API costs
    - Deterministic test scenarios
    - Offline development
    """
    
    def __init__(
        self, 
        result_type: Type[T],
        default_response: Optional[T] = None,
        response_factory: Optional[callable] = None
    ):
        """
        Initialize the mock client.
        
        Args:
            result_type: Pydantic model class for structured output
            default_response: Fixed response to always return
            response_factory: Function that takes prompt and returns response
        """
        self.result_type = result_type
        self.default_response = default_response
        self.response_factory = response_factory
        self.call_history: list[str] = []
    
    async def run(self, prompt: str, **kwargs) -> T:
        """Mock run that returns predetermined response."""
        self.call_history.append(prompt)
        
        if self.response_factory:
            return self.response_factory(prompt)
        
        if self.default_response:
            return self.default_response
        
        # Try to create a default instance
        raise ValueError(
            "MockLLMClient needs either default_response or response_factory"
        )
    
    def run_sync(self, prompt: str, **kwargs) -> T:
        """Synchronous version of mock run."""
        return asyncio.run(self.run(prompt, **kwargs))


def create_llm_client(
    config: LLMConfig,
    result_type: Type[T],
    system_prompt: Optional[str] = None
) -> LLMClient[T]:
    """
    Factory function to create an LLM client.
    
    Args:
        config: LLM provider configuration
        result_type: Pydantic model class for structured output
        system_prompt: Optional system prompt
        
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(config, result_type, system_prompt)


# Pre-configured default configurations
DEFAULT_CONFIGS = {
    "gpt-4": LLMConfig.openai("gpt-4"),
    "gpt-4-turbo": LLMConfig.openai("gpt-4-turbo"),
    "gpt-3.5-turbo": LLMConfig.openai("gpt-3.5-turbo"),
    "claude-3-opus": LLMConfig.anthropic("claude-3-opus-20240229"),
    "claude-3-sonnet": LLMConfig.anthropic("claude-3-sonnet-20240229"),
    "claude-3-haiku": LLMConfig.anthropic("claude-3-haiku-20240307"),
    "gemini-pro": LLMConfig.gemini("gemini-2.5-pro"),
    "gemini-flash": LLMConfig.gemini("gemini-2.5-flash"),
    "llama3": LLMConfig.ollama("llama3"),
    "mistral": LLMConfig.ollama("mistral"),
}


def get_config(name: str) -> LLMConfig:
    """
    Get a pre-configured LLM config by name.
    
    Args:
        name: One of the keys in DEFAULT_CONFIGS
        
    Returns:
        LLMConfig for the specified model
    """
    if name not in DEFAULT_CONFIGS:
        available = ", ".join(DEFAULT_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return DEFAULT_CONFIGS[name]

