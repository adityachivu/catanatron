"""
Model factory for LLM player configuration.

Provides flexible model creation with support for:
- Multiple providers (Anthropic, OpenAI, Gemini, etc.)
- Provider-specific settings (temperature, max_tokens)
- TestModel/FunctionModel for testing
- Environment variable overrides
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any

from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.settings import ModelSettings

# Environment variables
TEST_MODE_ENV = "CATAN_LLM_TEST_MODE"
DEFAULT_MODEL_ENV = "CATAN_LLM_MODEL"

# Default model identifier
DEFAULT_MODEL = "anthropic:claude-sonnet-4-20250514"


@dataclass
class ModelConfig:
    """
    Configuration for creating LLM models.
    
    This dataclass allows specifying both the model identifier and
    runtime settings in a single object.
    
    Attributes:
        model_name: Model identifier (e.g., "anthropic:claude-sonnet-4-20250514")
        temperature: Sampling temperature (0.0-1.0+)
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        timeout: Request timeout in seconds
        extra_settings: Provider-specific settings
    
    Example:
        config = ModelConfig(
            model_name="openai:gpt-4o",
            temperature=0.7,
            max_tokens=2048
        )
        player = PydanticAIPlayer(Color.RED, model=config)
    """
    
    # Model identifier (e.g., "anthropic:claude-sonnet-4-20250514", "openai:gpt-4o")
    model_name: str = DEFAULT_MODEL
    
    # Model settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    timeout: Optional[float] = 120.0
    
    # Provider-specific settings (passed through to provider)
    extra_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_model_settings(self) -> Optional[ModelSettings]:
        """
        Convert to PydanticAI ModelSettings.
        
        Returns:
            ModelSettings instance if any settings are configured, None otherwise.
        """
        settings = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            settings["top_p"] = self.top_p
        if self.timeout is not None:
            settings["timeout"] = self.timeout
        
        # Merge extra settings
        settings.update(self.extra_settings)
        
        return ModelSettings(**settings) if settings else None


# Type alias for model input
ModelInput = Union[str, Model, ModelConfig, None]


def create_model(
    model: ModelInput = None,
    **kwargs
) -> Union[Model, str]:
    """
    Factory function to create a Model instance or return a model identifier.
    
    This function provides a unified interface for model configuration.
    It handles:
    - String model identifiers (passthrough)
    - Pre-configured Model instances (passthrough)  
    - ModelConfig objects (extract model_name)
    - Environment variable overrides
    - Test mode override
    
    Args:
        model: Can be:
            - str: Model identifier (e.g., "anthropic:claude-sonnet-4-20250514")
            - Model: Pre-configured Model instance (including TestModel, FunctionModel)
            - ModelConfig: Configuration dataclass
            - None: Use default from env var or fallback
        **kwargs: Additional settings (currently unused, reserved for future)
    
    Returns:
        A Model instance or string identifier ready for use with Agent
    
    Examples:
        # String shorthand
        model = create_model("openai:gpt-4o")
        
        # Direct Model instance (passthrough)
        model = create_model(TestModel(seed=42))
        
        # From ModelConfig
        config = ModelConfig(model_name="openai:gpt-4o", temperature=0.5)
        model = create_model(config)
        
        # Default from environment
        model = create_model()  # Uses CATAN_LLM_MODEL env var or default
    """
    # Check for test mode override first
    if os.environ.get(TEST_MODE_ENV, "").lower() in ("1", "true"):
        return TestModel()
    
    # If already a Model instance, return as-is (passthrough)
    if isinstance(model, Model):
        return model
    
    # If ModelConfig, extract the model name
    if isinstance(model, ModelConfig):
        return model.model_name
    
    # If None, check env var or use default
    if model is None:
        return os.environ.get(DEFAULT_MODEL_ENV, DEFAULT_MODEL)
    
    # If string, return as-is
    if isinstance(model, str):
        return model
    
    raise ValueError(f"Invalid model type: {type(model)}. Expected str, Model, ModelConfig, or None.")


def is_test_model(model: ModelInput) -> bool:
    """
    Check if a model is a test model (TestModel or FunctionModel).
    
    Args:
        model: Model input to check
    
    Returns:
        True if the model is a test/mock model
    """
    if isinstance(model, (TestModel, FunctionModel)):
        return True
    if os.environ.get(TEST_MODE_ENV, "").lower() in ("1", "true"):
        return True
    return False


# ============ Convenience Functions for Common Providers ============


def anthropic_model(
    model_name: str = "claude-sonnet-4-20250514",
) -> str:
    """
    Create an Anthropic model identifier.
    
    Args:
        model_name: Anthropic model name (without provider prefix)
    
    Returns:
        Full model identifier string
    
    Example:
        model = anthropic_model("claude-sonnet-4-20250514")
        # Returns: "anthropic:claude-sonnet-4-20250514"
    """
    return f"anthropic:{model_name}"


def openai_model(
    model_name: str = "gpt-4o",
) -> str:
    """
    Create an OpenAI model identifier.
    
    Args:
        model_name: OpenAI model name (without provider prefix)
    
    Returns:
        Full model identifier string
    
    Example:
        model = openai_model("gpt-4o")
        # Returns: "openai:gpt-4o"
    """
    return f"openai:{model_name}"


def gemini_model(
    model_name: str = "gemini-1.5-pro",
) -> str:
    """
    Create a Google Gemini model identifier.
    
    Args:
        model_name: Gemini model name (without provider prefix)
    
    Returns:
        Full model identifier string
    
    Example:
        model = gemini_model("gemini-1.5-pro")
        # Returns: "google-gla:gemini-1.5-pro"
    """
    return f"google-gla:{model_name}"


def groq_model(
    model_name: str = "llama-3.3-70b-versatile",
) -> str:
    """
    Create a Groq model identifier.
    
    Args:
        model_name: Groq model name (without provider prefix)
    
    Returns:
        Full model identifier string
    
    Example:
        model = groq_model("llama-3.3-70b-versatile")
        # Returns: "groq:llama-3.3-70b-versatile"
    """
    return f"groq:{model_name}"
