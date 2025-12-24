"""
Model abstraction layer for SIA-LLM experiments.
Supports GPT-3.5-turbo, GPT-4-turbo, and GPT-4o with cost tracking.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_id: str
    display_name: str
    max_context_tokens: int
    max_output_tokens: int
    cost_per_1k_input: float  # USD per 1K input tokens
    cost_per_1k_output: float  # USD per 1K output tokens
    supports_json_mode: bool = True

    @property
    def cost_per_input_token(self) -> float:
        return self.cost_per_1k_input / 1000

    @property
    def cost_per_output_token(self) -> float:
        return self.cost_per_1k_output / 1000


# Model configurations with pricing (as of Dec 2024)
MODELS = {
    "gpt-3.5-turbo": ModelConfig(
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        max_context_tokens=16385,
        max_output_tokens=4096,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
    ),
    "gpt-4-turbo": ModelConfig(
        model_id="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        max_context_tokens=128000,
        max_output_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        display_name="GPT-4o",
        max_context_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        max_context_tokens=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
}

# Default model selection based on feature flag
DEFAULT_MODEL = "gpt-3.5-turbo"
GPT4_MODEL = "gpt-4o"  # Use GPT-4o as default GPT-4 (best price/performance)


@dataclass
class UsageStats:
    """Track token usage and costs."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0
    total_duration_seconds: float = 0.0

    def add_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        duration: float,
    ):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost_usd += cost
        self.api_calls += 1
        self.total_duration_seconds += duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "api_calls": self.api_calls,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
        }


class ModelHandler:
    """
    Enhanced OpenAI model handler with cost tracking.
    Drop-in replacement for OpenAIHandler with additional features.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        track_costs: bool = True,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=self.api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.track_costs = track_costs

        # Get model config
        if model_id not in MODELS:
            raise ValueError(f"Unknown model '{model_id}'. Available: {list(MODELS.keys())}")
        self.config = MODELS[model_id]

        # Conversation state (compatible with original OpenAIHandler)
        self.conversation_log: List[Dict[str, str]] = []

        # Usage tracking
        self.usage = UsageStats()

    def get_model_response(self) -> Any:
        """
        Get response from the model (compatible with OpenAIHandler).
        Returns the raw OpenAI response object.
        """
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.conversation_log,
            temperature=self.temperature,
        )

        duration = time.time() - start_time

        # Track usage
        if self.track_costs and response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.usage.add_call(prompt_tokens, completion_tokens, cost, duration)

        return response

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for a single API call."""
        return (
            prompt_tokens * self.config.cost_per_input_token +
            completion_tokens * self.config.cost_per_output_token
        )

    def add_model_turn(self, response: Any):
        """Add model response to conversation log (compatible with OpenAIHandler)."""
        turn = {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content.strip()
        }
        self.conversation_log.append(turn)

    def add_user_turn(self, user_turn_utterance: str):
        """Add user message to conversation log (compatible with OpenAIHandler)."""
        turn = {"role": "user", "content": user_turn_utterance}
        self.conversation_log.append(turn)

    def remove_turns(self, i_begin: int, i_end: Optional[int] = None):
        """Remove turns from conversation log (compatible with OpenAIHandler)."""
        if i_end is not None:
            del self.conversation_log[i_begin:i_end]
        else:
            del self.conversation_log[i_begin]

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of all API usage."""
        return {
            "model": self.model_id,
            "model_display_name": self.config.display_name,
            **self.usage.to_dict(),
        }

    def reset_usage(self):
        """Reset usage tracking."""
        self.usage = UsageStats()


class ModelFactory:
    """Factory for creating model handlers based on feature flags."""

    @staticmethod
    def create(
        use_gpt4: bool = False,
        model_override: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        track_costs: bool = True,
    ) -> ModelHandler:
        """
        Create a model handler based on configuration.

        Args:
            use_gpt4: If True, use GPT-4 model (from feature flags)
            model_override: Specific model ID to use (overrides use_gpt4)
            api_key: OpenAI API key (uses env var if not provided)
            temperature: Sampling temperature
            track_costs: Whether to track token usage and costs

        Returns:
            Configured ModelHandler instance
        """
        if model_override:
            model_id = model_override
        elif use_gpt4:
            model_id = GPT4_MODEL
        else:
            model_id = DEFAULT_MODEL

        return ModelHandler(
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            track_costs=track_costs,
        )

    @staticmethod
    def from_feature_flags(flags: Any, **kwargs) -> ModelHandler:
        """
        Create model handler from FeatureFlags object.

        Args:
            flags: FeatureFlags instance
            **kwargs: Additional arguments passed to create()

        Returns:
            Configured ModelHandler instance
        """
        return ModelFactory.create(use_gpt4=flags.use_gpt4, **kwargs)


def get_model(use_gpt4: bool = False, **kwargs) -> ModelHandler:
    """Convenience function to get a model handler."""
    return ModelFactory.create(use_gpt4=use_gpt4, **kwargs)


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model '{model_id}'. Available: {list(MODELS.keys())}")
    return MODELS[model_id]


def list_models() -> List[str]:
    """List available model IDs."""
    return list(MODELS.keys())


if __name__ == "__main__":
    # Demo usage
    print("Available models:")
    for model_id, config in MODELS.items():
        print(f"  {model_id}:")
        print(f"    Context: {config.max_context_tokens:,} tokens")
        print(f"    Cost: ${config.cost_per_1k_input}/1K input, ${config.cost_per_1k_output}/1K output")

    print("\nTesting model creation...")
    model = get_model(use_gpt4=False)
    print(f"Created: {model.config.display_name} ({model.model_id})")

    model_gpt4 = get_model(use_gpt4=True)
    print(f"Created: {model_gpt4.config.display_name} ({model_gpt4.model_id})")
