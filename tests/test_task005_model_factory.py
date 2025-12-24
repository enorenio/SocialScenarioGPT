"""
Tests for TASK-005: GPT-4 Model Integration
Tests the model factory and model handler with cost tracking.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_factory import (
    ModelConfig,
    ModelHandler,
    ModelFactory,
    UsageStats,
    MODELS,
    get_model,
    get_model_config,
    list_models,
    DEFAULT_MODEL,
    GPT4_MODEL,
)
from config.feature_flags import FeatureFlags


def test_models_defined():
    """Test that all expected models are defined."""
    expected = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]

    for model_id in expected:
        assert model_id in MODELS, f"Missing model: {model_id}"
        config = MODELS[model_id]
        assert isinstance(config, ModelConfig)
        assert config.max_context_tokens > 0
        assert config.cost_per_1k_input >= 0
        assert config.cost_per_1k_output >= 0

    print(f"✓ All {len(expected)} models defined correctly")


def test_model_config_properties():
    """Test ModelConfig cost calculation properties."""
    config = MODELS["gpt-3.5-turbo"]

    # Cost per token should be cost per 1K / 1000
    assert config.cost_per_input_token == config.cost_per_1k_input / 1000
    assert config.cost_per_output_token == config.cost_per_1k_output / 1000

    print("✓ ModelConfig cost properties work correctly")


def test_get_model_default():
    """Test get_model with default settings."""
    model = get_model(use_gpt4=False)

    assert model.model_id == DEFAULT_MODEL
    assert model.model_id == "gpt-3.5-turbo"
    assert model.config.display_name == "GPT-3.5 Turbo"

    print(f"✓ Default model: {model.model_id}")


def test_get_model_gpt4():
    """Test get_model with use_gpt4=True."""
    model = get_model(use_gpt4=True)

    assert model.model_id == GPT4_MODEL
    assert model.model_id == "gpt-4o"
    assert model.config.max_context_tokens == 128000

    print(f"✓ GPT-4 model: {model.model_id}")


def test_model_override():
    """Test model override parameter."""
    model = get_model(use_gpt4=False, model_override="gpt-4o-mini")

    assert model.model_id == "gpt-4o-mini"

    print(f"✓ Model override works: {model.model_id}")


def test_model_factory_from_feature_flags():
    """Test ModelFactory.from_feature_flags()."""
    # Baseline flags (use_gpt4=False)
    flags_baseline = FeatureFlags()
    model_baseline = ModelFactory.from_feature_flags(flags_baseline)
    assert model_baseline.model_id == "gpt-3.5-turbo"

    # GPT-4 flags
    flags_gpt4 = FeatureFlags(use_gpt4=True)
    model_gpt4 = ModelFactory.from_feature_flags(flags_gpt4)
    assert model_gpt4.model_id == "gpt-4o"

    print("✓ ModelFactory.from_feature_flags() works")


def test_usage_stats():
    """Test UsageStats tracking."""
    stats = UsageStats()

    assert stats.total_tokens == 0
    assert stats.api_calls == 0

    # Add a call
    stats.add_call(
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.001,
        duration=1.5,
    )

    assert stats.prompt_tokens == 100
    assert stats.completion_tokens == 50
    assert stats.total_tokens == 150
    assert stats.api_calls == 1
    assert stats.total_cost_usd == 0.001
    assert stats.total_duration_seconds == 1.5

    # Add another call
    stats.add_call(
        prompt_tokens=200,
        completion_tokens=100,
        cost=0.002,
        duration=2.0,
    )

    assert stats.total_tokens == 450
    assert stats.api_calls == 2
    assert stats.total_cost_usd == 0.003

    print("✓ UsageStats tracking works")


def test_usage_stats_to_dict():
    """Test UsageStats serialization."""
    stats = UsageStats()
    stats.add_call(100, 50, 0.00123456, 1.5)

    d = stats.to_dict()

    assert "prompt_tokens" in d
    assert "completion_tokens" in d
    assert "total_tokens" in d
    assert "total_cost_usd" in d
    assert "api_calls" in d
    assert d["total_cost_usd"] == 0.001235  # Rounded to 6 decimals

    print("✓ UsageStats.to_dict() works")


def test_model_handler_conversation_log():
    """Test ModelHandler conversation management."""
    model = get_model(use_gpt4=False)

    assert len(model.conversation_log) == 0

    model.add_user_turn("Hello")
    assert len(model.conversation_log) == 1
    assert model.conversation_log[0]["role"] == "user"
    assert model.conversation_log[0]["content"] == "Hello"

    model.remove_turns(-1)
    assert len(model.conversation_log) == 0

    print("✓ ModelHandler conversation management works")


def test_list_models():
    """Test list_models utility."""
    models = list_models()

    assert isinstance(models, list)
    assert len(models) >= 4
    assert "gpt-3.5-turbo" in models
    assert "gpt-4o" in models

    print(f"✓ list_models() returns {len(models)} models")


def test_get_model_config():
    """Test get_model_config utility."""
    config = get_model_config("gpt-4o")

    assert config.model_id == "gpt-4o"
    assert config.max_context_tokens == 128000

    # Test invalid model
    try:
        get_model_config("invalid-model")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown model" in str(e)

    print("✓ get_model_config() works")


def test_api_call_live():
    """Test actual API call (requires OPENAI_API_KEY)."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ Skipping live API test: OPENAI_API_KEY not set")
        return

    model = get_model(use_gpt4=False)
    model.add_user_turn("Say 'test ok' and nothing else.")

    response = model.get_model_response()

    assert response is not None
    assert response.choices[0].message.content
    assert model.usage.api_calls == 1
    assert model.usage.total_tokens > 0

    summary = model.get_usage_summary()
    assert summary["model"] == "gpt-3.5-turbo"
    assert summary["api_calls"] == 1

    print(f"✓ Live API call works, tokens: {model.usage.total_tokens}")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-005: GPT-4 Model Integration Tests")
    print("=" * 60)

    test_models_defined()
    test_model_config_properties()
    test_get_model_default()
    test_get_model_gpt4()
    test_model_override()
    test_model_factory_from_feature_flags()
    test_usage_stats()
    test_usage_stats_to_dict()
    test_model_handler_conversation_log()
    test_list_models()
    test_get_model_config()
    test_api_call_live()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
