"""
Tests for TASK-007: Verification Loop for Conditions/Effects
Tests the verification system for checking consistency.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenario_state import ScenarioState, Agent, Action
from core.verification import (
    VerificationError,
    VerificationResult,
    ConditionEffectVerifier,
    BeliefDesireParser,
    verify_conditions_effects,
    verify_scenario,
    ErrorSeverity,
)
from core.error_feedback import (
    format_errors_for_llm,
    format_regeneration_prompt,
    format_knowledge_base_context,
    create_regeneration_request,
    RegenerationRequest,
    format_fix_summary,
)


# ============================================================
# Parser Tests
# ============================================================

def test_parser_with_value():
    """Test parsing BEL/DES statements with values."""
    parser = BeliefDesireParser()

    result = parser.parse("BEL(Alice, happy) = True")
    assert result is not None
    assert result["type"] == "BEL"
    assert result["agent"] == "Alice"
    assert result["property"] == "happy"
    assert result["value"] == "True"

    result = parser.parse("DES(Bob, help_friend) = False")
    assert result["type"] == "DES"
    assert result["agent"] == "Bob"
    assert result["value"] == "False"

    print("✓ Parser handles statements with values")


def test_parser_without_value():
    """Test parsing statements without values (should return None for value)."""
    parser = BeliefDesireParser()

    result = parser.parse("BEL(Alice, happy)")
    assert result is not None
    assert result["type"] == "BEL"
    assert result["agent"] == "Alice"
    assert result["property"] == "happy"
    assert result["value"] is None

    print("✓ Parser handles statements without values")


def test_parser_intent():
    """Test parsing INTENT statements."""
    parser = BeliefDesireParser()

    result = parser.parse("INTENT(Alice, help_Bob) = True")
    assert result is not None
    assert result["type"] == "INTENT"
    assert result["agent"] == "Alice"
    assert result["property"] == "help_Bob"

    print("✓ Parser handles INTENT statements")


def test_parser_extract_key():
    """Test extracting key from statement."""
    parser = BeliefDesireParser()

    key = parser.extract_key("BEL(Alice, happy) = True")
    assert key == "BEL(Alice, happy)"

    key = parser.extract_key("DES(Bob, rest) = False")
    assert key == "DES(Bob, rest)"

    print("✓ extract_key works correctly")


def test_parser_invalid():
    """Test parser returns None for invalid input."""
    parser = BeliefDesireParser()

    assert parser.parse("invalid input") is None
    assert parser.parse("Hello(world)") is None
    assert parser.parse("") is None

    print("✓ Parser returns None for invalid input")


# ============================================================
# Verifier Tests
# ============================================================

def test_verify_missing_value():
    """Test verification catches missing values."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, happy) = True")

    verifier = ConditionEffectVerifier(state)
    errors = verifier.verify_condition("BEL(Alice, sad)", "Alice", "TestAction")

    assert len(errors) == 1
    assert errors[0].error_type == "MISSING_VALUE"

    print("✓ Verification catches missing values")


def test_verify_unknown_agent():
    """Test verification catches unknown agent references."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")

    verifier = ConditionEffectVerifier(state)
    errors = verifier.verify_condition("BEL(Bob, happy) = True", "Alice", "TestAction")

    assert len(errors) == 1
    assert errors[0].error_type == "UNKNOWN_AGENT"

    print("✓ Verification catches unknown agents")


def test_verify_invalid_format():
    """Test verification catches invalid format."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")

    verifier = ConditionEffectVerifier(state)
    errors = verifier.verify_condition("invalid condition", "Alice", "TestAction")

    assert len(errors) == 1
    assert errors[0].error_type == "INVALID_FORMAT"

    print("✓ Verification catches invalid format")


def test_verify_valid_condition():
    """Test verification passes for valid conditions."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, happy) = True")

    verifier = ConditionEffectVerifier(state)
    errors = verifier.verify_condition("BEL(Alice, happy) = True", "Alice", "TestAction")

    # Should have no errors (maybe a warning about unknown belief, but that's ok)
    error_count = len([e for e in errors if e.severity != ErrorSeverity.WARNING])
    assert error_count == 0

    print("✓ Verification passes for valid conditions")


def test_verify_effect():
    """Test effect verification."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")

    verifier = ConditionEffectVerifier(state)

    # Valid effect
    errors = verifier.verify_effect("BEL(Alice, helped) = True", "Alice", "TestAction")
    error_count = len([e for e in errors if e.severity == ErrorSeverity.ERROR])
    assert error_count == 0

    # Invalid effect (unknown agent)
    errors = verifier.verify_effect("BEL(Unknown, x) = True", "Alice", "TestAction")
    assert any(e.error_type == "UNKNOWN_AGENT" for e in errors)

    print("✓ Effect verification works")


def test_verify_action():
    """Test action verification."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")

    # Add an action with issues
    state.add_action(
        "Alice",
        "DoSomething(Alice)",
        conditions=["BEL(Alice, ready) = True", "BEL(Unknown, x) = True"],
        effects=["BEL(Alice, done) = True"],
    )

    verifier = ConditionEffectVerifier(state)
    action = state.agents["Alice"].actions["DoSomething(Alice)"]
    result = verifier.verify_action(action, "Alice")

    assert not result.valid
    assert len(result.errors) >= 1

    print("✓ Action verification works")


def test_verify_scenario():
    """Test scenario-wide verification."""
    state = ScenarioState("test", "Test scenario")
    state.add_agent("Alice")
    state.add_agent("Bob")
    state.add_belief("Alice", "BEL(Alice, happy) = True")

    state.add_action(
        "Alice",
        "Greet(Alice, Bob)",
        conditions=["BEL(Alice, happy) = True"],
        effects=["BEL(Bob, greeted) = True"],
    )

    result = verify_scenario(state)

    # This should be valid
    assert result.valid or len(result.errors) == 0

    print("✓ Scenario verification works")


def test_verify_real_scenario():
    """Test verification on a real scenario file."""
    test_file = Path("Data/test_Brother.json")
    if not test_file.exists():
        print("⚠ Skipping real scenario test: file not found")
        return

    state = ScenarioState.from_file(str(test_file))
    result = verify_scenario(state)

    # Real scenarios likely have some issues
    print(f"  Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")

    # Verify we can process the whole thing without crashing
    assert isinstance(result, VerificationResult)

    print("✓ Real scenario verification completes")


def test_verify_conditions_effects_function():
    """Test the convenience function."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, exists) = True")

    conditions = ["BEL(Alice, exists) = True"]
    effects = ["BEL(Alice, acted) = True"]

    result = verify_conditions_effects(
        conditions, effects, state, "Alice", "TestAction"
    )

    assert isinstance(result, VerificationResult)

    print("✓ verify_conditions_effects function works")


# ============================================================
# Error Feedback Tests
# ============================================================

def test_format_errors_for_llm():
    """Test error formatting for LLM."""
    errors = [
        VerificationError(
            error_type="MISSING_VALUE",
            message="Condition missing value",
            severity=ErrorSeverity.ERROR,
            agent="Alice",
            action="Test",
            element="BEL(Alice, x)",
            suggestion="Add = True",
        ),
    ]

    formatted = format_errors_for_llm(errors)

    assert "MISSING_VALUE" in formatted
    assert "Condition missing value" in formatted
    assert "Add = True" in formatted

    print("✓ format_errors_for_llm works")


def test_format_errors_empty():
    """Test error formatting with empty list."""
    formatted = format_errors_for_llm([])
    assert formatted == ""

    print("✓ format_errors_for_llm handles empty list")


def test_format_knowledge_base_context():
    """Test knowledge base context formatting."""
    context = format_knowledge_base_context(
        "Alice",
        ["BEL(Alice, happy) = True", "DES(Alice, help) = True"],
        ["Alice", "Bob"],
    )

    assert "Alice" in context
    assert "Bob" in context
    assert "BEL(Alice, happy)" in context
    assert "DES(Alice, help)" in context

    print("✓ format_knowledge_base_context works")


def test_regeneration_request():
    """Test regeneration request creation."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")

    result = VerificationResult(valid=False, errors=[
        VerificationError(
            error_type="TEST_ERROR",
            message="Test error",
            agent="Alice",
            action="TestAction",
        ),
    ])

    request = create_regeneration_request(
        action_name="TestAction",
        agent_name="Alice",
        conditions=["BEL(Alice, x)"],
        effects=["BEL(Alice, y) = True"],
        result=result,
        state=state,
    )

    assert request.action_name == "TestAction"
    assert request.agent_name == "Alice"
    assert len(request.errors) >= 1

    print("✓ create_regeneration_request works")


def test_regeneration_prompt():
    """Test regeneration prompt generation."""
    request = RegenerationRequest(
        action_name="TestAction",
        agent_name="Alice",
        original_conditions=["BEL(Alice, x)"],
        original_effects=["BEL(Alice, y) = True"],
        errors=[
            VerificationError(
                error_type="MISSING_VALUE",
                message="Missing value",
                suggestion="Add = True",
            ),
        ],
        context="Agent: Alice\nBeliefs: BEL(Alice, exists) = True",
    )

    prompt = request.to_prompt()

    assert "TestAction" in prompt
    assert "Alice" in prompt
    assert "MISSING_VALUE" in prompt
    assert "BEL(Alice, x)" in prompt

    print("✓ RegenerationRequest.to_prompt() works")


def test_format_fix_summary():
    """Test fix summary formatting."""
    # All fixed
    summary = format_fix_summary(5, 0, 1, 3)
    assert "✅" in summary
    assert "5" in summary

    # Partial fix
    summary = format_fix_summary(5, 2, 2, 3)
    assert "⚠️" in summary or "Fixed" in summary

    # No improvement
    summary = format_fix_summary(5, 5, 3, 3)
    assert "❌" in summary or "No improvement" in summary

    print("✓ format_fix_summary works")


# ============================================================
# VerificationResult Tests
# ============================================================

def test_verification_result_merge():
    """Test merging verification results."""
    result1 = VerificationResult(valid=True)
    result1.add_error(VerificationError("E1", "Error 1", ErrorSeverity.ERROR))

    result2 = VerificationResult(valid=True)
    result2.add_error(VerificationError("W1", "Warning 1", ErrorSeverity.WARNING))

    result1.merge(result2)

    assert len(result1.errors) == 1
    assert len(result1.warnings) == 1
    assert not result1.valid  # Should be invalid due to error

    print("✓ VerificationResult.merge() works")


def test_verification_result_properties():
    """Test VerificationResult properties."""
    result = VerificationResult(valid=True)

    assert not result.has_errors
    assert not result.has_warnings

    result.add_error(VerificationError("W", "Warning", ErrorSeverity.WARNING))
    assert not result.has_errors
    assert result.has_warnings

    result.add_error(VerificationError("E", "Error", ErrorSeverity.ERROR))
    assert result.has_errors
    assert not result.valid

    print("✓ VerificationResult properties work")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-007: Verification Loop Tests")
    print("=" * 60)

    # Parser tests
    test_parser_with_value()
    test_parser_without_value()
    test_parser_intent()
    test_parser_extract_key()
    test_parser_invalid()

    # Verifier tests
    test_verify_missing_value()
    test_verify_unknown_agent()
    test_verify_invalid_format()
    test_verify_valid_condition()
    test_verify_effect()
    test_verify_action()
    test_verify_scenario()
    test_verify_real_scenario()
    test_verify_conditions_effects_function()

    # Error feedback tests
    test_format_errors_for_llm()
    test_format_errors_empty()
    test_format_knowledge_base_context()
    test_regeneration_request()
    test_regeneration_prompt()
    test_format_fix_summary()

    # Result tests
    test_verification_result_merge()
    test_verification_result_properties()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
