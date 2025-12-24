"""
Tests for TASK-006: Full Context State Management
Tests the ScenarioState class for managing scenario elements.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenario_state import (
    ScenarioState,
    Agent,
    Action,
    SpeakAction,
    Intention,
    ValidationError,
)


def test_scenario_state_init():
    """Test ScenarioState initialization."""
    state = ScenarioState("test_scenario", "A test description.")

    assert state.scenario_name == "test_scenario"
    assert state.scenario_description == "A test description."
    assert state.last_ended == "scenario"
    assert len(state.agents) == 0
    assert len(state.dialogue_tree) == 0

    print("✓ ScenarioState initializes correctly")


def test_add_agent():
    """Test adding agents."""
    state = ScenarioState("test", "Test")

    agent = state.add_agent("Alice")
    assert isinstance(agent, Agent)
    assert agent.name == "Alice"
    assert "Alice" in state.agents

    # Adding same agent returns existing
    agent2 = state.add_agent("Alice")
    assert agent is agent2
    assert len(state.agents) == 1

    # Add different agent
    state.add_agent("Bob")
    assert len(state.agents) == 2

    print("✓ add_agent() works correctly")


def test_add_beliefs_desires():
    """Test adding beliefs and desires."""
    state = ScenarioState("test", "Test")

    state.add_belief("Alice", "BEL(Alice, happy) = True")
    state.add_desire("Alice", "DES(Alice, help_Bob) = True")

    agent = state.get_agent("Alice")
    assert len(agent.knowledge_base) == 2
    assert "BEL(Alice, happy) = True" in agent.knowledge_base
    assert "DES(Alice, help_Bob) = True" in agent.knowledge_base

    # Test get_beliefs/get_desires
    assert len(agent.get_beliefs()) == 1
    assert len(agent.get_desires()) == 1

    print("✓ add_belief() and add_desire() work correctly")


def test_add_intention():
    """Test adding intentions."""
    state = ScenarioState("test", "Test")

    state.add_intention("Alice", "INTENT(Alice, help_Bob) = True", [
        "FindBob(Alice)",
        "OfferHelp(Alice, Bob)",
    ])

    agent = state.get_agent("Alice")
    assert "INTENT(Alice, help_Bob) = True" in agent.intentions

    intent = agent.intentions["INTENT(Alice, help_Bob) = True"]
    assert len(intent.action_plan) == 2
    assert intent.action_plan[0] == "FindBob(Alice)"

    print("✓ add_intention() works correctly")


def test_add_action():
    """Test adding actions."""
    state = ScenarioState("test", "Test")

    state.add_action(
        "Alice",
        "OfferHelp(Alice, Bob)",
        conditions=["BEL(Alice, friends_with_Bob) = True"],
        effects=["BEL(Bob, helped) = True"],
        occ_emotion=["Joy"],
        emotion_condition=["Hope"],
    )

    agent = state.get_agent("Alice")
    assert "OfferHelp(Alice, Bob)" in agent.actions

    action = agent.actions["OfferHelp(Alice, Bob)"]
    assert action.conditions == ["BEL(Alice, friends_with_Bob) = True"]
    assert action.effects == ["BEL(Bob, helped) = True"]
    assert action.occ_emotion == ["Joy"]

    print("✓ add_action() works correctly")


def test_add_speak_action():
    """Test adding speak actions."""
    state = ScenarioState("test", "Test")

    state.add_speak_action(
        "Alice",
        "Speak(Start, Greeting, None, None)",
        conditions=["BEL(Alice, sees_Bob) = True"],
        effects=["BEL(Bob, greeted) = True"],
    )

    agent = state.get_agent("Alice")
    assert "Speak(Start, Greeting, None, None)" in agent.speak_actions

    print("✓ add_speak_action() works correctly")


def test_dialogue_tree():
    """Test dialogue tree management."""
    state = ScenarioState("test", "Test")

    state.add_dialogue_line("<Start, Greeting, None, None, 'Hello!'>")
    state.add_dialogue_line("<Greeting, Response, None, None, 'Hi there!'>")

    assert len(state.dialogue_tree) == 2

    print("✓ Dialogue tree management works")


def test_to_dict():
    """Test dictionary serialization."""
    state = ScenarioState("test", "Test description")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, happy) = True")

    d = state.to_dict()

    assert d["scenario_name"] == "test"
    assert d["scenario_description"] == "Test description"
    assert "Alice" in d["agents"]
    assert "knowledge_base" in d["agents"]["Alice"]

    print("✓ to_dict() works correctly")


def test_to_json():
    """Test JSON serialization."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")

    json_str = state.to_json()
    parsed = json.loads(json_str)

    assert parsed["scenario_name"] == "test"
    assert "Alice" in parsed["agents"]

    print("✓ to_json() works correctly")


def test_from_dict():
    """Test loading from dictionary."""
    data = {
        "scenario_name": "loaded_test",
        "scenario_description": "Loaded description",
        "last_ended": "end",
        "agents": {
            "Bob": {
                "knowledge_base": ["BEL(Bob, exists) = True"],
                "intentions": {},
                "actions": {},
                "speak_actions": {},
                "initial_occ_emotion": ["Hope"],
            }
        },
        "dialogue_tree": ["<line1>", "<line2>"],
    }

    state = ScenarioState.from_dict(data)

    assert state.scenario_name == "loaded_test"
    assert state.last_ended == "end"
    assert "Bob" in state.agents
    assert len(state.dialogue_tree) == 2

    print("✓ from_dict() works correctly")


def test_from_file():
    """Test loading from JSON file."""
    test_file = Path("Data/test_Brother.json")
    if not test_file.exists():
        print("⚠ Skipping from_file test: test file not found")
        return

    state = ScenarioState.from_file(str(test_file))

    assert state.scenario_name == "test_Brother"
    assert len(state.agents) > 0
    assert state.last_ended == "end"

    print(f"✓ from_file() works: loaded {len(state.agents)} agents")


def test_round_trip():
    """Test dict -> state -> dict round trip."""
    test_file = Path("Data/test_Brother.json")
    if not test_file.exists():
        print("⚠ Skipping round-trip test: test file not found")
        return

    state = ScenarioState.from_file(str(test_file))
    original_dict = state.to_dict()

    # Reconstruct
    reconstructed = ScenarioState.from_dict(original_dict)
    reconstructed_dict = reconstructed.to_dict()

    # Compare
    orig_json = json.dumps(original_dict, sort_keys=True)
    recon_json = json.dumps(reconstructed_dict, sort_keys=True)

    assert orig_json == recon_json, "Round-trip failed: dicts differ"

    print("✓ Round-trip serialization works")


def test_to_prompt_context():
    """Test prompt context generation."""
    state = ScenarioState("test", "A test scenario.")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, happy) = True")
    state.add_intention("Alice", "INTENT(Alice, greet) = True", ["SayHello(Alice)"])

    context = state.to_prompt_context()

    assert "=== CURRENT SCENARIO STATE ===" in context
    assert "Alice" in context
    assert "BEL(Alice, happy)" in context
    assert "INTENT(Alice, greet)" in context

    print(f"✓ to_prompt_context() works, length: {len(context)} chars")


def test_to_compact_context():
    """Test compact context generation."""
    state = ScenarioState("test", "A test scenario.")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, happy) = True")

    compact = state.to_compact_context()

    assert "[Scenario: test]" in compact
    assert "[Agent:Alice]" in compact
    assert len(compact) < 500  # Should be compact

    print(f"✓ to_compact_context() works, length: {len(compact)} chars")


def test_get_stats():
    """Test statistics calculation."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_agent("Bob")
    state.add_belief("Alice", "BEL(Alice, happy) = True")
    state.add_desire("Alice", "DES(Alice, help) = True")
    state.add_intention("Alice", "INTENT(Alice, help) = True", ["Help()"])
    state.add_action("Alice", "Help()", ["cond"], ["effect"])
    state.add_dialogue_line("<line1>")

    stats = state.get_stats()

    assert stats["agents"] == 2
    assert stats["beliefs"] == 1
    assert stats["desires"] == 1
    assert stats["intentions"] == 1
    assert stats["actions"] == 1
    assert stats["dialogue_lines"] == 1

    print(f"✓ get_stats() works: {stats}")


def test_validate_empty_action_plan():
    """Test validation catches empty action plans."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")

    # Add intention with empty action plan
    agent = state.get_agent("Alice")
    agent.intentions["INTENT(Alice, something) = True"] = Intention(
        name="INTENT(Alice, something) = True",
        action_plan=[],
    )

    errors = state.validate()

    assert len(errors) == 1
    assert errors[0].error_type == "EMPTY_ACTION_PLAN"
    assert errors[0].agent == "Alice"

    print("✓ validate() catches empty action plans")


def test_set_stage():
    """Test stage setting."""
    state = ScenarioState("test", "Test")

    assert state.last_ended == "scenario"

    state.set_stage("agents")
    assert state.last_ended == "agents"

    state.set_stage("end")
    assert state.last_ended == "end"

    print("✓ set_stage() works")


def test_get_all_beliefs_desires():
    """Test getting all beliefs/desires across agents."""
    state = ScenarioState("test", "Test")
    state.add_belief("Alice", "BEL(Alice, happy) = True")
    state.add_belief("Bob", "BEL(Bob, sad) = True")
    state.add_desire("Alice", "DES(Alice, help) = True")

    beliefs = state.get_all_beliefs()
    desires = state.get_all_desires()

    assert "Alice" in beliefs
    assert "Bob" in beliefs
    assert len(beliefs["Alice"]) == 1
    assert len(desires["Alice"]) == 1
    assert len(desires["Bob"]) == 0

    print("✓ get_all_beliefs() and get_all_desires() work")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-006: Full Context State Management Tests")
    print("=" * 60)

    test_scenario_state_init()
    test_add_agent()
    test_add_beliefs_desires()
    test_add_intention()
    test_add_action()
    test_add_speak_action()
    test_dialogue_tree()
    test_to_dict()
    test_to_json()
    test_from_dict()
    test_from_file()
    test_round_trip()
    test_to_prompt_context()
    test_to_compact_context()
    test_get_stats()
    test_validate_empty_action_plan()
    test_set_stage()
    test_get_all_beliefs_desires()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
