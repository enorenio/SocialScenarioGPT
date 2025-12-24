"""
Tests for TASK-008: Symbolic Consistency Verification
Tests the reachability analysis system for checking intention completability.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scenario_state import ScenarioState, Agent, Action
from core.reachability import (
    KnowledgeState,
    ActionNode,
    ActionStatus,
    IntentionAnalysis,
    AgentAnalysis,
    ScenarioAnalysis,
    ReachabilityAnalyzer,
    analyze_scenario_reachability,
    print_analysis_report,
)
from core.verification import BeliefDesireParser


# ============================================================
# KnowledgeState Tests
# ============================================================

def test_knowledge_state_init():
    """Test KnowledgeState initialization."""
    state = KnowledgeState()

    assert state.beliefs == {}
    assert state.desires == {}

    print("✓ KnowledgeState initializes correctly")


def test_knowledge_state_set_get():
    """Test setting and getting beliefs/desires."""
    state = KnowledgeState()

    state.set("BEL(Alice, happy)", "True")
    state.set("DES(Alice, help)", "True")

    assert state.has("BEL(Alice, happy)", "True")
    assert state.has("DES(Alice, help)", "True")
    assert not state.has("BEL(Alice, happy)", "False")
    assert not state.has("BEL(Alice, sad)", "True")

    print("✓ KnowledgeState set/get works")


def test_knowledge_state_copy():
    """Test copying knowledge state."""
    state = KnowledgeState()
    state.set("BEL(Alice, happy)", "True")

    copy = state.copy()
    copy.set("BEL(Alice, sad)", "True")

    # Original should be unchanged
    assert "BEL(Alice, sad)" not in state.beliefs
    assert "BEL(Alice, sad)" in copy.beliefs

    print("✓ KnowledgeState copy works")


def test_knowledge_state_from_knowledge_base():
    """Test creating state from knowledge base."""
    kb = [
        "BEL(Alice, happy) = True",
        "DES(Alice, help_Bob) = True",
        "BEL(Alice, location) = home",
    ]

    state = KnowledgeState.from_knowledge_base(kb)

    assert len(state.beliefs) == 2
    assert len(state.desires) == 1
    assert state.beliefs["BEL(Alice, happy)"] == "True"
    assert state.beliefs["BEL(Alice, location)"] == "home"

    print("✓ KnowledgeState.from_knowledge_base() works")


def test_knowledge_state_check_condition():
    """Test condition checking."""
    kb = ["BEL(Alice, ready) = True"]
    state = KnowledgeState.from_knowledge_base(kb)
    parser = BeliefDesireParser()

    assert state.check_condition("BEL(Alice, ready) = True", parser)
    assert not state.check_condition("BEL(Alice, ready) = False", parser)
    assert not state.check_condition("BEL(Alice, notready) = True", parser)

    print("✓ KnowledgeState.check_condition() works")


def test_knowledge_state_apply_effect():
    """Test effect application."""
    state = KnowledgeState()
    parser = BeliefDesireParser()

    state.apply_effect("BEL(Alice, done) = True", parser)

    assert state.beliefs["BEL(Alice, done)"] == "True"

    print("✓ KnowledgeState.apply_effect() works")


# ============================================================
# ReachabilityAnalyzer Tests
# ============================================================

def test_check_action_executable_true():
    """Test action executable when conditions met."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")
    state.add_action(
        "Alice",
        "DoSomething(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, done) = True"],
    )

    analyzer = ReachabilityAnalyzer(state)
    knowledge = KnowledgeState.from_knowledge_base(state.agents["Alice"].knowledge_base)
    action = state.agents["Alice"].actions["DoSomething(Alice)"]

    is_exec, blocking = analyzer.check_action_executable(action, knowledge)

    assert is_exec
    assert blocking == []

    print("✓ check_action_executable returns True when conditions met")


def test_check_action_executable_false():
    """Test action not executable when conditions not met."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = False")
    state.add_action(
        "Alice",
        "DoSomething(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, done) = True"],
    )

    analyzer = ReachabilityAnalyzer(state)
    knowledge = KnowledgeState.from_knowledge_base(state.agents["Alice"].knowledge_base)
    action = state.agents["Alice"].actions["DoSomething(Alice)"]

    is_exec, blocking = analyzer.check_action_executable(action, knowledge)

    assert not is_exec
    assert "BEL(Alice, ready) = True" in blocking

    print("✓ check_action_executable returns False when conditions not met")


def test_execute_action():
    """Test action execution updates state."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")
    state.add_action(
        "Alice",
        "DoSomething(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, done) = True"],
    )

    analyzer = ReachabilityAnalyzer(state)
    knowledge = KnowledgeState.from_knowledge_base(state.agents["Alice"].knowledge_base)
    action = state.agents["Alice"].actions["DoSomething(Alice)"]

    new_knowledge = analyzer.execute_action(action, knowledge)

    # Original should be unchanged
    assert "BEL(Alice, done)" not in knowledge.beliefs
    # New state should have effect
    assert new_knowledge.beliefs["BEL(Alice, done)"] == "True"

    print("✓ execute_action updates state correctly")


def test_analyze_intention_completable():
    """Test analyzing a completable intention."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")
    state.add_action(
        "Alice",
        "Step1(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, step1_done) = True"],
    )
    state.add_action(
        "Alice",
        "Step2(Alice)",
        conditions=["BEL(Alice, step1_done) = True"],
        effects=["BEL(Alice, all_done) = True"],
    )
    state.add_intention(
        "Alice",
        "INTENT(Alice, complete_task) = True",
        ["Step1(Alice)", "Step2(Alice)"],
    )

    analyzer = ReachabilityAnalyzer(state)
    agent = state.agents["Alice"]
    result = analyzer.analyze_intention(agent, "INTENT(Alice, complete_task) = True")

    assert result.completion_possible
    assert result.executable_actions == 2
    assert result.blocked_actions == 0

    print("✓ analyze_intention detects completable intentions")


def test_analyze_intention_blocked():
    """Test analyzing a blocked intention."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = False")  # Not ready
    state.add_action(
        "Alice",
        "Step1(Alice)",
        conditions=["BEL(Alice, ready) = True"],  # Will fail
        effects=["BEL(Alice, done) = True"],
    )
    state.add_intention(
        "Alice",
        "INTENT(Alice, do_thing) = True",
        ["Step1(Alice)"],
    )

    analyzer = ReachabilityAnalyzer(state)
    agent = state.agents["Alice"]
    result = analyzer.analyze_intention(agent, "INTENT(Alice, do_thing) = True")

    assert not result.completion_possible
    assert result.blocked_actions == 1
    assert len(result.blocking_conditions) > 0

    print("✓ analyze_intention detects blocked intentions")


def test_analyze_intention_not_found():
    """Test analyzing non-existent intention."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")

    analyzer = ReachabilityAnalyzer(state)
    agent = state.agents["Alice"]
    result = analyzer.analyze_intention(agent, "INTENT(Alice, nonexistent) = True")

    assert not result.completion_possible
    assert "Intention not found" in result.blocking_conditions

    print("✓ analyze_intention handles missing intentions")


def test_analyze_agent():
    """Test analyzing all intentions for an agent."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_belief("Alice", "BEL(Alice, ready) = True")
    state.add_action(
        "Alice",
        "DoThing(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, done) = True"],
    )
    state.add_intention(
        "Alice",
        "INTENT(Alice, first) = True",
        ["DoThing(Alice)"],
    )
    state.add_intention(
        "Alice",
        "INTENT(Alice, second) = True",
        ["DoThing(Alice)"],
    )

    analyzer = ReachabilityAnalyzer(state)
    result = analyzer.analyze_agent("Alice")

    assert result.total_intentions == 2
    assert result.completable_intentions == 2
    assert result.intention_completion_rate == 1.0

    print("✓ analyze_agent works correctly")


def test_analyze_agent_not_found():
    """Test analyzing non-existent agent."""
    state = ScenarioState("test", "Test")

    analyzer = ReachabilityAnalyzer(state)
    result = analyzer.analyze_agent("NonExistent")

    assert result.total_intentions == 0
    assert result.completable_intentions == 0

    print("✓ analyze_agent handles missing agents")


def test_analyze_scenario():
    """Test analyzing entire scenario."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")
    state.add_agent("Bob")
    state.add_belief("Alice", "BEL(Alice, ready) = True")
    state.add_action(
        "Alice",
        "DoThing(Alice)",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, done) = True"],
    )
    state.add_intention(
        "Alice",
        "INTENT(Alice, task) = True",
        ["DoThing(Alice)"],
    )

    analyzer = ReachabilityAnalyzer(state)
    result = analyzer.analyze_scenario()

    assert result.scenario_name == "test"
    assert len(result.agents) == 2
    assert result.total_intentions == 1
    assert result.completable_intentions == 1

    print("✓ analyze_scenario works correctly")


def test_convenience_function():
    """Test the analyze_scenario_reachability convenience function."""
    state = ScenarioState("test", "Test")
    state.add_agent("Alice")

    result = analyze_scenario_reachability(state)

    assert isinstance(result, ScenarioAnalysis)
    assert result.scenario_name == "test"

    print("✓ analyze_scenario_reachability() works")


# ============================================================
# Property Tests
# ============================================================

def test_intention_analysis_completion_rate():
    """Test completion rate calculation."""
    analysis = IntentionAnalysis(
        intention_name="test",
        action_plan=["a", "b", "c"],
        total_actions=3,
        executable_actions=2,
        blocked_actions=1,
        completion_possible=False,
        blocking_conditions=[],
        execution_trace=[],
    )

    assert analysis.completion_rate == 2/3

    # Test zero actions
    analysis2 = IntentionAnalysis(
        intention_name="empty",
        action_plan=[],
        total_actions=0,
        executable_actions=0,
        blocked_actions=0,
        completion_possible=False,
        blocking_conditions=[],
        execution_trace=[],
    )

    assert analysis2.completion_rate == 0.0

    print("✓ IntentionAnalysis.completion_rate works")


def test_agent_analysis_rates():
    """Test agent analysis rate calculations."""
    analysis = AgentAnalysis(
        agent_name="test",
        intentions=[],
        total_intentions=4,
        completable_intentions=2,
        immediately_executable_actions=3,
        total_actions=10,
    )

    assert analysis.intention_completion_rate == 0.5
    assert analysis.action_executability_rate == 0.3

    # Test zero totals
    empty = AgentAnalysis(
        agent_name="empty",
        intentions=[],
        total_intentions=0,
        completable_intentions=0,
        immediately_executable_actions=0,
        total_actions=0,
    )

    assert empty.intention_completion_rate == 0.0
    assert empty.action_executability_rate == 0.0

    print("✓ AgentAnalysis rate properties work")


def test_scenario_analysis_rates():
    """Test scenario analysis rate calculations."""
    analysis = ScenarioAnalysis(
        scenario_name="test",
        agents=[],
        total_intentions=10,
        completable_intentions=3,
        total_actions=20,
        immediately_executable_actions=8,
    )

    assert analysis.intention_completion_rate == 0.3
    assert analysis.action_executability_rate == 0.4

    print("✓ ScenarioAnalysis rate properties work")


# ============================================================
# Real Scenario Tests
# ============================================================

def test_real_scenario_reachability():
    """Test reachability analysis on real scenario."""
    test_file = Path("Data/test_Brother.json")
    if not test_file.exists():
        print("⚠ Skipping real scenario test: file not found")
        return

    state = ScenarioState.from_file(str(test_file))
    analysis = analyze_scenario_reachability(state)

    # Based on baseline, we expect poor reachability
    assert analysis.total_intentions > 0
    assert analysis.total_actions > 0

    # Baseline has ~0% intention completion, ~37% action executability
    assert analysis.intention_completion_rate < 0.5

    print(f"✓ Real scenario analysis: {analysis.completable_intentions}/{analysis.total_intentions} intentions completable")
    print(f"  Action executability: {analysis.action_executability_rate*100:.1f}%")


def test_chained_action_effects():
    """Test that chained actions properly propagate effects."""
    state = ScenarioState("test", "Test chaining")
    state.add_agent("Alice")

    # Chain: ready -> step1 -> step2 -> done
    state.add_belief("Alice", "BEL(Alice, ready) = True")

    state.add_action(
        "Alice", "Step1()",
        conditions=["BEL(Alice, ready) = True"],
        effects=["BEL(Alice, step1_complete) = True"],
    )
    state.add_action(
        "Alice", "Step2()",
        conditions=["BEL(Alice, step1_complete) = True"],
        effects=["BEL(Alice, step2_complete) = True"],
    )
    state.add_action(
        "Alice", "Step3()",
        conditions=["BEL(Alice, step2_complete) = True"],
        effects=["BEL(Alice, all_done) = True"],
    )

    state.add_intention(
        "Alice",
        "INTENT(Alice, complete_chain) = True",
        ["Step1()", "Step2()", "Step3()"],
    )

    analyzer = ReachabilityAnalyzer(state)
    result = analyzer.analyze_intention(
        state.agents["Alice"],
        "INTENT(Alice, complete_chain) = True"
    )

    assert result.completion_possible
    assert result.executable_actions == 3
    assert result.blocked_actions == 0

    # Check execution trace
    assert "✓ Step1()" in result.execution_trace[0]
    assert "✓ Step2()" in result.execution_trace[1]
    assert "✓ Step3()" in result.execution_trace[2]

    print("✓ Chained action effects propagate correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-008: Symbolic Consistency Verification Tests")
    print("=" * 60)

    # KnowledgeState tests
    test_knowledge_state_init()
    test_knowledge_state_set_get()
    test_knowledge_state_copy()
    test_knowledge_state_from_knowledge_base()
    test_knowledge_state_check_condition()
    test_knowledge_state_apply_effect()

    # ReachabilityAnalyzer tests
    test_check_action_executable_true()
    test_check_action_executable_false()
    test_execute_action()
    test_analyze_intention_completable()
    test_analyze_intention_blocked()
    test_analyze_intention_not_found()
    test_analyze_agent()
    test_analyze_agent_not_found()
    test_analyze_scenario()
    test_convenience_function()

    # Property tests
    test_intention_analysis_completion_rate()
    test_agent_analysis_rates()
    test_scenario_analysis_rates()

    # Real scenario tests
    test_real_scenario_reachability()
    test_chained_action_effects()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
