"""
Tests for TASK-010: Dialogue Generation Improvement
Tests the dialogue prompts, personality system, and analyzer.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.dialogue import (
    DIALOGUE_PROMPTS,
    CharacterPersonality,
    DialogueContext,
    PersonalityTrait,
    DialogueStyle,
    generate_personality_prompt,
    generate_dialogue_context,
    DialogueLine,
    DialogueState,
    DialogueGraph,
    DialogueMetrics,
    analyze_dialogue,
    analyze_scenario_dialogue,
    print_dialogue_report,
    compare_dialogue_metrics,
)


# ============================================================
# PersonalityTrait and DialogueStyle Tests
# ============================================================

def test_personality_trait_enum():
    """Test PersonalityTrait enum has Big Five traits."""
    assert PersonalityTrait.OPENNESS.value == "openness"
    assert PersonalityTrait.CONSCIENTIOUSNESS.value == "conscientiousness"
    assert PersonalityTrait.EXTRAVERSION.value == "extraversion"
    assert PersonalityTrait.AGREEABLENESS.value == "agreeableness"
    assert PersonalityTrait.NEUROTICISM.value == "neuroticism"

    print("✓ PersonalityTrait enum has all Big Five traits")


def test_dialogue_style_enum():
    """Test DialogueStyle enum has expected styles."""
    styles = [s.value for s in DialogueStyle]

    assert "formal" in styles
    assert "casual" in styles
    assert "emotional" in styles
    assert "professional" in styles

    print(f"✓ DialogueStyle enum has {len(styles)} styles")


# ============================================================
# CharacterPersonality Tests
# ============================================================

def test_character_personality_init():
    """Test CharacterPersonality initialization."""
    personality = CharacterPersonality(
        name="Alice",
        role="protagonist",
        traits={"openness": 0.8, "extraversion": -0.5},
        speech_patterns=["uses formal language"],
        emotional_tendencies=["expresses worry"],
    )

    assert personality.name == "Alice"
    assert personality.role == "protagonist"
    assert personality.traits["openness"] == 0.8
    assert len(personality.speech_patterns) == 1

    print("✓ CharacterPersonality initializes correctly")


def test_character_personality_to_prompt_context():
    """Test generating prompt context from personality."""
    personality = CharacterPersonality(
        name="Bob",
        role="helper",
        traits={"agreeableness": 0.7, "neuroticism": -0.6},
        speech_patterns=["speaks briefly", "uses simple words"],
        emotional_tendencies=["stays calm"],
    )

    context = personality.to_prompt_context()

    assert "Bob" in context
    assert "helper" in context
    assert "high agreeableness" in context
    assert "low neuroticism" in context
    assert "speaks briefly" in context

    print("✓ CharacterPersonality.to_prompt_context() works")


def test_character_personality_empty_traits():
    """Test personality with no traits."""
    personality = CharacterPersonality(name="Empty")
    context = personality.to_prompt_context()

    assert "Empty" in context
    assert "Personality:" not in context  # No traits to show

    print("✓ CharacterPersonality handles empty traits")


# ============================================================
# DialogueContext Tests
# ============================================================

def test_dialogue_context_init():
    """Test DialogueContext initialization."""
    ctx = DialogueContext(
        scenario_description="A meeting at work",
        agents=["Alice", "Bob"],
        current_emotions={"Alice": "nervous", "Bob": "calm"},
        conversation_goals=["discuss project", "resolve conflict"],
    )

    assert ctx.scenario_description == "A meeting at work"
    assert len(ctx.agents) == 2
    assert ctx.current_emotions["Alice"] == "nervous"

    print("✓ DialogueContext initializes correctly")


def test_dialogue_context_to_prompt_context():
    """Test generating prompt context."""
    personality = CharacterPersonality(
        name="Alice",
        role="protagonist",
        traits={"extraversion": 0.5},
    )

    ctx = DialogueContext(
        scenario_description="Coffee shop conversation",
        agents=["Alice", "Bob"],
        personalities={"Alice": personality},
        current_emotions={"Alice": "happy"},
        conversation_goals=["catch up", "share news"],
    )

    prompt_ctx = ctx.to_prompt_context()

    assert "Coffee shop" in prompt_ctx
    assert "Alice" in prompt_ctx
    assert "Bob" in prompt_ctx
    assert "catch up" in prompt_ctx
    assert "happy" in prompt_ctx

    print("✓ DialogueContext.to_prompt_context() works")


def test_generate_dialogue_context():
    """Test the helper function."""
    ctx = generate_dialogue_context(
        scenario="Test scenario",
        agents=["Agent1", "Agent2"],
        emotions={"Agent1": "happy"},
    )

    assert ctx.scenario_description == "Test scenario"
    assert "Agent1" in ctx.agents
    assert ctx.current_emotions["Agent1"] == "happy"

    print("✓ generate_dialogue_context() works")


# ============================================================
# DialogueLine Tests
# ============================================================

def test_dialogue_line_parse_angle_brackets():
    """Test parsing dialogue line with angle brackets."""
    line = '<Start, Greeting, hello, Friendly, "Hi there!">'
    parsed = DialogueLine.parse(line)

    assert parsed is not None
    assert parsed.current_state == "Start"
    assert parsed.next_state == "Greeting"
    assert parsed.meaning == "hello"
    assert parsed.style == "Friendly"
    assert parsed.utterance == "Hi there!"

    print("✓ DialogueLine.parse() handles angle brackets")


def test_dialogue_line_parse_double_brackets():
    """Test parsing dialogue line with double brackets."""
    line = '[[Greeting, Question, ask, Casual, "How are you?"]]'
    parsed = DialogueLine.parse(line)

    assert parsed is not None
    assert parsed.current_state == "Greeting"
    assert parsed.next_state == "Question"
    assert parsed.utterance == "How are you?"

    print("✓ DialogueLine.parse() handles double brackets")


def test_dialogue_line_parse_invalid():
    """Test parsing invalid dialogue line."""
    line = "This is not a valid dialogue line"
    parsed = DialogueLine.parse(line)

    assert parsed is None

    print("✓ DialogueLine.parse() returns None for invalid input")


def test_dialogue_line_parse_with_commas_in_utterance():
    """Test parsing line where utterance contains commas."""
    line = '<State1, State2, meaning, style, "Hello, how are you, friend?">'
    parsed = DialogueLine.parse(line)

    assert parsed is not None
    # The utterance should be complete
    assert "Hello" in parsed.utterance

    print("✓ DialogueLine.parse() handles commas in utterance")


# ============================================================
# DialogueState Tests
# ============================================================

def test_dialogue_state_is_start():
    """Test detecting start states."""
    start = DialogueState(name="Start")
    initial = DialogueState(name="Initial")
    middle = DialogueState(name="Middle")

    assert start.is_start is True
    assert initial.is_start is True
    assert middle.is_start is False

    print("✓ DialogueState.is_start works")


def test_dialogue_state_is_end():
    """Test detecting end states."""
    end = DialogueState(name="End")
    final = DialogueState(name="Final")
    middle = DialogueState(name="Middle")

    assert end.is_end is True
    assert final.is_end is True
    assert middle.is_end is False

    print("✓ DialogueState.is_end works")


def test_dialogue_state_branch_count():
    """Test counting outgoing branches."""
    state = DialogueState(name="Test")

    # Add some mock outgoing lines
    state.outgoing.append(DialogueLine("Test", "A", "m", "s", "u1"))
    state.outgoing.append(DialogueLine("Test", "B", "m", "s", "u2"))
    state.outgoing.append(DialogueLine("Test", "C", "m", "s", "u3"))

    assert state.branch_count == 3

    print("✓ DialogueState.branch_count works")


# ============================================================
# DialogueGraph Tests
# ============================================================

def test_dialogue_graph_from_dialogue_tree():
    """Test building graph from dialogue tree."""
    tree = [
        '<Start, Middle, greet, Friendly, "Hello!">',
        '<Middle, End, farewell, Casual, "Goodbye!">',
    ]

    graph = DialogueGraph.from_dialogue_tree(tree)

    assert len(graph.lines) == 2
    assert len(graph.states) == 3  # Start, Middle, End
    assert "Start" in graph.states
    assert "Middle" in graph.states
    assert "End" in graph.states

    print("✓ DialogueGraph.from_dialogue_tree() works")


def test_dialogue_graph_get_start_states():
    """Test getting start states."""
    tree = [
        '<Start, A, m, s, "u1">',
        '<A, End, m, s, "u2">',
    ]

    graph = DialogueGraph.from_dialogue_tree(tree)
    starts = graph.get_start_states()

    # Start should be detected
    assert any(s.name == "Start" for s in starts)

    print("✓ DialogueGraph.get_start_states() works")


def test_dialogue_graph_get_end_states():
    """Test getting end states."""
    tree = [
        '<Start, A, m, s, "u1">',
        '<A, End, m, s, "u2">',
    ]

    graph = DialogueGraph.from_dialogue_tree(tree)
    ends = graph.get_end_states()

    # End should be detected
    assert any(s.name == "End" for s in ends)

    print("✓ DialogueGraph.get_end_states() works")


def test_dialogue_graph_get_branch_points():
    """Test detecting branch points."""
    tree = [
        '<Start, A, m, s, "u1">',
        '<Start, B, m, s, "u2">',  # Start has 2 outgoing = branch point
        '<A, End, m, s, "u3">',
        '<B, End, m, s, "u4">',
    ]

    graph = DialogueGraph.from_dialogue_tree(tree)
    branches = graph.get_branch_points()

    assert len(branches) == 1
    assert branches[0].name == "Start"

    print("✓ DialogueGraph.get_branch_points() works")


def test_dialogue_graph_count_paths():
    """Test counting paths through dialogue."""
    # Simple linear path
    tree_linear = [
        '<Start, A, m, s, "u1">',
        '<A, End, m, s, "u2">',
    ]
    graph_linear = DialogueGraph.from_dialogue_tree(tree_linear)
    assert graph_linear.count_paths() == 1

    # Two paths
    tree_branch = [
        '<Start, A, m, s, "u1">',
        '<Start, B, m, s, "u2">',
        '<A, End, m, s, "u3">',
        '<B, End, m, s, "u4">',
    ]
    graph_branch = DialogueGraph.from_dialogue_tree(tree_branch)
    assert graph_branch.count_paths() == 2

    print("✓ DialogueGraph.count_paths() works")


# ============================================================
# DialogueMetrics Tests
# ============================================================

def test_dialogue_metrics_init():
    """Test DialogueMetrics initialization."""
    metrics = DialogueMetrics()

    assert metrics.total_lines == 0
    assert metrics.unique_states == 0
    assert metrics.branch_points == 0
    assert metrics.has_start is False
    assert metrics.has_end is False

    print("✓ DialogueMetrics initializes with defaults")


def test_dialogue_metrics_to_dict():
    """Test converting metrics to dict."""
    metrics = DialogueMetrics(
        total_lines=10,
        unique_states=5,
        branch_points=2,
        max_branches=3,
        approximate_paths=4,
        styles_used={"Formal", "Casual"},
        meanings_used={"greet", "ask", "respond"},
        avg_utterance_length=45.5,
        has_start=True,
        has_end=True,
    )

    d = metrics.to_dict()

    assert d["total_lines"] == 10
    assert d["unique_states"] == 5
    assert d["has_start"] is True
    assert len(d["styles_used"]) == 2

    print("✓ DialogueMetrics.to_dict() works")


# ============================================================
# analyze_dialogue Tests
# ============================================================

def test_analyze_dialogue_basic():
    """Test basic dialogue analysis."""
    tree = [
        '<Start, A, greet, Formal, "Good morning.">',
        '<A, B, ask, Casual, "How are you?">',
        '<B, End, respond, Friendly, "I am fine, thanks!">',
    ]

    metrics = analyze_dialogue(tree)

    assert metrics.total_lines == 3
    assert metrics.unique_states == 4  # Start, A, B, End
    assert metrics.has_start is True
    assert metrics.has_end is True
    assert "Formal" in metrics.styles_used
    assert "Casual" in metrics.styles_used

    print("✓ analyze_dialogue() works for basic tree")


def test_analyze_dialogue_with_branches():
    """Test analyzing dialogue with branches."""
    tree = [
        '<Start, Ask, init, Formal, "Hello">',
        '<Ask, AgreeResponse, agree, Friendly, "Yes, sure!">',
        '<Ask, DisagreeResponse, disagree, Tense, "No, sorry.">',
        '<AgreeResponse, End, close, Casual, "Great!">',
        '<DisagreeResponse, End, close, Casual, "Okay then.">',
    ]

    metrics = analyze_dialogue(tree)

    assert metrics.total_lines == 5
    assert metrics.branch_points == 1  # Ask state has 2 outgoing
    assert metrics.approximate_paths == 2  # Two paths through dialogue

    print("✓ analyze_dialogue() correctly identifies branches")


def test_analyze_dialogue_empty():
    """Test analyzing empty dialogue."""
    metrics = analyze_dialogue([])

    assert metrics.total_lines == 0
    assert metrics.unique_states == 0

    print("✓ analyze_dialogue() handles empty input")


def test_analyze_scenario_dialogue():
    """Test analyzing from scenario dict."""
    scenario = {
        "dialogue_tree": [
            '<Start, End, hello, Casual, "Hi!">',
        ],
        "agents": {
            "Alice": {
                "speak_actions": {"hello": {}}
            },
            "Bob": {
                "speak_actions": {"respond": {}, "ask": {}}
            },
        },
    }

    metrics = analyze_scenario_dialogue(scenario)

    assert metrics.total_lines == 1
    assert "Alice" in metrics.speakers
    assert metrics.speakers["Alice"] == 1
    assert metrics.speakers["Bob"] == 2

    print("✓ analyze_scenario_dialogue() works")


# ============================================================
# compare_dialogue_metrics Tests
# ============================================================

def test_compare_dialogue_metrics():
    """Test comparing two dialogue metrics."""
    baseline = DialogueMetrics(
        total_lines=5,
        unique_states=4,
        branch_points=1,
        approximate_paths=2,
        styles_used={"Casual"},
    )

    improved = DialogueMetrics(
        total_lines=15,
        unique_states=10,
        branch_points=4,
        approximate_paths=8,
        styles_used={"Casual", "Formal", "Emotional"},
    )

    comparison = compare_dialogue_metrics(baseline, improved)

    assert comparison["lines_improvement"] == 10
    assert comparison["lines_ratio"] == 3.0
    assert comparison["states_improvement"] == 6
    assert comparison["branches_improvement"] == 3
    assert comparison["paths_improvement"] == 6
    assert comparison["variety_improvement"] == 2

    print("✓ compare_dialogue_metrics() works")


# ============================================================
# DIALOGUE_PROMPTS Registry Tests
# ============================================================

def test_dialogue_prompts_registry_exists():
    """Test that dialogue prompts registry has entries."""
    assert len(DIALOGUE_PROMPTS) > 0

    print(f"✓ DIALOGUE_PROMPTS has {len(DIALOGUE_PROMPTS)} prompts")


def test_dialogue_prompts_have_required_entries():
    """Test required prompts exist."""
    required = [
        "dialogue_tree_improved",
        "speak_actions_improved",
        "speak_conditions_effects_improved",
    ]

    for name in required:
        assert name in DIALOGUE_PROMPTS, f"Missing: {name}"
        assert "template" in DIALOGUE_PROMPTS[name]
        assert "description" in DIALOGUE_PROMPTS[name]

    print("✓ All required dialogue prompts exist")


def test_dialogue_prompts_have_branching_guidance():
    """Test that improved prompts include branching guidance."""
    template = DIALOGUE_PROMPTS["dialogue_tree_improved"]["template"]

    assert "branch" in template.lower()
    assert "path" in template.lower()

    print("✓ dialogue_tree_improved includes branching guidance")


def test_dialogue_prompts_have_line_targets():
    """Test that prompts specify target line counts."""
    template = DIALOGUE_PROMPTS["dialogue_tree_improved"]["template"]

    # Should mention target line counts
    assert "12" in template or "15" in template

    print("✓ dialogue_tree_improved specifies line targets")


# ============================================================
# generate_personality_prompt Tests
# ============================================================

def test_generate_personality_prompt():
    """Test personality prompt generation."""
    prompt = generate_personality_prompt("TestAgent", "A test scenario")

    assert "TestAgent" in prompt
    assert "test scenario" in prompt.lower()
    assert "Big Five" in prompt or "Openness" in prompt

    print("✓ generate_personality_prompt() works")


# ============================================================
# Integration Tests
# ============================================================

def test_full_workflow():
    """Test complete dialogue analysis workflow."""
    # 1. Create personality
    personality = CharacterPersonality(
        name="Alice",
        role="protagonist",
        traits={"openness": 0.6, "agreeableness": 0.8},
    )

    # 2. Create context
    context = DialogueContext(
        scenario_description="Alice asks Bob for help",
        agents=["Alice", "Bob"],
        personalities={"Alice": personality},
        conversation_goals=["get help", "explain problem"],
    )

    # 3. Analyze sample dialogue
    tree = [
        '<Start, AskHelp, request, Friendly, "Bob, could you help me?">',
        '<AskHelp, Explain, agree, Supportive, "Of course! What do you need?">',
        '<Explain, Clarify, describe, Casual, "I need help with this task.">',
        '<Clarify, Offer, question, Professional, "What specifically is unclear?">',
        '<Offer, Accept, clarify, Casual, "This part here.">',
        '<Accept, End, resolve, Friendly, "Ah, I see. Let me show you.">',
    ]

    metrics = analyze_dialogue(tree)

    # 4. Verify results
    assert metrics.total_lines == 6
    assert metrics.has_start is True
    assert metrics.has_end is True
    assert len(metrics.styles_used) >= 3

    print("✓ Full dialogue workflow works")


def test_baseline_improvement_tracking():
    """Test that we can track improvement over baseline."""
    # Baseline: original average of ~5-6 lines
    baseline_tree = [
        '<Start, A, m, s, "u1">',
        '<A, B, m, s, "u2">',
        '<B, C, m, s, "u3">',
        '<C, D, m, s, "u4">',
        '<D, End, m, s, "u5">',
    ]

    # Improved: target of 12-15 lines with branching
    improved_tree = [
        '<Start, Greet, init, Formal, "Hello">',
        '<Greet, Ask, greet, Friendly, "Hi there!">',
        '<Ask, YesPath, agree, Supportive, "Sure, I can help">',
        '<Ask, NoPath, disagree, Tense, "Sorry, I cannot">',
        '<Ask, MaybePath, uncertain, Casual, "Let me think...">',
        '<YesPath, Details, explain, Professional, "Here is what we do">',
        '<NoPath, Apologize, regret, Emotional, "I wish I could">',
        '<MaybePath, Consider, think, Casual, "Actually, maybe yes">',
        '<Details, Confirm, verify, Formal, "Does this make sense?">',
        '<Apologize, End, close, Friendly, "No worries, goodbye">',
        '<Consider, YesPath, decide, Supportive, "Yes, I will help">',
        '<Confirm, End, close, Casual, "Great, thanks!">',
    ]

    baseline = analyze_dialogue(baseline_tree)
    improved = analyze_dialogue(improved_tree)

    comparison = compare_dialogue_metrics(baseline, improved)

    # Improved should have more lines
    assert comparison["lines_improvement"] > 5
    # Improved should have more branches
    assert comparison["branches_improvement"] > 0
    # Improved should have more paths
    assert comparison["paths_improvement"] > 0

    print("✓ Baseline improvement tracking works")
    print(f"  Lines: {baseline.total_lines} -> {improved.total_lines} (+{comparison['lines_improvement']})")
    print(f"  Paths: {baseline.approximate_paths} -> {improved.approximate_paths} (+{comparison['paths_improvement']})")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-010: Dialogue Generation Improvement Tests")
    print("=" * 60)

    # Enum tests
    test_personality_trait_enum()
    test_dialogue_style_enum()

    # CharacterPersonality tests
    test_character_personality_init()
    test_character_personality_to_prompt_context()
    test_character_personality_empty_traits()

    # DialogueContext tests
    test_dialogue_context_init()
    test_dialogue_context_to_prompt_context()
    test_generate_dialogue_context()

    # DialogueLine tests
    test_dialogue_line_parse_angle_brackets()
    test_dialogue_line_parse_double_brackets()
    test_dialogue_line_parse_invalid()
    test_dialogue_line_parse_with_commas_in_utterance()

    # DialogueState tests
    test_dialogue_state_is_start()
    test_dialogue_state_is_end()
    test_dialogue_state_branch_count()

    # DialogueGraph tests
    test_dialogue_graph_from_dialogue_tree()
    test_dialogue_graph_get_start_states()
    test_dialogue_graph_get_end_states()
    test_dialogue_graph_get_branch_points()
    test_dialogue_graph_count_paths()

    # DialogueMetrics tests
    test_dialogue_metrics_init()
    test_dialogue_metrics_to_dict()

    # analyze_dialogue tests
    test_analyze_dialogue_basic()
    test_analyze_dialogue_with_branches()
    test_analyze_dialogue_empty()
    test_analyze_scenario_dialogue()

    # compare tests
    test_compare_dialogue_metrics()

    # Registry tests
    test_dialogue_prompts_registry_exists()
    test_dialogue_prompts_have_required_entries()
    test_dialogue_prompts_have_branching_guidance()
    test_dialogue_prompts_have_line_targets()

    # Prompt generation tests
    test_generate_personality_prompt()

    # Integration tests
    test_full_workflow()
    test_baseline_improvement_tracking()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
