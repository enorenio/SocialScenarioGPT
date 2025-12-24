"""
Tests for TASK-011: LLM-as-Judge Evaluation System
Tests the rubrics and LLM judge components (without making actual API calls).
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.rubrics import (
    EVALUATION_RUBRICS,
    EvaluationDimension,
    EvaluationRubric,
    RubricLevel,
    get_rubric,
    list_dimensions,
    get_all_rubrics_prompt,
    calculate_weighted_average,
)
from evaluation.llm_judge import (
    LLMJudge,
    EvaluationResult,
    DimensionScore,
    compare_evaluations,
)


# ============================================================
# Rubric Tests
# ============================================================

def test_evaluation_dimension_enum():
    """Test that all expected dimensions exist."""
    expected = [
        "agent_relevance",
        "belief_coherence",
        "desire_appropriateness",
        "intention_validity",
        "action_feasibility",
        "condition_effect_logic",
        "dialogue_quality",
        "dialogue_naturalness",
        "emotional_consistency",
        "overall_coherence",
    ]

    actual = [d.value for d in EvaluationDimension]

    for exp in expected:
        assert exp in actual, f"Missing dimension: {exp}"

    print(f"✓ EvaluationDimension has all {len(expected)} expected dimensions")


def test_rubric_levels_complete():
    """Test that each rubric has complete 1-5 scale."""
    for dim, rubric in EVALUATION_RUBRICS.items():
        scores = [level.score for level in rubric.levels]

        assert 1 in scores, f"{dim.value} missing score 1"
        assert 2 in scores, f"{dim.value} missing score 2"
        assert 3 in scores, f"{dim.value} missing score 3"
        assert 4 in scores, f"{dim.value} missing score 4"
        assert 5 in scores, f"{dim.value} missing score 5"

    print(f"✓ All {len(EVALUATION_RUBRICS)} rubrics have complete 1-5 scale")


def test_rubric_has_question():
    """Test that each rubric has an evaluation question."""
    for dim, rubric in EVALUATION_RUBRICS.items():
        assert rubric.question, f"{dim.value} missing question"
        assert "?" in rubric.question, f"{dim.value} question should end with ?"

    print("✓ All rubrics have evaluation questions")


def test_rubric_levels_have_descriptions():
    """Test that each level has a description."""
    for dim, rubric in EVALUATION_RUBRICS.items():
        for level in rubric.levels:
            assert level.label, f"{dim.value} level {level.score} missing label"
            assert level.description, f"{dim.value} level {level.score} missing description"

    print("✓ All rubric levels have labels and descriptions")


def test_get_rubric():
    """Test getting rubric by dimension."""
    rubric = get_rubric(EvaluationDimension.AGENT_RELEVANCE)

    assert rubric is not None
    assert rubric.name == "Agent Relevance"
    assert len(rubric.levels) == 5

    print("✓ get_rubric() works")


def test_list_dimensions():
    """Test listing all dimensions."""
    dims = list_dimensions()

    assert len(dims) == 10
    assert EvaluationDimension.AGENT_RELEVANCE in dims
    assert EvaluationDimension.OVERALL_COHERENCE in dims

    print(f"✓ list_dimensions() returns {len(dims)} dimensions")


def test_get_all_rubrics_prompt():
    """Test generating combined rubrics prompt."""
    prompt = get_all_rubrics_prompt()

    # Should contain all dimension names
    assert "Agent Relevance" in prompt
    assert "Belief Coherence" in prompt
    assert "Overall Coherence" in prompt

    # Should contain scoring guide
    assert "Scoring Guide" in prompt
    assert "1 -" in prompt
    assert "5 -" in prompt

    print(f"✓ get_all_rubrics_prompt() generates {len(prompt)} char prompt")


def test_calculate_weighted_average():
    """Test weighted average calculation."""
    scores = {
        EvaluationDimension.AGENT_RELEVANCE: 4,  # weight 1.0
        EvaluationDimension.BELIEF_COHERENCE: 3,  # weight 1.2
        EvaluationDimension.ACTION_FEASIBILITY: 5,  # weight 1.5
    }

    avg = calculate_weighted_average(scores)

    # Manual calculation: (4*1.0 + 3*1.2 + 5*1.5) / (1.0 + 1.2 + 1.5)
    # = (4 + 3.6 + 7.5) / 3.7 = 15.1 / 3.7 ≈ 4.08
    assert 4.0 <= avg <= 4.2

    print(f"✓ calculate_weighted_average() returns {avg:.2f}")


def test_calculate_weighted_average_empty():
    """Test weighted average with empty scores."""
    avg = calculate_weighted_average({})
    assert avg == 0.0

    print("✓ calculate_weighted_average() handles empty input")


def test_rubric_get_prompt_section():
    """Test generating prompt section from rubric."""
    rubric = get_rubric(EvaluationDimension.DIALOGUE_QUALITY)
    section = rubric.get_prompt_section()

    assert "Dialogue Quality" in section
    assert "Question:" in section
    assert "Scoring Guide:" in section
    assert "1 -" in section
    assert "5 -" in section

    print("✓ rubric.get_prompt_section() works")


# ============================================================
# DimensionScore Tests
# ============================================================

def test_dimension_score_init():
    """Test DimensionScore initialization."""
    score = DimensionScore(
        dimension=EvaluationDimension.AGENT_RELEVANCE,
        score=4,
        reasoning="Agents match scenario well",
        confidence=0.9,
    )

    assert score.dimension == EvaluationDimension.AGENT_RELEVANCE
    assert score.score == 4
    assert "Agents" in score.reasoning
    assert score.confidence == 0.9

    print("✓ DimensionScore initializes correctly")


# ============================================================
# EvaluationResult Tests
# ============================================================

def test_evaluation_result_init():
    """Test EvaluationResult initialization."""
    scores = {
        EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
            dimension=EvaluationDimension.AGENT_RELEVANCE,
            score=4,
            reasoning="Good agents",
        ),
        EvaluationDimension.BELIEF_COHERENCE: DimensionScore(
            dimension=EvaluationDimension.BELIEF_COHERENCE,
            score=3,
            reasoning="Average beliefs",
        ),
    }

    result = EvaluationResult(
        scenario_name="test_scenario",
        scores=scores,
        weighted_average=3.5,
        model_used="gpt-4o",
    )

    assert result.scenario_name == "test_scenario"
    assert len(result.scores) == 2
    assert result.weighted_average == 3.5

    print("✓ EvaluationResult initializes correctly")


def test_evaluation_result_to_dict():
    """Test converting result to dictionary."""
    scores = {
        EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
            dimension=EvaluationDimension.AGENT_RELEVANCE,
            score=4,
            reasoning="Good agents",
        ),
    }

    result = EvaluationResult(
        scenario_name="test",
        scores=scores,
        weighted_average=4.0,
        model_used="gpt-4o",
    )

    d = result.to_dict()

    assert d["scenario_name"] == "test"
    assert "agent_relevance" in d["scores"]
    assert d["scores"]["agent_relevance"]["score"] == 4
    assert d["weighted_average"] == 4.0

    print("✓ EvaluationResult.to_dict() works")


def test_evaluation_result_get_summary():
    """Test getting evaluation summary."""
    scores = {
        EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
            dimension=EvaluationDimension.AGENT_RELEVANCE,
            score=4,
            reasoning="Good",
        ),
        EvaluationDimension.BELIEF_COHERENCE: DimensionScore(
            dimension=EvaluationDimension.BELIEF_COHERENCE,
            score=3,
            reasoning="OK",
        ),
    }

    result = EvaluationResult(
        scenario_name="test_scenario",
        scores=scores,
        weighted_average=3.5,
    )

    summary = result.get_summary()

    assert "test_scenario" in summary
    assert "3.50" in summary
    assert "Agent Relevance" in summary

    print("✓ EvaluationResult.get_summary() works")


# ============================================================
# LLMJudge Tests (without API calls)
# ============================================================

def test_llm_judge_init():
    """Test LLMJudge initialization."""
    judge = LLMJudge(model_name="gpt-4o")

    assert judge.model_name == "gpt-4o"
    assert len(judge.dimensions) == 10  # All dimensions

    print("✓ LLMJudge initializes correctly")


def test_llm_judge_init_with_dimensions():
    """Test LLMJudge with specific dimensions."""
    dims = [
        EvaluationDimension.AGENT_RELEVANCE,
        EvaluationDimension.DIALOGUE_QUALITY,
    ]
    judge = LLMJudge(dimensions=dims)

    assert len(judge.dimensions) == 2
    assert EvaluationDimension.AGENT_RELEVANCE in judge.dimensions

    print("✓ LLMJudge accepts custom dimensions")


def test_llm_judge_format_scenario():
    """Test scenario formatting for evaluation."""
    judge = LLMJudge()

    scenario = {
        "scenario_name": "test",
        "scenario_description": "A test scenario",
        "agents": {
            "Alice": {
                "knowledge_base": ["BEL(Alice, happy) = True"],
                "intentions": {"help": {"action_plan": ["action1"]}},
                "actions": {
                    "action1": {
                        "conditions": ["cond1"],
                        "effects": ["effect1"],
                    }
                },
                "speak_actions": {"say_hi": {}},
            }
        },
        "dialogue_tree": ["line1", "line2"],
    }

    formatted = judge._format_scenario_for_evaluation(scenario)

    assert "SCENARIO DESCRIPTION" in formatted
    assert "test scenario" in formatted
    assert "AGENTS" in formatted
    assert "Alice" in formatted
    assert "DIALOGUE TREE" in formatted

    print("✓ LLMJudge._format_scenario_for_evaluation() works")


def test_llm_judge_build_evaluation_prompt():
    """Test evaluation prompt building."""
    judge = LLMJudge()

    scenario_text = "Test scenario content"
    dims = [EvaluationDimension.AGENT_RELEVANCE]

    prompt = judge._build_evaluation_prompt(scenario_text, dims)

    assert "expert evaluator" in prompt.lower()
    assert "Agent Relevance" in prompt
    assert "Test scenario content" in prompt
    assert "JSON" in prompt
    assert "1-5" in prompt

    print("✓ LLMJudge._build_evaluation_prompt() works")


def test_llm_judge_parse_json_response():
    """Test parsing JSON evaluation response."""
    judge = LLMJudge()

    response = '''
    Here is my evaluation:
    ```json
    {
      "evaluations": [
        {
          "dimension": "agent_relevance",
          "score": 4,
          "reasoning": "Agents match well"
        },
        {
          "dimension": "belief_coherence",
          "score": 3,
          "reasoning": "Some gaps"
        }
      ]
    }
    ```
    '''

    dims = [
        EvaluationDimension.AGENT_RELEVANCE,
        EvaluationDimension.BELIEF_COHERENCE,
    ]

    scores = judge._parse_evaluation_response(response, dims)

    assert EvaluationDimension.AGENT_RELEVANCE in scores
    assert scores[EvaluationDimension.AGENT_RELEVANCE].score == 4
    assert EvaluationDimension.BELIEF_COHERENCE in scores
    assert scores[EvaluationDimension.BELIEF_COHERENCE].score == 3

    print("✓ LLMJudge._parse_evaluation_response() parses JSON")


def test_llm_judge_parse_freeform_response():
    """Test parsing freeform evaluation response."""
    judge = LLMJudge()

    response = """
    My evaluation:
    Agent Relevance: 4/5 - Good match
    Belief Coherence: 3/5 - Some issues
    """

    dims = [
        EvaluationDimension.AGENT_RELEVANCE,
        EvaluationDimension.BELIEF_COHERENCE,
    ]

    scores = judge._parse_freeform_response(response, dims)

    assert EvaluationDimension.AGENT_RELEVANCE in scores
    assert scores[EvaluationDimension.AGENT_RELEVANCE].score == 4
    assert EvaluationDimension.BELIEF_COHERENCE in scores
    assert scores[EvaluationDimension.BELIEF_COHERENCE].score == 3

    print("✓ LLMJudge._parse_freeform_response() works")


def test_llm_judge_score_clamping():
    """Test that scores are clamped to 1-5 range."""
    judge = LLMJudge()

    # Response with out-of-range scores
    response = '''
    {
      "evaluations": [
        {"dimension": "agent_relevance", "score": 10, "reasoning": "Too high"},
        {"dimension": "belief_coherence", "score": 0, "reasoning": "Too low"}
      ]
    }
    '''

    dims = [
        EvaluationDimension.AGENT_RELEVANCE,
        EvaluationDimension.BELIEF_COHERENCE,
    ]

    scores = judge._parse_evaluation_response(response, dims)

    assert scores[EvaluationDimension.AGENT_RELEVANCE].score == 5  # Clamped to max
    assert scores[EvaluationDimension.BELIEF_COHERENCE].score == 1  # Clamped to min

    print("✓ LLMJudge clamps scores to 1-5 range")


# ============================================================
# Comparison Tests
# ============================================================

def test_compare_evaluations():
    """Test comparing baseline and improved evaluations."""
    baseline = [
        EvaluationResult(
            scenario_name="test1",
            scores={
                EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
                    EvaluationDimension.AGENT_RELEVANCE, 3, "OK"
                ),
            },
            weighted_average=3.0,
        ),
        EvaluationResult(
            scenario_name="test2",
            scores={
                EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
                    EvaluationDimension.AGENT_RELEVANCE, 3, "OK"
                ),
            },
            weighted_average=3.0,
        ),
    ]

    improved = [
        EvaluationResult(
            scenario_name="test1",
            scores={
                EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
                    EvaluationDimension.AGENT_RELEVANCE, 4, "Good"
                ),
            },
            weighted_average=4.0,
        ),
        EvaluationResult(
            scenario_name="test2",
            scores={
                EvaluationDimension.AGENT_RELEVANCE: DimensionScore(
                    EvaluationDimension.AGENT_RELEVANCE, 5, "Great"
                ),
            },
            weighted_average=5.0,
        ),
    ]

    comparison = compare_evaluations(baseline, improved)

    assert comparison["n_baseline"] == 2
    assert comparison["n_improved"] == 2
    assert "agent_relevance" in comparison["dimensions"]
    assert comparison["dimensions"]["agent_relevance"]["baseline_avg"] == 3.0
    assert comparison["dimensions"]["agent_relevance"]["improved_avg"] == 4.5
    assert comparison["dimensions"]["agent_relevance"]["improvement"] == 1.5
    assert comparison["overall"]["baseline_avg"] == 3.0
    assert comparison["overall"]["improved_avg"] == 4.5

    print("✓ compare_evaluations() works")


# ============================================================
# Integration Tests
# ============================================================

def test_full_rubric_coverage():
    """Test that rubrics cover key evaluation aspects."""
    # Check we have rubrics for all core FAtiMA components
    dimensions = list_dimensions()

    # BDI components
    assert EvaluationDimension.BELIEF_COHERENCE in dimensions
    assert EvaluationDimension.DESIRE_APPROPRIATENESS in dimensions
    assert EvaluationDimension.INTENTION_VALIDITY in dimensions

    # Action planning
    assert EvaluationDimension.ACTION_FEASIBILITY in dimensions
    assert EvaluationDimension.CONDITION_EFFECT_LOGIC in dimensions

    # Dialogue
    assert EvaluationDimension.DIALOGUE_QUALITY in dimensions
    assert EvaluationDimension.DIALOGUE_NATURALNESS in dimensions

    # Emotions
    assert EvaluationDimension.EMOTIONAL_CONSISTENCY in dimensions

    print("✓ Rubrics cover all FAtiMA components")


def test_rubric_weights_reasonable():
    """Test that rubric weights are reasonable."""
    total_weight = sum(r.weight for r in EVALUATION_RUBRICS.values())

    # Weights should sum to something reasonable (around 10-12)
    assert 8 <= total_weight <= 15

    # Core dimensions should have higher weights
    assert EVALUATION_RUBRICS[EvaluationDimension.ACTION_FEASIBILITY].weight >= 1.0
    assert EVALUATION_RUBRICS[EvaluationDimension.CONDITION_EFFECT_LOGIC].weight >= 1.0
    assert EVALUATION_RUBRICS[EvaluationDimension.OVERALL_COHERENCE].weight >= 1.0

    print(f"✓ Rubric weights sum to {total_weight:.1f}")


def test_evaluation_workflow():
    """Test complete evaluation workflow (without API)."""
    # Create mock scenario
    scenario = {
        "scenario_name": "test_workflow",
        "scenario_description": "A test scenario",
        "agents": {
            "Agent1": {
                "knowledge_base": ["BEL(Agent1, test) = True"],
                "intentions": {},
                "actions": {},
            }
        },
        "dialogue_tree": [],
    }

    # Create judge
    judge = LLMJudge(
        model_name="gpt-4o",
        dimensions=[EvaluationDimension.AGENT_RELEVANCE],
    )

    # Format scenario
    formatted = judge._format_scenario_for_evaluation(scenario)
    assert "test_workflow" in formatted or "test scenario" in formatted

    # Build prompt
    prompt = judge._build_evaluation_prompt(
        formatted,
        [EvaluationDimension.AGENT_RELEVANCE],
    )
    assert "Agent Relevance" in prompt

    # Parse mock response
    mock_response = '''{"evaluations": [{"dimension": "agent_relevance", "score": 4, "reasoning": "Good"}]}'''
    scores = judge._parse_evaluation_response(
        mock_response,
        [EvaluationDimension.AGENT_RELEVANCE],
    )
    assert scores[EvaluationDimension.AGENT_RELEVANCE].score == 4

    print("✓ Complete evaluation workflow works (without API)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-011: LLM-as-Judge Evaluation System Tests")
    print("=" * 60)

    # Rubric tests
    test_evaluation_dimension_enum()
    test_rubric_levels_complete()
    test_rubric_has_question()
    test_rubric_levels_have_descriptions()
    test_get_rubric()
    test_list_dimensions()
    test_get_all_rubrics_prompt()
    test_calculate_weighted_average()
    test_calculate_weighted_average_empty()
    test_rubric_get_prompt_section()

    # DimensionScore tests
    test_dimension_score_init()

    # EvaluationResult tests
    test_evaluation_result_init()
    test_evaluation_result_to_dict()
    test_evaluation_result_get_summary()

    # LLMJudge tests
    test_llm_judge_init()
    test_llm_judge_init_with_dimensions()
    test_llm_judge_format_scenario()
    test_llm_judge_build_evaluation_prompt()
    test_llm_judge_parse_json_response()
    test_llm_judge_parse_freeform_response()
    test_llm_judge_score_clamping()

    # Comparison tests
    test_compare_evaluations()

    # Integration tests
    test_full_rubric_coverage()
    test_rubric_weights_reasonable()
    test_evaluation_workflow()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
