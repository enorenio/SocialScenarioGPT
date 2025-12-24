"""
Tests for TASK-013: Automated Metrics Dashboard.

Tests metrics calculation, comparison functions, and data loading.
"""

import json
import tempfile
from pathlib import Path

import pytest

from analysis.metrics import (
    ScenarioMetrics,
    ExperimentMetrics,
    calculate_scenario_metrics,
    calculate_experiment_metrics,
    load_scenarios_from_directory,
    _count_executable_actions,
    _count_completable_intentions,
)
from analysis.comparison import (
    ConditionComparison,
    compare_conditions,
    compare_multiple_conditions,
    calculate_improvement,
    generate_comparison_table,
    summarize_comparisons,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_scenario():
    """A simple scenario for testing."""
    return {
        "scenario_name": "test_scenario",
        "agents": {
            "Agent1": {
                "knowledge_base": [
                    "BEL(fact1)",
                    "BEL(fact2)",
                    "DES(goal1)",
                ],
                "intentions": {
                    "intent1": {
                        "action_plan": ["action1", "action2"],
                    },
                },
                "actions": {
                    "action1": {
                        "conditions": ["BEL(fact1)"],
                        "effects": ["BEL(result1)"],
                        "emotion_condition": ["happy"],
                        "occ_emotion": ["joy"],
                    },
                    "action2": {
                        "conditions": ["BEL(result1)"],
                        "effects": ["BEL(result2)"],
                    },
                },
                "speak_actions": {
                    "speak1": {},
                },
            },
        },
        "dialogue_tree": [
            '<S1, S2, Inform, Neutral, "Hello">',
            '<S2, S3, Inform, Neutral, "World">',
        ],
    }


@pytest.fixture
def multi_agent_scenario():
    """A scenario with multiple agents."""
    return {
        "scenario_name": "multi_agent",
        "agents": {
            "Agent1": {
                "knowledge_base": ["BEL(a)", "DES(b)"],
                "intentions": {"i1": {"action_plan": ["a1"]}},
                "actions": {"a1": {"conditions": ["BEL(a)"], "effects": ["BEL(c)"]}},
            },
            "Agent2": {
                "knowledge_base": ["BEL(x)", "DES(y)"],
                "intentions": {"i2": {"action_plan": ["a2"]}},
                "actions": {"a2": {"conditions": ["BEL(x)"], "effects": ["BEL(z)"]}},
            },
        },
        "dialogue_tree": [
            '<S1, S2, Greet, Neutral, "Hi">',
            '<S2, S3, Greet, Neutral, "Hello">',
            '<S3, S4, Farewell, Neutral, "Bye">',
        ],
    }


@pytest.fixture
def incomplete_scenario():
    """A scenario missing key components."""
    return {
        "scenario_name": "incomplete",
        "agents": {},
        "dialogue_tree": [],
    }


@pytest.fixture
def scenario_list(simple_scenario, multi_agent_scenario):
    """List of scenarios for experiment testing."""
    return [simple_scenario, multi_agent_scenario]


# =============================================================================
# ScenarioMetrics Tests
# =============================================================================

class TestScenarioMetrics:
    """Tests for ScenarioMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = ScenarioMetrics(scenario_name="test")
        assert metrics.agents == 0
        assert metrics.beliefs == 0
        assert metrics.is_complete is True
        assert metrics.has_dialogue is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ScenarioMetrics(
            scenario_name="test",
            agents=2,
            beliefs=5,
            dialogue_lines=10,
        )
        result = metrics.to_dict()

        assert result["scenario_name"] == "test"
        assert result["artifact_counts"]["agents"] == 2
        assert result["artifact_counts"]["beliefs"] == 5
        assert result["dialogue_metrics"]["lines"] == 10


class TestCalculateScenarioMetrics:
    """Tests for calculate_scenario_metrics function."""

    def test_counts_agents(self, simple_scenario):
        """Test agent counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.agents == 1

    def test_counts_beliefs_and_desires(self, simple_scenario):
        """Test belief and desire counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.beliefs == 2
        assert metrics.desires == 1

    def test_counts_intentions(self, simple_scenario):
        """Test intention counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.intentions == 1

    def test_counts_actions(self, simple_scenario):
        """Test action counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.actions == 2

    def test_counts_conditions_and_effects(self, simple_scenario):
        """Test condition and effect counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.conditions == 2
        assert metrics.effects == 2

    def test_counts_emotions(self, simple_scenario):
        """Test emotion counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.emotion_before == 1
        assert metrics.emotion_after == 1

    def test_counts_dialogue_lines(self, simple_scenario):
        """Test dialogue line counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.dialogue_lines == 2

    def test_counts_speak_actions(self, simple_scenario):
        """Test speak action counting."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.speak_actions == 1

    def test_total_artifacts(self, simple_scenario):
        """Test total artifact calculation."""
        metrics = calculate_scenario_metrics(simple_scenario)
        # Should sum all artifact counts
        expected = (
            metrics.agents + metrics.beliefs + metrics.desires +
            metrics.intentions + metrics.actions + metrics.conditions +
            metrics.effects + metrics.emotion_before + metrics.emotion_after +
            metrics.dialogue_lines + metrics.speak_actions
        )
        assert metrics.total_artifacts == expected

    def test_multi_agent_counting(self, multi_agent_scenario):
        """Test counting across multiple agents."""
        metrics = calculate_scenario_metrics(multi_agent_scenario)
        assert metrics.agents == 2
        assert metrics.intentions == 2
        assert metrics.actions == 2

    def test_incomplete_scenario(self, incomplete_scenario):
        """Test handling of incomplete scenario."""
        metrics = calculate_scenario_metrics(incomplete_scenario)
        assert metrics.agents == 0
        assert metrics.is_complete is False
        assert metrics.has_dialogue is False

    def test_quality_flags(self, simple_scenario):
        """Test quality flag calculation."""
        metrics = calculate_scenario_metrics(simple_scenario)
        assert metrics.is_complete is True
        assert metrics.has_dialogue is True
        assert metrics.has_emotions is True


class TestExecutableActions:
    """Tests for executable action counting."""

    def test_counts_executable(self, simple_scenario):
        """Test executable action counting."""
        total, executable = _count_executable_actions(simple_scenario)
        assert total == 2
        # action1 is executable (fact1 in knowledge), action2 is not (result1 not in knowledge)
        assert executable == 1

    def test_empty_scenario(self, incomplete_scenario):
        """Test with empty scenario."""
        total, executable = _count_executable_actions(incomplete_scenario)
        assert total == 0
        assert executable == 0


class TestCompletableIntentions:
    """Tests for completable intention counting."""

    def test_counts_completable(self, simple_scenario):
        """Test completable intention counting."""
        completed, total = _count_completable_intentions(simple_scenario)
        assert total == 1
        # Intent can complete: action1 adds result1, then action2 can execute
        assert completed == 1

    def test_empty_scenario(self, incomplete_scenario):
        """Test with empty scenario."""
        completed, total = _count_completable_intentions(incomplete_scenario)
        assert total == 0
        assert completed == 0


# =============================================================================
# ExperimentMetrics Tests
# =============================================================================

class TestExperimentMetrics:
    """Tests for ExperimentMetrics dataclass."""

    def test_default_timestamp(self):
        """Test automatic timestamp generation."""
        metrics = ExperimentMetrics(experiment_name="test")
        assert metrics.timestamp != ""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = ExperimentMetrics(
            experiment_name="test",
            total_scenarios=5,
            mean_agents=2.5,
        )
        result = metrics.to_dict()

        assert result["experiment_name"] == "test"
        assert result["summary"]["total_scenarios"] == 5
        assert result["artifact_counts_mean"]["agents"] == 2.5

    def test_to_json(self):
        """Test JSON serialization."""
        metrics = ExperimentMetrics(experiment_name="test")
        json_str = metrics.to_json()
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "test"


class TestCalculateExperimentMetrics:
    """Tests for calculate_experiment_metrics function."""

    def test_counts_scenarios(self, scenario_list):
        """Test scenario counting."""
        metrics = calculate_experiment_metrics(scenario_list, "test")
        assert metrics.total_scenarios == 2

    def test_calculates_means(self, scenario_list):
        """Test mean calculation."""
        metrics = calculate_experiment_metrics(scenario_list, "test")
        # simple_scenario has 1 agent, multi_agent has 2
        assert metrics.mean_agents == 1.5

    def test_empty_list(self):
        """Test with empty scenario list."""
        metrics = calculate_experiment_metrics([], "empty")
        assert metrics.total_scenarios == 0
        assert metrics.mean_agents == 0

    def test_stores_scenario_metrics(self, scenario_list):
        """Test that individual scenario metrics are stored."""
        metrics = calculate_experiment_metrics(scenario_list, "test")
        assert len(metrics.scenario_metrics) == 2

    def test_calculates_rates(self, scenario_list):
        """Test rate calculations."""
        metrics = calculate_experiment_metrics(scenario_list, "test")
        assert 0 <= metrics.intention_completion_rate <= 1
        assert 0 <= metrics.executable_actions_rate <= 1


# =============================================================================
# File Loading Tests
# =============================================================================

class TestLoadScenariosFromDirectory:
    """Tests for load_scenarios_from_directory function."""

    def test_loads_json_files(self, simple_scenario):
        """Test loading from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test file
            filepath = Path(tmpdir) / "test_scenario.json"
            with open(filepath, 'w') as f:
                json.dump(simple_scenario, f)

            scenarios = load_scenarios_from_directory(tmpdir, "test_*.json")
            assert len(scenarios) == 1
            assert scenarios[0]["scenario_name"] == "test_scenario"

    def test_nonexistent_directory(self):
        """Test with non-existent directory."""
        scenarios = load_scenarios_from_directory("/nonexistent/path")
        assert scenarios == []

    def test_invalid_json(self):
        """Test handling of invalid JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_invalid.json"
            with open(filepath, 'w') as f:
                f.write("not valid json")

            scenarios = load_scenarios_from_directory(tmpdir, "test_*.json")
            assert scenarios == []


# =============================================================================
# Comparison Tests
# =============================================================================

class TestCalculateImprovement:
    """Tests for calculate_improvement function."""

    def test_positive_improvement(self):
        """Test positive improvement calculation."""
        result = calculate_improvement(0.5, 0.75)
        assert result == 50.0

    def test_negative_improvement(self):
        """Test negative improvement calculation."""
        result = calculate_improvement(0.5, 0.25)
        assert result == -50.0

    def test_zero_baseline(self):
        """Test with zero baseline."""
        assert calculate_improvement(0, 0.5) == 100.0
        assert calculate_improvement(0, -0.5) == -100.0
        assert calculate_improvement(0, 0) == 0.0

    def test_no_change(self):
        """Test with no change."""
        result = calculate_improvement(0.5, 0.5)
        assert result == 0.0


class TestCompareConditions:
    """Tests for compare_conditions function."""

    def test_creates_comparison(self, scenario_list):
        """Test basic comparison creation."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        treatment = calculate_experiment_metrics([scenario_list[1]], "treatment")

        comparison = compare_conditions(baseline, treatment)

        assert comparison.baseline_name == "baseline"
        assert comparison.treatment_name == "treatment"
        assert comparison.baseline_n == 1
        assert comparison.treatment_n == 1

    def test_calculates_differences(self, scenario_list):
        """Test difference calculation."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        treatment = calculate_experiment_metrics([scenario_list[1]], "treatment")

        comparison = compare_conditions(baseline, treatment)

        # Treatment has more dialogue lines
        assert comparison.dialogue_lines_diff == 1.0  # 3 - 2

    def test_to_dict(self, scenario_list):
        """Test comparison to dict conversion."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        treatment = calculate_experiment_metrics([scenario_list[1]], "treatment")

        comparison = compare_conditions(baseline, treatment)
        result = comparison.to_dict()

        assert "comparison" in result
        assert "absolute_differences" in result
        assert "percent_improvements" in result


class TestCompareMultipleConditions:
    """Tests for compare_multiple_conditions function."""

    def test_compares_all_treatments(self, scenario_list):
        """Test comparison of multiple treatments."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        t1 = calculate_experiment_metrics([scenario_list[1]], "treatment1")
        t2 = calculate_experiment_metrics(scenario_list, "treatment2")

        comparisons = compare_multiple_conditions(baseline, [t1, t2])

        assert len(comparisons) == 2
        assert comparisons[0].treatment_name == "treatment1"
        assert comparisons[1].treatment_name == "treatment2"


class TestGenerateComparisonTable:
    """Tests for generate_comparison_table function."""

    def test_generates_markdown(self, scenario_list):
        """Test markdown table generation."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        treatment = calculate_experiment_metrics([scenario_list[1]], "treatment")

        comparison = compare_conditions(baseline, treatment)
        table = generate_comparison_table([comparison])

        assert "treatment" in table
        assert "|" in table  # Markdown table syntax
        assert "Intention Completion Rate" in table

    def test_empty_comparisons(self):
        """Test with no comparisons."""
        table = generate_comparison_table([])
        assert "No comparisons" in table


class TestSummarizeComparisons:
    """Tests for summarize_comparisons function."""

    def test_summarizes_multiple(self, scenario_list):
        """Test summary of multiple comparisons."""
        baseline = calculate_experiment_metrics([scenario_list[0]], "baseline")
        t1 = calculate_experiment_metrics([scenario_list[1]], "treatment1")
        t2 = calculate_experiment_metrics(scenario_list, "treatment2")

        comparisons = compare_multiple_conditions(baseline, [t1, t2])
        summary = summarize_comparisons(comparisons)

        assert summary["num_comparisons"] == 2
        assert "intention_completion" in summary
        assert "best_overall" in summary

    def test_empty_comparisons(self):
        """Test with empty list."""
        summary = summarize_comparisons([])
        assert "error" in summary


# =============================================================================
# ConditionComparison Tests
# =============================================================================

class TestConditionComparison:
    """Tests for ConditionComparison dataclass."""

    def test_default_values(self):
        """Test default comparison values."""
        comparison = ConditionComparison(
            baseline_name="baseline",
            treatment_name="treatment",
        )
        assert comparison.baseline_n == 0
        assert comparison.intention_completion_diff == 0.0

    def test_to_dict_structure(self):
        """Test dictionary structure."""
        comparison = ConditionComparison(
            baseline_name="baseline",
            treatment_name="treatment",
            intention_completion_improvement=25.0,
        )
        result = comparison.to_dict()

        assert result["comparison"] == "treatment vs baseline"
        assert result["percent_improvements"]["intention_completion_rate"] == 25.0
