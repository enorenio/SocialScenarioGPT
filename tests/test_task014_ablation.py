"""
Tests for TASK-014: Ablation Study Execution.

Tests the ablation runner, configuration parsing, condition management,
and statistical analysis utilities.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature_flags import FeatureFlags, get_profile, PROFILES
from analysis.metrics import (
    ScenarioMetrics,
    ExperimentMetrics,
    calculate_scenario_metrics,
    calculate_experiment_metrics,
)
from analysis.comparison import (
    compare_conditions,
    ConditionComparison,
    calculate_improvement,
)
from analysis.statistics import (
    calculate_mean,
    calculate_std,
    calculate_pooled_std,
    calculate_cohens_d,
    paired_t_test,
    independent_t_test,
    calculate_confidence_interval,
    compare_conditions_statistically,
    StatisticalTest,
    EffectSize,
    StatisticalComparison,
)
from experiments.ablation_runner import (
    AblationCondition,
    AblationRunner,
    ConditionResult,
    AblationStudyResults,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_scenario():
    """Create a sample scenario for testing."""
    return {
        "scenario_name": "test_scenario",
        "scenario_description": "A test scenario",
        "agents": {
            "Alice": {
                "knowledge_base": [
                    "BEL(Alice, happy) = True",
                    "BEL(Alice, at_home) = True",
                    "DES(Alice, meet_friend) = True",
                ],
                "intentions": {
                    "meet_bob": {
                        "action_plan": ["call_bob", "go_outside"]
                    }
                },
                "actions": {
                    "call_bob": {
                        "conditions": ["BEL(Alice, happy) = True"],
                        "effects": ["BEL(Alice, called_bob) = True"],
                        "occ_emotion": ["joy"],
                    },
                    "go_outside": {
                        "conditions": ["BEL(Alice, called_bob) = True"],
                        "effects": ["BEL(Alice, outside) = True"],
                        "occ_emotion": ["hope"],
                    },
                },
                "speak_actions": {
                    "greet": {"utterance": "Hello!"}
                },
            },
            "Bob": {
                "knowledge_base": [
                    "BEL(Bob, busy) = True",
                    "DES(Bob, rest) = True",
                ],
                "intentions": {},
                "actions": {},
                "speak_actions": {},
            },
        },
        "dialogue_tree": [
            "<Alice> Hi Bob!",
            "<Bob> Hello Alice!",
            "<Alice> How are you?",
        ],
        "last_ended": "end",
    }


@pytest.fixture
def sample_scenarios(sample_scenario):
    """Create multiple sample scenarios for testing."""
    scenarios = []
    for i in range(5):
        scenario = json.loads(json.dumps(sample_scenario))  # Deep copy
        scenario["scenario_name"] = f"test_scenario_{i}"
        # Vary some properties
        if i % 2 == 0:
            scenario["dialogue_tree"].append(f"<Alice> Line {i}")
        scenarios.append(scenario)
    return scenarios


@pytest.fixture
def baseline_metrics(sample_scenarios):
    """Create baseline experiment metrics."""
    return calculate_experiment_metrics(
        sample_scenarios,
        experiment_name="baseline",
        condition_id="C00",
    )


@pytest.fixture
def treatment_metrics(sample_scenarios):
    """Create treatment experiment metrics with improvements."""
    # Modify scenarios to simulate improvements
    improved_scenarios = []
    for scenario in sample_scenarios:
        scenario = json.loads(json.dumps(scenario))
        # Add more dialogue
        scenario["dialogue_tree"].extend([
            "<Alice> Extra line 1",
            "<Bob> Extra line 2",
        ])
        # Add more actions
        scenario["agents"]["Alice"]["actions"]["extra_action"] = {
            "conditions": [],
            "effects": ["BEL(Alice, extra) = True"],
            "occ_emotion": [],
        }
        improved_scenarios.append(scenario)

    return calculate_experiment_metrics(
        improved_scenarios,
        experiment_name="treatment",
        condition_id="C01",
    )


@pytest.fixture
def ablation_runner():
    """Create an ablation runner instance."""
    return AblationRunner()


# =============================================================================
# Statistical Utility Tests
# =============================================================================

class TestStatisticalUtilities:
    """Tests for statistical calculation utilities."""

    def test_calculate_mean_empty(self):
        """Test mean of empty list."""
        assert calculate_mean([]) == 0.0

    def test_calculate_mean_single(self):
        """Test mean of single value."""
        assert calculate_mean([5.0]) == 5.0

    def test_calculate_mean_multiple(self):
        """Test mean of multiple values."""
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0

    def test_calculate_std_empty(self):
        """Test std of empty list."""
        assert calculate_std([]) == 0.0

    def test_calculate_std_single(self):
        """Test std of single value."""
        assert calculate_std([5.0]) == 0.0

    def test_calculate_std_multiple(self):
        """Test std of multiple values."""
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        std = calculate_std(values)
        # Sample std is approximately 2.14
        assert 2.0 < std < 2.2

    def test_calculate_pooled_std(self):
        """Test pooled standard deviation."""
        pooled = calculate_pooled_std(1.0, 10, 1.0, 10)
        assert pooled == 1.0

    def test_calculate_pooled_std_unequal(self):
        """Test pooled std with unequal samples."""
        pooled = calculate_pooled_std(1.0, 5, 2.0, 10)
        assert pooled > 0

    def test_calculate_cohens_d_no_difference(self):
        """Test Cohen's d with no difference."""
        d, interpretation = calculate_cohens_d(
            mean1=5.0, std1=1.0, n1=10,
            mean2=5.0, std2=1.0, n2=10,
        )
        assert d == 0.0
        assert interpretation == "negligible"

    def test_calculate_cohens_d_small(self):
        """Test Cohen's d small effect."""
        d, interpretation = calculate_cohens_d(
            mean1=5.0, std1=1.0, n1=10,
            mean2=5.3, std2=1.0, n2=10,
        )
        assert 0.2 <= abs(d) < 0.5
        assert interpretation == "small"

    def test_calculate_cohens_d_medium(self):
        """Test Cohen's d medium effect."""
        d, interpretation = calculate_cohens_d(
            mean1=5.0, std1=1.0, n1=10,
            mean2=5.6, std2=1.0, n2=10,
        )
        assert 0.5 <= abs(d) < 0.8
        assert interpretation == "medium"

    def test_calculate_cohens_d_large(self):
        """Test Cohen's d large effect."""
        d, interpretation = calculate_cohens_d(
            mean1=5.0, std1=1.0, n1=10,
            mean2=6.0, std2=1.0, n2=10,
        )
        assert abs(d) >= 0.8
        assert interpretation == "large"

    def test_paired_t_test_equal_values(self):
        """Test paired t-test with equal values."""
        baseline = [1, 2, 3, 4, 5]
        treatment = [1, 2, 3, 4, 5]
        t_stat, p_value, df, significant = paired_t_test(baseline, treatment)
        assert t_stat == 0.0
        assert df == 4
        assert not significant

    def test_paired_t_test_different_values(self):
        """Test paired t-test with different values."""
        baseline = [1, 2, 3, 4, 5]
        treatment = [3, 5, 6, 7, 9]  # Variable increases
        t_stat, p_value, df, significant = paired_t_test(baseline, treatment)
        assert t_stat != 0  # Non-zero because differences vary
        assert df == 4

    def test_independent_t_test_equal(self):
        """Test independent t-test with equal groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1, 2, 3, 4, 5]
        t_stat, p_value, df, significant = independent_t_test(group1, group2)
        assert abs(t_stat) < 0.01
        assert not significant

    def test_independent_t_test_different(self):
        """Test independent t-test with different groups."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]
        t_stat, p_value, df, significant = independent_t_test(group1, group2)
        # t_stat is (mean2 - mean1) / SE, so positive when group2 > group1
        assert t_stat > 0  # group2 > group1, positive t-statistic
        assert significant  # Should be significant

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        values = [1, 2, 3, 4, 5]
        ci = calculate_confidence_interval(values, confidence=0.95)
        assert ci.mean == 3.0
        assert ci.lower < ci.mean
        assert ci.upper > ci.mean
        assert ci.confidence_level == 0.95


# =============================================================================
# Statistical Comparison Tests
# =============================================================================

class TestStatisticalComparison:
    """Tests for statistical comparison between conditions."""

    def test_compare_conditions_statistically(self, baseline_metrics, treatment_metrics):
        """Test statistical comparison."""
        comparison = compare_conditions_statistically(
            baseline_metrics,
            treatment_metrics,
        )

        assert comparison.baseline_name == "baseline"
        assert comparison.treatment_name == "treatment"
        assert len(comparison.tests) > 0
        assert len(comparison.effect_sizes) > 0

    def test_statistical_comparison_has_key_metrics(self, baseline_metrics, treatment_metrics):
        """Test that key metrics are compared."""
        comparison = compare_conditions_statistically(
            baseline_metrics,
            treatment_metrics,
        )

        metric_names = [t.metric_name for t in comparison.tests]
        assert "intention_completion_rate" in metric_names
        assert "dialogue_lines" in metric_names

    def test_get_significant_improvements(self, baseline_metrics, treatment_metrics):
        """Test getting significant improvements."""
        comparison = compare_conditions_statistically(
            baseline_metrics,
            treatment_metrics,
        )

        # Should return a list (may be empty or not depending on data)
        improvements = comparison.get_significant_improvements()
        assert isinstance(improvements, list)

    def test_get_large_effects(self, baseline_metrics, treatment_metrics):
        """Test getting large effect sizes."""
        comparison = compare_conditions_statistically(
            baseline_metrics,
            treatment_metrics,
        )

        large_effects = comparison.get_large_effects()
        assert isinstance(large_effects, list)


# =============================================================================
# Ablation Configuration Tests
# =============================================================================

class TestAblationConfiguration:
    """Tests for ablation study configuration."""

    def test_ablation_runner_has_conditions(self, ablation_runner):
        """Test that ablation runner has conditions from feature_flags.py."""
        assert ablation_runner.conditions is not None
        assert len(ablation_runner.conditions) >= 11  # C00-C10

    def test_ablation_runner_parses_conditions(self, ablation_runner):
        """Test that conditions are parsed correctly."""
        assert len(ablation_runner.conditions) > 0
        assert "C00" in ablation_runner.conditions
        assert "C01" in ablation_runner.conditions

    def test_condition_has_required_fields(self, ablation_runner):
        """Test that conditions have required fields."""
        for cond_id, condition in ablation_runner.conditions.items():
            assert condition.condition_id == cond_id
            assert condition.name is not None
            assert condition.description is not None
            assert condition.features is not None

    def test_get_condition(self, ablation_runner):
        """Test getting a specific condition."""
        condition = ablation_runner.get_condition("C00")
        assert condition.condition_id == "C00"
        assert condition.name == "Baseline"  # Now capitalized from CONDITION_DESCRIPTIONS

    def test_get_condition_invalid(self, ablation_runner):
        """Test getting invalid condition."""
        with pytest.raises(ValueError):
            ablation_runner.get_condition("INVALID")

    def test_get_conditions_in_group(self, ablation_runner):
        """Test getting conditions in a group."""
        conditions = ablation_runner.get_conditions_in_group("single_feature")
        assert "C01" in conditions
        assert "C02" in conditions

    def test_get_conditions_in_group_invalid(self, ablation_runner):
        """Test getting invalid group."""
        with pytest.raises(ValueError):
            ablation_runner.get_conditions_in_group("INVALID")


# =============================================================================
# Ablation Condition Tests
# =============================================================================

class TestAblationCondition:
    """Tests for AblationCondition class."""

    def test_condition_get_feature_flags_from_profile(self):
        """Test getting feature flags from profile."""
        condition = AblationCondition(
            condition_id="C00",
            name="baseline",
            description="Baseline",
            features={},
            profile="baseline",
        )
        flags = condition.get_feature_flags()
        assert isinstance(flags, FeatureFlags)
        assert not flags.use_gpt4

    def test_condition_get_feature_flags_from_features(self):
        """Test getting feature flags from features dict."""
        condition = AblationCondition(
            condition_id="C01",
            name="gpt4_only",
            description="GPT-4 only",
            features={"use_gpt4": True},
            profile=None,
        )
        flags = condition.get_feature_flags()
        assert flags.use_gpt4

    def test_condition_to_dict(self):
        """Test condition serialization."""
        condition = AblationCondition(
            condition_id="C00",
            name="baseline",
            description="Baseline test",
            features={"use_gpt4": False},
            notes="Test notes",
        )
        data = condition.to_dict()
        assert data["condition_id"] == "C00"
        assert data["name"] == "baseline"
        assert data["notes"] == "Test notes"


# =============================================================================
# Condition Result Tests
# =============================================================================

class TestConditionResult:
    """Tests for ConditionResult class."""

    def test_condition_result_to_dict(self, baseline_metrics):
        """Test condition result serialization."""
        condition = AblationCondition(
            condition_id="C00",
            name="baseline",
            description="Baseline",
            features={},
        )
        result = ConditionResult(
            condition=condition,
            metrics=baseline_metrics,
            timing={"total_time_seconds": 100},
            scenario_results=[{"name": "test", "success": True}],
            success=True,
        )
        data = result.to_dict()
        assert "condition" in data
        assert "metrics" in data
        assert "timing" in data
        assert data["success"] is True


# =============================================================================
# Ablation Study Results Tests
# =============================================================================

class TestAblationStudyResults:
    """Tests for AblationStudyResults class."""

    def test_study_results_to_dict(self, baseline_metrics):
        """Test study results serialization."""
        condition = AblationCondition(
            condition_id="C00",
            name="baseline",
            description="Baseline",
            features={},
        )
        cond_result = ConditionResult(
            condition=condition,
            metrics=baseline_metrics,
            timing={},
            scenario_results=[],
        )
        results = AblationStudyResults(
            study_name="Test Study",
            timestamp="20241224_120000",
            conditions_run=["C00"],
            condition_results={"C00": cond_result},
            comparisons={},
            summary={"total_conditions": 1},
        )
        data = results.to_dict()
        assert data["study_name"] == "Test Study"
        assert "C00" in data["condition_results"]


# =============================================================================
# Ablation Runner Tests
# =============================================================================

class TestAblationRunner:
    """Tests for AblationRunner class."""

    def test_runner_initialization(self, ablation_runner):
        """Test runner initialization."""
        assert ablation_runner.conditions is not None
        assert len(ablation_runner.conditions) >= 11  # C00-C10

    def test_runner_dry_run_condition(self, ablation_runner):
        """Test running a condition in dry run mode."""
        result = ablation_runner.run_condition(
            condition_id="C00",
            n_scenarios=5,
            dry_run=True,
        )
        assert result is not None
        assert result.condition.condition_id == "C00"
        assert "Dry run" in result.error_message

    def test_runner_generate_report(self, ablation_runner, baseline_metrics):
        """Test report generation."""
        condition = AblationCondition(
            condition_id="C00",
            name="baseline",
            description="Baseline",
            features={},
        )
        cond_result = ConditionResult(
            condition=condition,
            metrics=baseline_metrics,
            timing={"total_time_minutes": 1.0},
            scenario_results=[],
            success=True,
        )
        results = AblationStudyResults(
            study_name="Test Study",
            timestamp="20241224_120000",
            conditions_run=["C00"],
            condition_results={"C00": cond_result},
            comparisons={},
            summary={
                "successful_conditions": 1,
                "total_time_minutes": 1.0,
            },
        )
        report = ablation_runner.generate_report(results)
        assert "Test Study" in report
        assert "C00" in report


# =============================================================================
# Feature Flag Tests
# =============================================================================

class TestFeatureFlagIntegration:
    """Tests for feature flag integration with ablation runner."""

    def test_all_profiles_defined(self):
        """Test that all expected profiles are defined."""
        expected_profiles = [
            "baseline",
            "gpt4_only",
            "full_context_only",
            "cot_only",
            "dialogue_only",
            "gpt4_full_context",
            "gpt4_full_context_verification",
            "full_system",
        ]
        for profile in expected_profiles:
            assert profile in PROFILES

    def test_baseline_profile_all_disabled(self):
        """Test baseline profile has all features disabled."""
        flags = get_profile("baseline")
        assert not flags.use_gpt4
        assert not flags.full_context
        assert not flags.verification_loop
        assert not flags.cot_enhancement
        assert not flags.dialogue_improvement

    def test_full_system_profile_all_enabled(self):
        """Test full_system profile has core features enabled."""
        flags = get_profile("full_system")
        assert flags.use_gpt4
        assert flags.full_context
        assert flags.verification_loop
        assert flags.cot_enhancement
        assert flags.dialogue_improvement

    def test_feature_dependencies_respected(self):
        """Test that feature dependencies are enforced."""
        # verification_loop requires full_context
        with pytest.raises(ValueError):
            FeatureFlags(verification_loop=True, full_context=False)

    def test_ablation_conditions_match_profiles(self, ablation_runner):
        """Test that ablation conditions match expected feature profiles."""
        c00 = ablation_runner.get_condition("C00")
        flags = c00.get_feature_flags()
        assert not flags.use_gpt4
        assert not flags.full_context

        c01 = ablation_runner.get_condition("C01")
        flags = c01.get_feature_flags()
        assert flags.use_gpt4
        assert not flags.full_context


# =============================================================================
# Comparison Calculation Tests
# =============================================================================

class TestComparisonCalculations:
    """Tests for comparison calculations."""

    def test_calculate_improvement_positive(self):
        """Test positive improvement calculation."""
        improvement = calculate_improvement(0.3, 0.6)
        assert improvement == 100.0  # 100% improvement

    def test_calculate_improvement_negative(self):
        """Test negative improvement calculation."""
        improvement = calculate_improvement(0.6, 0.3)
        assert improvement == -50.0  # 50% decrease

    def test_calculate_improvement_from_zero(self):
        """Test improvement from zero baseline."""
        improvement = calculate_improvement(0.0, 0.5)
        assert improvement == 100.0

    def test_calculate_improvement_to_zero(self):
        """Test improvement to zero."""
        improvement = calculate_improvement(0.5, 0.0)
        assert improvement == -100.0

    def test_compare_conditions_basic(self, baseline_metrics, treatment_metrics):
        """Test basic condition comparison."""
        comparison = compare_conditions(baseline_metrics, treatment_metrics)
        assert comparison.baseline_name == "baseline"
        assert comparison.treatment_name == "treatment"

    def test_compare_conditions_has_improvements(self, baseline_metrics, treatment_metrics):
        """Test that comparison has improvement calculations."""
        comparison = compare_conditions(baseline_metrics, treatment_metrics)
        assert hasattr(comparison, "dialogue_lines_improvement")
        assert hasattr(comparison, "total_artifacts_improvement")


# =============================================================================
# Feature Flags Configuration Tests (Replaces YAML)
# =============================================================================

class TestFeatureFlagsConfiguration:
    """Tests for feature_flags.py-based configuration."""

    def test_profiles_has_all_conditions(self):
        """Test PROFILES has all C00-C10 conditions."""
        from config.feature_flags import PROFILES
        for i in range(11):
            cond_id = f"C{i:02d}"
            assert cond_id in PROFILES, f"{cond_id} not in PROFILES"

    def test_condition_groups_defined(self):
        """Test CONDITION_GROUPS are defined."""
        from config.feature_flags import CONDITION_GROUPS
        assert "single_feature" in CONDITION_GROUPS
        assert "combined" in CONDITION_GROUPS
        assert "ablation" in CONDITION_GROUPS
        assert "all" in CONDITION_GROUPS

    def test_all_group_has_all_conditions(self):
        """Test 'all' group contains all conditions."""
        from config.feature_flags import CONDITION_GROUPS
        all_conditions = CONDITION_GROUPS["all"]
        for i in range(11):
            cond_id = f"C{i:02d}"
            assert cond_id in all_conditions

    def test_ablation_runner_uses_profiles(self, ablation_runner):
        """Test runner uses PROFILES from feature_flags."""
        from config.feature_flags import PROFILES
        for cond_id in ["C00", "C01", "C02", "C08"]:
            condition = ablation_runner.get_condition(cond_id)
            flags = condition.get_feature_flags()
            expected_flags = PROFILES[cond_id]
            assert flags.to_dict() == expected_flags.to_dict()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scenario_list_metrics(self):
        """Test metrics calculation with empty list."""
        metrics = calculate_experiment_metrics([], experiment_name="empty")
        assert metrics.total_scenarios == 0

    def test_scenario_without_agents(self):
        """Test metrics for scenario without agents."""
        scenario = {"scenario_name": "test", "agents": {}}
        metrics = calculate_scenario_metrics(scenario)
        assert metrics.agents == 0

    def test_scenario_without_dialogue(self):
        """Test metrics for scenario without dialogue."""
        scenario = {
            "scenario_name": "test",
            "agents": {"Alice": {"knowledge_base": []}},
            "dialogue_tree": [],
        }
        metrics = calculate_scenario_metrics(scenario)
        assert metrics.dialogue_lines == 0
        assert not metrics.has_dialogue

    def test_statistical_test_single_sample(self):
        """Test statistical tests with single sample."""
        t_stat, p_value, df, sig = independent_t_test([5], [5])
        assert df == 0
        assert not sig

    def test_effect_size_zero_pooled_std(self):
        """Test effect size when pooled std is zero."""
        d, interp = calculate_cohens_d(
            mean1=5.0, std1=0.0, n1=1,
            mean2=5.0, std2=0.0, n2=1,
        )
        assert d == 0.0
        assert interp == "negligible"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
