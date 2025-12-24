"""
Cross-condition comparison for ablation studies.

Provides statistical comparison between experiment conditions,
including improvement calculations and significance testing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

from analysis.metrics import ExperimentMetrics


@dataclass
class ConditionComparison:
    """Comparison between two experimental conditions."""
    baseline_name: str
    treatment_name: str

    # Sample sizes
    baseline_n: int = 0
    treatment_n: int = 0

    # Key metric comparisons (treatment - baseline)
    intention_completion_diff: float = 0.0
    executable_actions_diff: float = 0.0
    dialogue_lines_diff: float = 0.0
    dialogue_branches_diff: float = 0.0
    total_artifacts_diff: float = 0.0

    # Percentage improvements
    intention_completion_improvement: float = 0.0
    executable_actions_improvement: float = 0.0
    dialogue_lines_improvement: float = 0.0
    dialogue_branches_improvement: float = 0.0
    total_artifacts_improvement: float = 0.0

    # Individual metric comparisons
    metric_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comparison": f"{self.treatment_name} vs {self.baseline_name}",
            "sample_sizes": {
                "baseline": self.baseline_n,
                "treatment": self.treatment_n,
            },
            "absolute_differences": {
                "intention_completion_rate": round(self.intention_completion_diff, 4),
                "executable_actions_rate": round(self.executable_actions_diff, 4),
                "mean_dialogue_lines": round(self.dialogue_lines_diff, 2),
                "mean_dialogue_branches": round(self.dialogue_branches_diff, 2),
                "mean_total_artifacts": round(self.total_artifacts_diff, 2),
            },
            "percent_improvements": {
                "intention_completion_rate": round(self.intention_completion_improvement, 1),
                "executable_actions_rate": round(self.executable_actions_improvement, 1),
                "mean_dialogue_lines": round(self.dialogue_lines_improvement, 1),
                "mean_dialogue_branches": round(self.dialogue_branches_improvement, 1),
                "mean_total_artifacts": round(self.total_artifacts_improvement, 1),
            },
            "metric_comparisons": self.metric_comparisons,
        }


def calculate_improvement(baseline: float, treatment: float) -> float:
    """
    Calculate percentage improvement from baseline to treatment.

    Args:
        baseline: Baseline value
        treatment: Treatment value

    Returns:
        Percentage improvement (positive = better, negative = worse)
    """
    if baseline == 0:
        if treatment == 0:
            return 0.0
        return 100.0 if treatment > 0 else -100.0

    return ((treatment - baseline) / abs(baseline)) * 100


def compare_conditions(
    baseline: ExperimentMetrics,
    treatment: ExperimentMetrics,
) -> ConditionComparison:
    """
    Compare two experimental conditions.

    Args:
        baseline: Baseline condition metrics
        treatment: Treatment condition metrics

    Returns:
        ConditionComparison with all calculated differences
    """
    comparison = ConditionComparison(
        baseline_name=baseline.experiment_name,
        treatment_name=treatment.experiment_name,
        baseline_n=baseline.total_scenarios,
        treatment_n=treatment.total_scenarios,
    )

    # Key metrics - absolute differences
    comparison.intention_completion_diff = (
        treatment.intention_completion_rate - baseline.intention_completion_rate
    )
    comparison.executable_actions_diff = (
        treatment.executable_actions_rate - baseline.executable_actions_rate
    )
    comparison.dialogue_lines_diff = (
        treatment.mean_dialogue_lines - baseline.mean_dialogue_lines
    )
    comparison.dialogue_branches_diff = (
        treatment.mean_dialogue_branch_points - baseline.mean_dialogue_branch_points
    )
    comparison.total_artifacts_diff = (
        treatment.mean_total_artifacts - baseline.mean_total_artifacts
    )

    # Percentage improvements
    comparison.intention_completion_improvement = calculate_improvement(
        baseline.intention_completion_rate,
        treatment.intention_completion_rate,
    )
    comparison.executable_actions_improvement = calculate_improvement(
        baseline.executable_actions_rate,
        treatment.executable_actions_rate,
    )
    comparison.dialogue_lines_improvement = calculate_improvement(
        baseline.mean_dialogue_lines,
        treatment.mean_dialogue_lines,
    )
    comparison.dialogue_branches_improvement = calculate_improvement(
        baseline.mean_dialogue_branch_points,
        treatment.mean_dialogue_branch_points,
    )
    comparison.total_artifacts_improvement = calculate_improvement(
        baseline.mean_total_artifacts,
        treatment.mean_total_artifacts,
    )

    # Detailed metric comparisons
    comparison.metric_comparisons = {
        "agents": {
            "baseline": baseline.mean_agents,
            "treatment": treatment.mean_agents,
            "diff": treatment.mean_agents - baseline.mean_agents,
            "improvement": calculate_improvement(baseline.mean_agents, treatment.mean_agents),
        },
        "beliefs": {
            "baseline": baseline.mean_beliefs,
            "treatment": treatment.mean_beliefs,
            "diff": treatment.mean_beliefs - baseline.mean_beliefs,
            "improvement": calculate_improvement(baseline.mean_beliefs, treatment.mean_beliefs),
        },
        "desires": {
            "baseline": baseline.mean_desires,
            "treatment": treatment.mean_desires,
            "diff": treatment.mean_desires - baseline.mean_desires,
            "improvement": calculate_improvement(baseline.mean_desires, treatment.mean_desires),
        },
        "intentions": {
            "baseline": baseline.mean_intentions,
            "treatment": treatment.mean_intentions,
            "diff": treatment.mean_intentions - baseline.mean_intentions,
            "improvement": calculate_improvement(baseline.mean_intentions, treatment.mean_intentions),
        },
        "actions": {
            "baseline": baseline.mean_actions,
            "treatment": treatment.mean_actions,
            "diff": treatment.mean_actions - baseline.mean_actions,
            "improvement": calculate_improvement(baseline.mean_actions, treatment.mean_actions),
        },
        "conditions": {
            "baseline": baseline.mean_conditions,
            "treatment": treatment.mean_conditions,
            "diff": treatment.mean_conditions - baseline.mean_conditions,
            "improvement": calculate_improvement(baseline.mean_conditions, treatment.mean_conditions),
        },
        "effects": {
            "baseline": baseline.mean_effects,
            "treatment": treatment.mean_effects,
            "diff": treatment.mean_effects - baseline.mean_effects,
            "improvement": calculate_improvement(baseline.mean_effects, treatment.mean_effects),
        },
        "dialogue_lines": {
            "baseline": baseline.mean_dialogue_lines,
            "treatment": treatment.mean_dialogue_lines,
            "diff": treatment.mean_dialogue_lines - baseline.mean_dialogue_lines,
            "improvement": calculate_improvement(
                baseline.mean_dialogue_lines, treatment.mean_dialogue_lines
            ),
        },
        "speak_actions": {
            "baseline": baseline.mean_speak_actions,
            "treatment": treatment.mean_speak_actions,
            "diff": treatment.mean_speak_actions - baseline.mean_speak_actions,
            "improvement": calculate_improvement(
                baseline.mean_speak_actions, treatment.mean_speak_actions
            ),
        },
    }

    return comparison


def generate_comparison_table(
    comparisons: List[ConditionComparison],
    metrics: Optional[List[str]] = None,
) -> str:
    """
    Generate a markdown comparison table.

    Args:
        comparisons: List of condition comparisons
        metrics: Which metrics to include (None = all key metrics)

    Returns:
        Markdown formatted table
    """
    if not comparisons:
        return "No comparisons to display."

    if metrics is None:
        metrics = [
            "intention_completion",
            "executable_actions",
            "dialogue_lines",
            "dialogue_branches",
            "total_artifacts",
        ]

    # Header
    lines = ["| Metric | " + " | ".join(
        c.treatment_name for c in comparisons
    ) + " |"]
    lines.append("|" + "|".join(["---"] * (len(comparisons) + 1)) + "|")

    # Metric rows
    metric_display = {
        "intention_completion": "Intention Completion Rate",
        "executable_actions": "Executable Actions Rate",
        "dialogue_lines": "Mean Dialogue Lines",
        "dialogue_branches": "Mean Dialogue Branches",
        "total_artifacts": "Mean Total Artifacts",
    }

    for metric in metrics:
        row = f"| {metric_display.get(metric, metric)} |"
        for comp in comparisons:
            if metric == "intention_completion":
                value = comp.intention_completion_improvement
            elif metric == "executable_actions":
                value = comp.executable_actions_improvement
            elif metric == "dialogue_lines":
                value = comp.dialogue_lines_improvement
            elif metric == "dialogue_branches":
                value = comp.dialogue_branches_improvement
            elif metric == "total_artifacts":
                value = comp.total_artifacts_improvement
            else:
                value = 0.0

            # Format with + sign for positive values
            sign = "+" if value > 0 else ""
            row += f" {sign}{value:.1f}% |"

        lines.append(row)

    return "\n".join(lines)


def compare_multiple_conditions(
    baseline: ExperimentMetrics,
    treatments: List[ExperimentMetrics],
) -> List[ConditionComparison]:
    """
    Compare multiple treatment conditions against a baseline.

    Args:
        baseline: Baseline condition metrics
        treatments: List of treatment condition metrics

    Returns:
        List of comparisons, one per treatment
    """
    return [compare_conditions(baseline, treatment) for treatment in treatments]


def summarize_comparisons(
    comparisons: List[ConditionComparison],
) -> Dict[str, Any]:
    """
    Generate summary statistics across multiple comparisons.

    Args:
        comparisons: List of condition comparisons

    Returns:
        Summary statistics dictionary
    """
    if not comparisons:
        return {"error": "No comparisons provided"}

    # Collect improvements for each metric
    intention_improvements = [c.intention_completion_improvement for c in comparisons]
    executable_improvements = [c.executable_actions_improvement for c in comparisons]
    dialogue_improvements = [c.dialogue_lines_improvement for c in comparisons]
    branch_improvements = [c.dialogue_branches_improvement for c in comparisons]
    artifact_improvements = [c.total_artifacts_improvement for c in comparisons]

    def stats(values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not values:
            return {"mean": 0, "min": 0, "max": 0}
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "num_comparisons": len(comparisons),
        "intention_completion": stats(intention_improvements),
        "executable_actions": stats(executable_improvements),
        "dialogue_lines": stats(dialogue_improvements),
        "dialogue_branches": stats(branch_improvements),
        "total_artifacts": stats(artifact_improvements),
        "best_overall": max(comparisons, key=lambda c: (
            c.intention_completion_improvement +
            c.dialogue_lines_improvement +
            c.total_artifacts_improvement
        ) / 3).treatment_name,
    }
