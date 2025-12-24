"""
Statistical analysis utilities for ablation studies.

Provides paired t-tests, effect size calculations (Cohen's d),
confidence intervals, and significance testing.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from analysis.metrics import ExperimentMetrics, ScenarioMetrics


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    metric_name: str
    baseline_mean: float
    treatment_mean: float
    difference: float
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    significant: bool
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "metric_name": self.metric_name,
            "baseline_mean": round(self.baseline_mean, 4),
            "treatment_mean": round(self.treatment_mean, 4),
            "difference": round(self.difference, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "degrees_of_freedom": self.degrees_of_freedom,
            "significant": self.significant,
            "alpha": self.alpha,
        }


@dataclass
class EffectSize:
    """Effect size calculation result."""
    metric_name: str
    cohens_d: float
    interpretation: str  # "negligible", "small", "medium", "large"
    baseline_std: float
    treatment_std: float
    pooled_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "cohens_d": round(self.cohens_d, 4),
            "interpretation": self.interpretation,
            "baseline_std": round(self.baseline_std, 4),
            "treatment_std": round(self.treatment_std, 4),
            "pooled_std": round(self.pooled_std, 4),
        }


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    metric_name: str
    mean: float
    lower: float
    upper: float
    confidence_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "mean": round(self.mean, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence_level": self.confidence_level,
        }


@dataclass
class StatisticalComparison:
    """Complete statistical comparison between two conditions."""
    baseline_name: str
    treatment_name: str
    n_baseline: int
    n_treatment: int
    tests: List[StatisticalTest] = field(default_factory=list)
    effect_sizes: List[EffectSize] = field(default_factory=list)
    confidence_intervals: Dict[str, ConfidenceInterval] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comparison": f"{self.treatment_name} vs {self.baseline_name}",
            "sample_sizes": {
                "baseline": self.n_baseline,
                "treatment": self.n_treatment,
            },
            "tests": [t.to_dict() for t in self.tests],
            "effect_sizes": [e.to_dict() for e in self.effect_sizes],
            "confidence_intervals": {
                k: v.to_dict() for k, v in self.confidence_intervals.items()
            },
        }

    def get_significant_improvements(self) -> List[str]:
        """Get list of metrics with significant improvements."""
        return [
            t.metric_name for t in self.tests
            if t.significant and t.difference > 0
        ]

    def get_large_effects(self) -> List[str]:
        """Get list of metrics with large effect sizes."""
        return [
            e.metric_name for e in self.effect_sizes
            if e.interpretation in ("large", "medium")
        ]


def calculate_mean(values: List[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_std(values: List[float], mean: Optional[float] = None) -> float:
    """Calculate sample standard deviation."""
    if len(values) < 2:
        return 0.0
    if mean is None:
        mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def calculate_pooled_std(
    std1: float,
    n1: int,
    std2: float,
    n2: int,
) -> float:
    """
    Calculate pooled standard deviation for two groups.

    Uses the formula for pooled variance:
    s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1 + n2 - 2))
    """
    if n1 + n2 <= 2:
        return 0.0

    pooled_var = (
        ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) /
        (n1 + n2 - 2)
    )
    return math.sqrt(pooled_var)


def calculate_cohens_d(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
) -> Tuple[float, str]:
    """
    Calculate Cohen's d effect size.

    Args:
        mean1: Mean of group 1 (baseline)
        std1: Standard deviation of group 1
        n1: Sample size of group 1
        mean2: Mean of group 2 (treatment)
        std2: Standard deviation of group 2
        n2: Sample size of group 2

    Returns:
        Tuple of (Cohen's d value, interpretation string)

    Interpretation thresholds (Cohen, 1988):
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    pooled = calculate_pooled_std(std1, n1, std2, n2)

    if pooled == 0:
        return (0.0, "negligible")

    d = (mean2 - mean1) / pooled

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return (d, interpretation)


def paired_t_test(
    baseline_values: List[float],
    treatment_values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, int, bool]:
    """
    Perform a paired t-test for dependent samples.

    Args:
        baseline_values: Values from baseline condition
        treatment_values: Values from treatment condition
        alpha: Significance level

    Returns:
        Tuple of (t-statistic, p-value, degrees of freedom, significant)

    Note: For simplicity, uses approximation for p-value calculation.
    For production use, consider scipy.stats.ttest_rel
    """
    if len(baseline_values) != len(treatment_values):
        raise ValueError("Paired t-test requires equal length arrays")

    n = len(baseline_values)
    if n < 2:
        return (0.0, 1.0, 0, False)

    # Calculate differences
    differences = [t - b for b, t in zip(baseline_values, treatment_values)]

    # Mean and std of differences
    mean_diff = calculate_mean(differences)
    std_diff = calculate_std(differences, mean_diff)

    if std_diff == 0:
        return (0.0, 1.0, n - 1, False)

    # t-statistic
    se = std_diff / math.sqrt(n)
    t_stat = mean_diff / se
    df = n - 1

    # Approximate p-value using t-distribution approximation
    # For a proper implementation, use scipy.stats.t.sf
    p_value = _approximate_t_pvalue(abs(t_stat), df)

    significant = p_value < alpha

    return (t_stat, p_value, df, significant)


def independent_t_test(
    group1_values: List[float],
    group2_values: List[float],
    alpha: float = 0.05,
) -> Tuple[float, float, int, bool]:
    """
    Perform an independent samples t-test.

    Args:
        group1_values: Values from group 1 (baseline)
        group2_values: Values from group 2 (treatment)
        alpha: Significance level

    Returns:
        Tuple of (t-statistic, p-value, degrees of freedom, significant)
    """
    n1 = len(group1_values)
    n2 = len(group2_values)

    if n1 < 2 or n2 < 2:
        return (0.0, 1.0, 0, False)

    mean1 = calculate_mean(group1_values)
    mean2 = calculate_mean(group2_values)
    std1 = calculate_std(group1_values, mean1)
    std2 = calculate_std(group2_values, mean2)

    # Pooled standard error
    se = math.sqrt(std1**2 / n1 + std2**2 / n2)

    if se == 0:
        return (0.0, 1.0, n1 + n2 - 2, False)

    # t-statistic
    t_stat = (mean2 - mean1) / se

    # Welch's degrees of freedom (for unequal variances)
    df_num = (std1**2 / n1 + std2**2 / n2) ** 2
    df_denom = (
        (std1**2 / n1)**2 / (n1 - 1) +
        (std2**2 / n2)**2 / (n2 - 1)
    )
    df = int(df_num / df_denom) if df_denom > 0 else n1 + n2 - 2

    # Approximate p-value
    p_value = _approximate_t_pvalue(abs(t_stat), df)

    significant = p_value < alpha

    return (t_stat, p_value, df, significant)


def _approximate_t_pvalue(t_stat: float, df: int) -> float:
    """
    Approximate two-tailed p-value for t-distribution.

    Uses a simplified approximation. For production, use scipy.stats.t.sf

    This approximation is based on:
    - For df > 30, t-distribution is approximately normal
    - For smaller df, uses a correction factor
    """
    if df <= 0:
        return 1.0

    # For large df, approximate with normal distribution
    if df > 30:
        # Normal approximation
        z = t_stat
        p = 2 * (1 - _normal_cdf(abs(z)))
    else:
        # Rough approximation for smaller df
        # This is a simplified heuristic - not exact
        correction = 1 + 1 / (4 * df)
        z = t_stat / correction
        p = 2 * (1 - _normal_cdf(abs(z)))

    return min(max(p, 0.0), 1.0)


def _normal_cdf(z: float) -> float:
    """
    Approximate standard normal CDF using error function approximation.
    """
    # Approximation using error function identity:
    # CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def _erf(x: float) -> float:
    """
    Approximate error function using Horner form of polynomial.

    Abramowitz and Stegun approximation (equation 7.1.26)
    Maximum error: 1.5e-7
    """
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # Constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y


def calculate_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate confidence interval for a set of values.

    Args:
        values: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        ConfidenceInterval with mean, lower, and upper bounds
    """
    n = len(values)
    if n < 2:
        mean = calculate_mean(values) if values else 0.0
        return ConfidenceInterval(
            metric_name="",
            mean=mean,
            lower=mean,
            upper=mean,
            confidence_level=confidence,
        )

    mean = calculate_mean(values)
    std = calculate_std(values, mean)
    se = std / math.sqrt(n)

    # Critical value approximation (for 95% CI, z ≈ 1.96 for large n)
    # For smaller n, would need t-distribution critical values
    if confidence == 0.95:
        if n > 30:
            z = 1.96
        else:
            # Approximate t critical value for smaller samples
            z = 2.0 + 0.5 / n
    elif confidence == 0.99:
        z = 2.576 if n > 30 else 2.8
    else:
        z = 1.96  # Default to 95%

    margin = z * se

    return ConfidenceInterval(
        metric_name="",
        mean=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
    )


def compare_conditions_statistically(
    baseline: ExperimentMetrics,
    treatment: ExperimentMetrics,
    alpha: float = 0.05,
) -> StatisticalComparison:
    """
    Perform comprehensive statistical comparison between two conditions.

    Args:
        baseline: Baseline condition metrics
        treatment: Treatment condition metrics
        alpha: Significance level for tests

    Returns:
        StatisticalComparison with all tests and effect sizes
    """
    comparison = StatisticalComparison(
        baseline_name=baseline.experiment_name,
        treatment_name=treatment.experiment_name,
        n_baseline=baseline.total_scenarios,
        n_treatment=treatment.total_scenarios,
    )

    # Extract per-scenario values for statistical tests
    baseline_scenarios = baseline.scenario_metrics
    treatment_scenarios = treatment.scenario_metrics

    # Define metrics to compare
    metric_extractors = {
        "intention_completion_rate": lambda s: s.intention_completion_rate,
        "executable_actions_rate": lambda s: s.executable_actions_rate,
        "dialogue_lines": lambda s: float(s.dialogue_lines),
        "total_artifacts": lambda s: float(s.total_artifacts),
        "agents": lambda s: float(s.agents),
        "beliefs": lambda s: float(s.beliefs),
        "actions": lambda s: float(s.actions),
    }

    for metric_name, extractor in metric_extractors.items():
        baseline_values = [extractor(s) for s in baseline_scenarios]
        treatment_values = [extractor(s) for s in treatment_scenarios]

        if not baseline_values or not treatment_values:
            continue

        # Calculate means and stds
        baseline_mean = calculate_mean(baseline_values)
        treatment_mean = calculate_mean(treatment_values)
        baseline_std = calculate_std(baseline_values, baseline_mean)
        treatment_std = calculate_std(treatment_values, treatment_mean)

        # Independent t-test (scenarios are independent between conditions)
        t_stat, p_value, df, significant = independent_t_test(
            baseline_values,
            treatment_values,
            alpha=alpha,
        )

        test = StatisticalTest(
            test_name="independent_t_test",
            metric_name=metric_name,
            baseline_mean=baseline_mean,
            treatment_mean=treatment_mean,
            difference=treatment_mean - baseline_mean,
            t_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            significant=significant,
            alpha=alpha,
        )
        comparison.tests.append(test)

        # Effect size
        d, interpretation = calculate_cohens_d(
            baseline_mean, baseline_std, len(baseline_values),
            treatment_mean, treatment_std, len(treatment_values),
        )
        pooled = calculate_pooled_std(
            baseline_std, len(baseline_values),
            treatment_std, len(treatment_values),
        )

        effect = EffectSize(
            metric_name=metric_name,
            cohens_d=d,
            interpretation=interpretation,
            baseline_std=baseline_std,
            treatment_std=treatment_std,
            pooled_std=pooled,
        )
        comparison.effect_sizes.append(effect)

        # Confidence intervals
        baseline_ci = calculate_confidence_interval(baseline_values)
        baseline_ci.metric_name = f"{metric_name}_baseline"
        treatment_ci = calculate_confidence_interval(treatment_values)
        treatment_ci.metric_name = f"{metric_name}_treatment"

        comparison.confidence_intervals[f"{metric_name}_baseline"] = baseline_ci
        comparison.confidence_intervals[f"{metric_name}_treatment"] = treatment_ci

    return comparison


def generate_statistical_report(
    comparisons: List[StatisticalComparison],
) -> str:
    """
    Generate a markdown report of statistical analyses.

    Args:
        comparisons: List of statistical comparisons

    Returns:
        Markdown formatted report
    """
    lines = [
        "# Statistical Analysis Report",
        "",
    ]

    for comp in comparisons:
        lines.extend([
            f"## {comp.treatment_name} vs {comp.baseline_name}",
            "",
            f"Sample sizes: Baseline n={comp.n_baseline}, Treatment n={comp.n_treatment}",
            "",
            "### Significance Tests",
            "",
            "| Metric | Baseline | Treatment | Diff | t | p | Sig |",
            "|--------|----------|-----------|------|---|---|-----|",
        ])

        for test in comp.tests:
            sig_mark = "*" if test.significant else ""
            lines.append(
                f"| {test.metric_name} | "
                f"{test.baseline_mean:.3f} | "
                f"{test.treatment_mean:.3f} | "
                f"{test.difference:+.3f} | "
                f"{test.t_statistic:.2f} | "
                f"{test.p_value:.4f} | "
                f"{sig_mark} |"
            )

        lines.extend([
            "",
            "### Effect Sizes (Cohen's d)",
            "",
            "| Metric | Cohen's d | Interpretation |",
            "|--------|-----------|----------------|",
        ])

        for effect in comp.effect_sizes:
            lines.append(
                f"| {effect.metric_name} | "
                f"{effect.cohens_d:+.3f} | "
                f"{effect.interpretation} |"
            )

        lines.extend([
            "",
            f"**Significant improvements:** {', '.join(comp.get_significant_improvements()) or 'None'}",
            "",
            f"**Large effects:** {', '.join(comp.get_large_effects()) or 'None'}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def perform_multiple_comparisons(
    baseline: ExperimentMetrics,
    treatments: List[ExperimentMetrics],
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> List[StatisticalComparison]:
    """
    Perform multiple comparisons with correction for multiple testing.

    Args:
        baseline: Baseline condition metrics
        treatments: List of treatment conditions
        alpha: Base significance level
        correction: Correction method ("bonferroni", "none")

    Returns:
        List of statistical comparisons with corrected alpha
    """
    n_comparisons = len(treatments)

    if correction == "bonferroni":
        adjusted_alpha = alpha / n_comparisons
    else:
        adjusted_alpha = alpha

    comparisons = []
    for treatment in treatments:
        comp = compare_conditions_statistically(
            baseline,
            treatment,
            alpha=adjusted_alpha,
        )
        comparisons.append(comp)

    return comparisons
