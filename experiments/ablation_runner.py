"""
Ablation Study Runner for TASK-014.

Systematically evaluates each feature and feature combination to determine
individual and combined contributions to performance improvements.

This runner integrates with the feature flag system (TASK-000) and uses all
the components built in TASK-005 through TASK-013:
- TASK-005: GPT-4 Model Integration (use_gpt4 flag)
- TASK-006: Full Context State Management (full_context flag)
- TASK-007: Verification Loop (verification_loop flag)
- TASK-009: Enhanced CoT Prompting (cot_enhancement flag)
- TASK-010: Dialogue Improvement (dialogue_improvement flag)
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature_flags import FeatureFlags, get_profile, PROFILES, CONDITION_GROUPS
from experiments.baseline_runner import load_rocstories_scenarios
from analysis.metrics import (
    calculate_experiment_metrics,
    ExperimentMetrics,
    ScenarioMetrics,
    load_scenarios_from_directory,
    print_metrics_report,
)
from analysis.comparison import (
    compare_conditions,
    compare_multiple_conditions,
    generate_comparison_table,
    summarize_comparisons,
    ConditionComparison,
)
from analysis.statistics import (
    compare_conditions_statistically,
    generate_statistical_report,
    perform_multiple_comparisons,
    StatisticalComparison,
)


@dataclass
class AblationCondition:
    """Configuration for a single ablation condition."""
    condition_id: str
    name: str
    description: str
    features: Dict[str, bool]
    profile: Optional[str] = None
    notes: str = ""

    def get_feature_flags(self) -> FeatureFlags:
        """Create FeatureFlags from condition configuration."""
        if self.profile and self.profile in PROFILES:
            return PROFILES[self.profile]
        return FeatureFlags(**self.features)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition_id": self.condition_id,
            "name": self.name,
            "description": self.description,
            "features": self.features,
            "profile": self.profile,
            "notes": self.notes,
        }


@dataclass
class ConditionResult:
    """Results for a single condition run."""
    condition: AblationCondition
    metrics: ExperimentMetrics
    timing: Dict[str, float]
    scenario_results: List[Dict[str, Any]]
    success: bool = True
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condition": self.condition.to_dict(),
            "metrics": self.metrics.to_dict(),
            "timing": self.timing,
            "scenario_results": self.scenario_results,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class AblationStudyResults:
    """Complete results from an ablation study."""
    study_name: str
    timestamp: str
    conditions_run: List[str]
    condition_results: Dict[str, ConditionResult]
    comparisons: Dict[str, ConditionComparison]
    summary: Dict[str, Any]
    statistical_comparisons: Dict[str, StatisticalComparison] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "study_name": self.study_name,
            "timestamp": self.timestamp,
            "conditions_run": self.conditions_run,
            "condition_results": {
                k: v.to_dict() for k, v in self.condition_results.items()
            },
            "comparisons": {
                k: v.to_dict() for k, v in self.comparisons.items()
            },
            "statistical_comparisons": {
                k: v.to_dict() for k, v in self.statistical_comparisons.items()
            },
            "summary": self.summary,
        }


class AblationRunner:
    """
    Runs ablation studies across multiple feature configurations.

    Uses the PROFILES and CONDITION_GROUPS from config/feature_flags.py
    directly (TASK-014 integration).

    Supports:
    - Running all conditions or specific subsets
    - Resuming from partial runs
    - Statistical comparison between conditions
    - Comprehensive result reporting
    """

    # Condition descriptions for reporting
    CONDITION_DESCRIPTIONS = {
        "C00": ("Baseline", "Original system (all features off)"),
        "C01": ("GPT-4 Only", "Only GPT-4 model enabled"),
        "C02": ("Full Context Only", "Only full context state management"),
        "C03": ("CoT Only", "Only enhanced Chain-of-Thought prompts"),
        "C04": ("Dialogue Only", "Only dialogue improvement"),
        "C05": ("GPT-4 + Context", "GPT-4 with full context state"),
        "C06": ("+ Verification", "GPT-4 + Context + Verification loop"),
        "C07": ("+ CoT", "GPT-4 + Context + Verification + CoT"),
        "C08": ("Full System", "All features enabled"),
        "C09": ("Full - Verification", "Full system minus verification loop"),
        "C10": ("Full - Context", "Full system minus full context"),
    }

    def __init__(
        self,
        output_dir: str = "experiments/results/ablation",
    ):
        """
        Initialize the ablation runner.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.conditions = self._build_conditions_from_profiles()

    def _build_conditions_from_profiles(self) -> Dict[str, AblationCondition]:
        """Build conditions from PROFILES in feature_flags.py."""
        conditions = {}

        # Get all condition IDs (C00-C10) from PROFILES
        condition_ids = [k for k in PROFILES.keys() if k.startswith("C")]

        for cond_id in condition_ids:
            flags = PROFILES[cond_id]
            name, description = self.CONDITION_DESCRIPTIONS.get(
                cond_id, (cond_id, "")
            )
            conditions[cond_id] = AblationCondition(
                condition_id=cond_id,
                name=name,
                description=description,
                features=flags.to_dict(),
                profile=cond_id,
                notes="",
            )

        return conditions

    def get_condition(self, condition_id: str) -> AblationCondition:
        """Get a specific condition by ID."""
        if condition_id not in self.conditions:
            raise ValueError(f"Unknown condition: {condition_id}")
        return self.conditions[condition_id]

    def get_conditions_in_group(self, group_name: str) -> List[str]:
        """Get condition IDs for a specific group."""
        if group_name not in CONDITION_GROUPS:
            raise ValueError(f"Unknown group: {group_name}. Available: {list(CONDITION_GROUPS.keys())}")
        return CONDITION_GROUPS[group_name]

    def run_condition(
        self,
        condition_id: str,
        n_scenarios: int = 20,
        random_state: int = 42,
        max_retries: int = 3,
        dry_run: bool = False,
    ) -> ConditionResult:
        """
        Run a single ablation condition.

        Args:
            condition_id: Condition identifier (e.g., "C00", "C01")
            n_scenarios: Number of scenarios to generate
            random_state: Random seed for reproducibility
            max_retries: Max retries per scenario on failure
            dry_run: If True, simulate without actual generation

        Returns:
            ConditionResult with metrics and timing
        """
        condition = self.get_condition(condition_id)
        flags = condition.get_feature_flags()

        print(f"\n{'='*70}")
        print(f"RUNNING CONDITION: {condition_id} - {condition.name}")
        print(f"Description: {condition.description}")
        print(f"Features: {flags.to_dict()}")
        print(f"{'='*70}\n")

        # Create output directory for this condition
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        condition_dir = self.output_dir / f"{condition_id}_{condition.name}_{timestamp}"
        condition_dir.mkdir(parents=True, exist_ok=True)

        # Save condition config
        config_data = {
            "condition": condition.to_dict(),
            "feature_flags": flags.to_dict(),
            "n_scenarios": n_scenarios,
            "random_state": random_state,
            "max_retries": max_retries,
            "timestamp": timestamp,
            "dry_run": dry_run,
        }
        with open(condition_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)

        if dry_run:
            # Return mock results for dry run
            return self._create_dry_run_result(condition, condition_dir)

        # Run actual experiment
        return self._run_experiment(
            condition=condition,
            flags=flags,
            n_scenarios=n_scenarios,
            random_state=random_state,
            max_retries=max_retries,
            output_dir=condition_dir,
        )

    def _run_experiment(
        self,
        condition: AblationCondition,
        flags: FeatureFlags,
        n_scenarios: int,
        random_state: int,
        max_retries: int,
        output_dir: Path,
    ) -> ConditionResult:
        """
        Execute the actual experiment for a condition.
        """
        from SocialScenarioGPT import generate_scenario

        # Load scenarios
        df = load_rocstories_scenarios(n_scenarios, random_state)

        # Track timing and results
        scenario_times = []
        scenario_results = []
        failed_scenarios = []
        generated_scenarios = []

        start_time = time.time()

        for idx, (_, row) in enumerate(df.iterrows()):
            scenario_description = " ".join([
                row['sentence1'], row['sentence2'], row['sentence3'],
                row['sentence4'], row['sentence5']
            ])
            scenario_name = f"ablation_{condition.condition_id}_{row['storytitle']}"

            print(f"\n[{idx+1}/{n_scenarios}] Generating: {scenario_name}")

            scenario_start = time.time()
            success = False

            for retry in range(max_retries):
                try:
                    # Set feature flags globally before generation
                    self._configure_features(flags)

                    generate_scenario(scenario_name, scenario_description)
                    success = True
                    break
                except Exception as e:
                    print(f"  Retry {retry+1}/{max_retries} failed: {e}")
                    continue

            scenario_time = time.time() - scenario_start
            scenario_times.append(scenario_time)

            if success:
                print(f"  Completed in {scenario_time/60:.2f} minutes")
                scenario_results.append({
                    "name": scenario_name,
                    "description": scenario_description,
                    "time_seconds": scenario_time,
                    "success": True,
                })
                generated_scenarios.append(scenario_name)
            else:
                print(f"  FAILED after {max_retries} retries")
                failed_scenarios.append(scenario_name)
                scenario_results.append({
                    "name": scenario_name,
                    "description": scenario_description,
                    "time_seconds": scenario_time,
                    "success": False,
                })

        total_time = time.time() - start_time

        # Save scenario results
        with open(output_dir / "scenario_results.json", "w") as f:
            json.dump(scenario_results, f, indent=2)

        # Load generated scenarios and compute metrics
        print("\n" + "-"*50)
        print("Computing metrics...")

        data_dir = Path(__file__).parent.parent / "Data"
        scenarios = self._load_condition_scenarios(
            data_dir,
            prefix=f"ablation_{condition.condition_id}_"
        )

        # Calculate metrics
        metrics = calculate_experiment_metrics(
            scenarios,
            experiment_name=condition.name,
            condition_id=condition.condition_id,
        )

        # Timing info
        timing = {
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "avg_time_per_scenario_seconds": (
                sum(scenario_times) / len(scenario_times) if scenario_times else 0
            ),
            "avg_time_per_scenario_minutes": (
                (sum(scenario_times) / len(scenario_times)) / 60 if scenario_times else 0
            ),
        }

        # Save metrics
        metrics_data = metrics.to_dict()
        metrics_data["timing"] = timing
        metrics_data["failed_scenarios"] = failed_scenarios

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Print summary
        print_metrics_report(metrics)
        print(f"\nTiming: {timing['total_time_minutes']:.2f} minutes total")
        print(f"Results saved to: {output_dir}")

        return ConditionResult(
            condition=condition,
            metrics=metrics,
            timing=timing,
            scenario_results=scenario_results,
            success=len(failed_scenarios) == 0,
            error_message="" if len(failed_scenarios) == 0 else f"{len(failed_scenarios)} scenarios failed",
        )

    def _configure_features(self, flags: FeatureFlags) -> None:
        """
        Configure global feature flags for the scenario generation pipeline.

        This sets up the environment so the generation pipeline uses the
        correct features for this condition (TASK-014 integration).
        """
        # Import and use the set_feature_flags function from SocialScenarioGPT
        from SocialScenarioGPT import set_feature_flags
        set_feature_flags(flags)

    def _load_condition_scenarios(
        self,
        data_dir: Path,
        prefix: str,
    ) -> List[Dict[str, Any]]:
        """Load scenarios generated for a specific condition."""
        scenarios = []

        for file_path in data_dir.glob("*.json"):
            if not file_path.name.startswith(prefix):
                continue

            try:
                with open(file_path) as f:
                    scenario = json.load(f)
                    if isinstance(scenario, dict) and "agents" in scenario:
                        scenarios.append(scenario)
            except (json.JSONDecodeError, IOError):
                continue

        return scenarios

    def _create_dry_run_result(
        self,
        condition: AblationCondition,
        output_dir: Path,
    ) -> ConditionResult:
        """Create mock result for dry run."""
        mock_metrics = ExperimentMetrics(
            experiment_name=condition.name,
            condition_id=condition.condition_id,
            total_scenarios=0,
        )

        return ConditionResult(
            condition=condition,
            metrics=mock_metrics,
            timing={
                "total_time_seconds": 0,
                "total_time_minutes": 0,
                "avg_time_per_scenario_seconds": 0,
                "avg_time_per_scenario_minutes": 0,
            },
            scenario_results=[],
            success=True,
            error_message="Dry run - no actual generation",
        )

    def run_ablation_study(
        self,
        conditions: Optional[List[str]] = None,
        group: Optional[str] = None,
        n_scenarios: int = 20,
        random_state: int = 42,
        max_retries: int = 3,
        dry_run: bool = False,
        resume_from: Optional[str] = None,
    ) -> AblationStudyResults:
        """
        Run the full ablation study.

        Args:
            conditions: List of condition IDs to run (None = all)
            group: Group name to run (overrides conditions)
            n_scenarios: Number of scenarios per condition
            random_state: Random seed for reproducibility
            max_retries: Max retries per scenario
            dry_run: If True, simulate without actual generation
            resume_from: Condition ID to resume from (skips earlier conditions)

        Returns:
            AblationStudyResults with all condition results and comparisons
        """
        # Determine which conditions to run
        if group:
            condition_ids = self.get_conditions_in_group(group)
        elif conditions:
            condition_ids = conditions
        else:
            condition_ids = list(self.conditions.keys())

        # Handle resume
        if resume_from:
            try:
                resume_idx = condition_ids.index(resume_from)
                condition_ids = condition_ids[resume_idx:]
                print(f"Resuming from condition {resume_from}")
            except ValueError:
                print(f"Warning: Resume condition {resume_from} not in list, starting from beginning")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = self.output_dir / f"study_{timestamp}"
        study_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"ABLATION STUDY")
        print(f"{'#'*70}")
        print(f"Conditions: {condition_ids}")
        print(f"Scenarios per condition: {n_scenarios}")
        print(f"Output directory: {study_dir}")
        print(f"{'#'*70}\n")

        # Save study config
        study_config = {
            "study_name": "SIA-LLM Ablation Study",
            "timestamp": timestamp,
            "conditions": condition_ids,
            "n_scenarios": n_scenarios,
            "random_state": random_state,
            "max_retries": max_retries,
            "dry_run": dry_run,
        }
        with open(study_dir / "study_config.json", "w") as f:
            json.dump(study_config, f, indent=2)

        # Run each condition
        condition_results: Dict[str, ConditionResult] = {}

        for cond_id in condition_ids:
            try:
                result = self.run_condition(
                    condition_id=cond_id,
                    n_scenarios=n_scenarios,
                    random_state=random_state,
                    max_retries=max_retries,
                    dry_run=dry_run,
                )
                condition_results[cond_id] = result

                # Save intermediate results
                with open(study_dir / f"{cond_id}_result.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            except Exception as e:
                print(f"ERROR running condition {cond_id}: {e}")
                condition_results[cond_id] = ConditionResult(
                    condition=self.get_condition(cond_id),
                    metrics=ExperimentMetrics(
                        experiment_name=cond_id,
                        condition_id=cond_id,
                    ),
                    timing={},
                    scenario_results=[],
                    success=False,
                    error_message=str(e),
                )

        # Generate comparisons (if baseline exists)
        comparisons: Dict[str, ConditionComparison] = {}
        statistical_comparisons: Dict[str, StatisticalComparison] = {}

        if "C00" in condition_results and condition_results["C00"].success:
            baseline_metrics = condition_results["C00"].metrics

            for cond_id, result in condition_results.items():
                if cond_id != "C00" and result.success:
                    # Basic comparison
                    comparison = compare_conditions(
                        baseline_metrics,
                        result.metrics,
                    )
                    comparisons[f"C00_vs_{cond_id}"] = comparison

                    # Statistical comparison with significance tests
                    stat_comparison = compare_conditions_statistically(
                        baseline_metrics,
                        result.metrics,
                    )
                    statistical_comparisons[f"C00_vs_{cond_id}"] = stat_comparison

        # Generate summary
        summary = self._generate_study_summary(condition_results, comparisons)

        # Create results object
        results = AblationStudyResults(
            study_name="SIA-LLM Ablation Study",
            timestamp=timestamp,
            conditions_run=condition_ids,
            condition_results=condition_results,
            comparisons=comparisons,
            summary=summary,
            statistical_comparisons=statistical_comparisons,
        )

        # Save final results
        with open(study_dir / "final_results.json", "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        # Generate and save report
        report = self.generate_report(results)
        with open(study_dir / "report.md", "w") as f:
            f.write(report)

        print(f"\n{'#'*70}")
        print("ABLATION STUDY COMPLETE")
        print(f"Results saved to: {study_dir}")
        print(f"{'#'*70}\n")

        return results

    def _generate_study_summary(
        self,
        condition_results: Dict[str, ConditionResult],
        comparisons: Dict[str, ConditionComparison],
    ) -> Dict[str, Any]:
        """Generate summary statistics for the study."""
        successful = [r for r in condition_results.values() if r.success]

        if not successful:
            return {"error": "No successful condition runs"}

        # Find best performing condition for each metric
        best_intention = max(
            successful,
            key=lambda r: r.metrics.intention_completion_rate
        )
        best_executable = max(
            successful,
            key=lambda r: r.metrics.executable_actions_rate
        )
        best_dialogue = max(
            successful,
            key=lambda r: r.metrics.mean_dialogue_lines
        )

        # Calculate total time
        total_time = sum(r.timing.get("total_time_minutes", 0) for r in successful)

        return {
            "total_conditions": len(condition_results),
            "successful_conditions": len(successful),
            "total_time_minutes": total_time,
            "best_performers": {
                "intention_completion": {
                    "condition": best_intention.condition.condition_id,
                    "name": best_intention.condition.name,
                    "rate": best_intention.metrics.intention_completion_rate,
                },
                "executable_actions": {
                    "condition": best_executable.condition.condition_id,
                    "name": best_executable.condition.name,
                    "rate": best_executable.metrics.executable_actions_rate,
                },
                "dialogue_lines": {
                    "condition": best_dialogue.condition.condition_id,
                    "name": best_dialogue.condition.name,
                    "mean": best_dialogue.metrics.mean_dialogue_lines,
                },
            },
            "comparison_summary": (
                summarize_comparisons(list(comparisons.values()))
                if comparisons else {}
            ),
        }

    def generate_report(self, results: AblationStudyResults) -> str:
        """Generate a markdown report of the study results."""
        lines = [
            f"# {results.study_name}",
            "",
            f"**Generated:** {results.timestamp}",
            "",
            "## Overview",
            "",
            f"- **Conditions Run:** {len(results.conditions_run)}",
            f"- **Successful:** {results.summary.get('successful_conditions', 0)}",
            f"- **Total Time:** {results.summary.get('total_time_minutes', 0):.1f} minutes",
            "",
            "## Conditions",
            "",
            "| ID | Name | Description |",
            "|---|---|---|",
        ]

        for cond_id in results.conditions_run:
            if cond_id in results.condition_results:
                cond = results.condition_results[cond_id].condition
                status = "SUCCESS" if results.condition_results[cond_id].success else "FAILED"
                lines.append(f"| {cond_id} | {cond.name} | {cond.description} ({status}) |")

        lines.extend([
            "",
            "## Key Metrics",
            "",
            "| Condition | Intention Completion | Executable Actions | Dialogue Lines |",
            "|---|---|---|---|",
        ])

        for cond_id, result in results.condition_results.items():
            if result.success:
                m = result.metrics
                lines.append(
                    f"| {cond_id} ({result.condition.name}) | "
                    f"{m.intention_completion_rate:.1%} | "
                    f"{m.executable_actions_rate:.1%} | "
                    f"{m.mean_dialogue_lines:.1f} |"
                )

        # Add comparison table if available
        if results.comparisons:
            lines.extend([
                "",
                "## Comparisons vs Baseline (C00)",
                "",
                generate_comparison_table(list(results.comparisons.values())),
            ])

        # Add statistical analysis section
        if results.statistical_comparisons:
            lines.extend([
                "",
                "## Statistical Analysis",
                "",
                "### Significance Tests (Independent t-tests, α=0.05)",
                "",
                "| Condition | Metric | Difference | t-stat | p-value | Significant |",
                "|-----------|--------|------------|--------|---------|-------------|",
            ])

            for key, stat_comp in results.statistical_comparisons.items():
                cond_name = key.replace("C00_vs_", "")
                for test in stat_comp.tests:
                    sig = "Yes*" if test.significant else "No"
                    lines.append(
                        f"| {cond_name} | {test.metric_name} | "
                        f"{test.difference:+.3f} | "
                        f"{test.t_statistic:.2f} | "
                        f"{test.p_value:.4f} | {sig} |"
                    )

            lines.extend([
                "",
                "### Effect Sizes (Cohen's d)",
                "",
                "| Condition | Metric | Cohen's d | Interpretation |",
                "|-----------|--------|-----------|----------------|",
            ])

            for key, stat_comp in results.statistical_comparisons.items():
                cond_name = key.replace("C00_vs_", "")
                for effect in stat_comp.effect_sizes:
                    lines.append(
                        f"| {cond_name} | {effect.metric_name} | "
                        f"{effect.cohens_d:+.3f} | {effect.interpretation} |"
                    )

            # Summary of significant findings
            lines.extend([
                "",
                "### Significant Improvements Summary",
                "",
            ])

            for key, stat_comp in results.statistical_comparisons.items():
                cond_name = key.replace("C00_vs_", "")
                sig_improvements = stat_comp.get_significant_improvements()
                large_effects = stat_comp.get_large_effects()

                if sig_improvements or large_effects:
                    lines.append(f"**{cond_name}:**")
                    if sig_improvements:
                        lines.append(f"- Significant improvements: {', '.join(sig_improvements)}")
                    if large_effects:
                        lines.append(f"- Large effect sizes: {', '.join(large_effects)}")
                    lines.append("")

        # Best performers
        if "best_performers" in results.summary:
            bp = results.summary["best_performers"]
            lines.extend([
                "",
                "## Best Performers",
                "",
                f"- **Intention Completion:** {bp['intention_completion']['name']} "
                f"({bp['intention_completion']['rate']:.1%})",
                f"- **Executable Actions:** {bp['executable_actions']['name']} "
                f"({bp['executable_actions']['rate']:.1%})",
                f"- **Dialogue Lines:** {bp['dialogue_lines']['name']} "
                f"({bp['dialogue_lines']['mean']:.1f} lines)",
            ])

        # Feature contribution analysis
        lines.extend([
            "",
            "## Feature Contribution Analysis",
            "",
            "### Individual Feature Contributions (vs Baseline)",
            "",
        ])

        # Map single-feature conditions
        single_feature_conditions = {
            "C01": "GPT-4 Model",
            "C02": "Full Context",
            "C03": "CoT Enhancement",
            "C04": "Dialogue Improvement",
        }

        for cond_id, feature_name in single_feature_conditions.items():
            if cond_id in results.condition_results and results.condition_results[cond_id].success:
                metrics = results.condition_results[cond_id].metrics
                baseline_metrics = results.condition_results.get("C00")
                if baseline_metrics and baseline_metrics.success:
                    baseline = baseline_metrics.metrics
                    intent_diff = metrics.intention_completion_rate - baseline.intention_completion_rate
                    exec_diff = metrics.executable_actions_rate - baseline.executable_actions_rate
                    dial_diff = metrics.mean_dialogue_lines - baseline.mean_dialogue_lines

                    lines.extend([
                        f"**{feature_name} ({cond_id}):**",
                        f"- Intention Completion: {intent_diff:+.1%}",
                        f"- Executable Actions: {exec_diff:+.1%}",
                        f"- Dialogue Lines: {dial_diff:+.1f}",
                        "",
                    ])

        return "\n".join(lines)

    def load_existing_results(self, study_dir: str) -> Optional[AblationStudyResults]:
        """Load results from a previous study run."""
        results_path = Path(study_dir) / "final_results.json"

        if not results_path.exists():
            return None

        with open(results_path) as f:
            data = json.load(f)

        # Reconstruct the results object
        # (Simplified - full reconstruction would need more work)
        return data  # Return raw dict for now

    def analyze_existing_data(
        self,
        data_dir: str = "Data",
        prefix_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, ExperimentMetrics]:
        """
        Analyze existing scenario data without generating new scenarios.

        Useful for analyzing previously generated scenarios.

        Args:
            data_dir: Directory containing scenario JSON files
            prefix_mapping: Map of condition_id -> filename prefix

        Returns:
            Dictionary of condition_id -> ExperimentMetrics
        """
        data_path = Path(__file__).parent.parent / data_dir

        if prefix_mapping is None:
            # Default: look for ablation_{condition_id}_ prefix
            prefix_mapping = {
                cond_id: f"ablation_{cond_id}_"
                for cond_id in self.conditions.keys()
            }

        results = {}

        for cond_id, prefix in prefix_mapping.items():
            scenarios = self._load_condition_scenarios(data_path, prefix)

            if scenarios:
                metrics = calculate_experiment_metrics(
                    scenarios,
                    experiment_name=self.conditions[cond_id].name,
                    condition_id=cond_id,
                )
                results[cond_id] = metrics
                print(f"{cond_id}: {len(scenarios)} scenarios found")
            else:
                print(f"{cond_id}: No scenarios found with prefix '{prefix}'")

        return results


def main():
    """Main entry point for ablation study."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--condition", "-c",
        type=str,
        help="Run specific condition (e.g., C00, C01)"
    )
    parser.add_argument(
        "--group", "-g",
        type=str,
        help="Run condition group (e.g., single_feature, incremental)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all conditions"
    )
    parser.add_argument(
        "--n-scenarios", "-n",
        type=int,
        default=20,
        help="Number of scenarios per condition (default: 20)"
    )
    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without actual generation"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from specific condition ID"
    )
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="Analyze existing data instead of generating new"
    )
    parser.add_argument(
        "--list-conditions",
        action="store_true",
        help="List all available conditions"
    )

    args = parser.parse_args()

    runner = AblationRunner()

    if args.list_conditions:
        print("\nAvailable Conditions:")
        print("-" * 70)
        for cond_id, cond in runner.conditions.items():
            print(f"{cond_id}: {cond.name}")
            print(f"    {cond.description}")
            print(f"    Features: {cond.features}")
            print()
        return

    if args.analyze_existing:
        print("\nAnalyzing existing data...")
        results = runner.analyze_existing_data()
        for cond_id, metrics in results.items():
            print_metrics_report(metrics)
        return

    if args.condition:
        # Run single condition
        runner.run_condition(
            condition_id=args.condition,
            n_scenarios=args.n_scenarios,
            random_state=args.random_state,
            dry_run=args.dry_run,
        )
    elif args.group or args.all:
        # Run group or all conditions
        runner.run_ablation_study(
            group=args.group if args.group else None,
            conditions=None if args.group else list(runner.conditions.keys()),
            n_scenarios=args.n_scenarios,
            random_state=args.random_state,
            dry_run=args.dry_run,
            resume_from=args.resume_from,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
