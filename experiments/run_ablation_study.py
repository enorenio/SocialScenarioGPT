#!/usr/bin/env python3
"""
TASK-015: Ablation Study Execution Script

This script runs the complete ablation study with all 11 conditions (C00-C10).
It follows experimental best practices:
1. Uses the SAME 40 curated scenarios for all conditions (fair comparison)
2. Reproducible random seeds
3. Checkpointing for resumability
4. Comprehensive logging
5. Cost tracking
6. Statistical analysis

Usage:
    # Full study - all 11 conditions × 40 scenarios (recommended: run overnight)
    python experiments/run_ablation_study.py

    # Quick test - 4 conditions × 5 scenarios (verify setup works)
    python experiments/run_ablation_study.py --quick

    # Single condition test
    python experiments/run_ablation_study.py --condition C08 --n-scenarios 5

    # Resume from checkpoint
    python experiments/run_ablation_study.py --resume

    # Dry run (no API calls)
    python experiments/run_ablation_study.py --dry-run

    # Analyze existing results only
    python experiments/run_ablation_study.py --analyze-only
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature_flags import FeatureFlags, PROFILES, CONDITION_GROUPS
from experiments.ablation_runner import AblationRunner, AblationStudyResults
from analysis.metrics import (
    calculate_experiment_metrics,
    load_scenarios_from_directory,
)
from analysis.comparison import (
    compare_conditions,
    generate_comparison_table,
)
from analysis.statistics import compare_conditions_statistically


# Study configuration
STUDY_CONFIG = {
    "name": "SIA-LLM Ablation Study",
    "version": "1.0",
    "random_seed": 42,
    "max_retries": 3,

    # Default curated scenarios for fair comparison
    "curated_scenarios_path": "data_utils/rocstories_scenarios.json",
    "total_curated_scenarios": 43,  # Must match baseline for valid comparison

    # Condition order (baseline first, then single features, then combinations)
    "condition_order": [
        "C00",  # Baseline
        "C01",  # Upgraded model only (GPT-5 Nano)
        "C02",  # Full context only
        "C03",  # CoT only
        "C04",  # Dialogue only
        "C05",  # Upgraded model + context
        "C06",  # + verification
        "C07",  # + CoT
        "C08",  # Full system
        "C09",  # Full - verification
        "C10",  # Full - context
    ],

    # Quick test uses subset of conditions AND scenarios
    "quick_conditions": ["C00", "C01", "C03", "C08"],
    "quick_n_scenarios": 5,
}

# Dataset presets
DATASET_PATHS = {
    "curated": "data_utils/rocstories_scenarios.json",
    "paper_2023": "data_utils/rocstories_scenarios_paper_2023.json",
}


def setup_output_directory(base_dir: str = "experiments/results/ablation_study") -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(output_dir: Path, completed: list, in_progress: str = None):
    """Save checkpoint for resumability."""
    checkpoint = {
        "completed_conditions": completed,
        "in_progress": in_progress,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir: Path) -> dict:
    """Load checkpoint if exists."""
    checkpoint_file = output_dir / "checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return json.load(f)
    return {"completed_conditions": [], "in_progress": None}


def find_condition_dir(output_dir: Path, condition_id: str) -> Path:
    """Find the most recent condition directory for a given condition."""
    matches = sorted(
        [d for d in output_dir.glob(f"{condition_id}_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def find_latest_study_dir(base_dir: str = "experiments/results/ablation_study") -> Path:
    """Find the most recent study directory for resuming."""
    base = Path(base_dir)
    if not base.exists():
        return None

    study_dirs = sorted(base.glob("study_*"), reverse=True)
    for d in study_dirs:
        if (d / "checkpoint.json").exists():
            return d
    return study_dirs[0] if study_dirs else None


def load_curated_scenarios(n_scenarios: int = None, scenarios_path: str = None) -> list:
    """
    Load curated scenarios from the original paper.

    These are the same scenarios used in baseline, ensuring fair comparison.
    """
    scenarios_path = Path(scenarios_path or STUDY_CONFIG["curated_scenarios_path"])
    if not scenarios_path.exists():
        raise FileNotFoundError(
            f"Curated scenarios not found at {scenarios_path}. "
            "Run TASK-004 first to prepare the dataset."
        )

    with open(scenarios_path) as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    if n_scenarios and n_scenarios < len(scenarios):
        # For quick tests, take first N (deterministic)
        scenarios = scenarios[:n_scenarios]

    return scenarios


def estimate_cost_and_time(n_scenarios: int, conditions: list) -> dict:
    """Estimate total cost and time for the study."""
    # Rough estimates based on typical scenario generation
    # GPT-3.5: ~$0.01 per scenario, ~2 min per scenario
    # GPT-5-nano: ~$0.02 per scenario (much cheaper than GPT-4o), ~2 min per scenario

    gpt35_conditions = [c for c in conditions if not PROFILES[c].use_upgraded_model]
    upgraded_conditions = [c for c in conditions if PROFILES[c].use_upgraded_model]

    n_gpt35 = len(gpt35_conditions) * n_scenarios
    n_upgraded = len(upgraded_conditions) * n_scenarios

    # GPT-5-nano is very cheap: ~$0.05/1M input, $0.40/1M output
    # Typical scenario uses ~50K tokens total -> ~$0.02 per scenario
    cost_low = n_gpt35 * 0.01 + n_upgraded * 0.01
    cost_high = n_gpt35 * 0.02 + n_upgraded * 0.03

    # GPT-5-nano is fast, similar to GPT-3.5
    time_min = (n_gpt35 * 2 + n_upgraded * 2) / 60  # hours
    time_max = (n_gpt35 * 3 + n_upgraded * 3) / 60  # hours

    return {
        "total_scenarios": (len(gpt35_conditions) + len(upgraded_conditions)) * n_scenarios,
        "gpt35_scenarios": n_gpt35,
        "gpt5_nano_scenarios": n_upgraded,
        "estimated_cost_low": cost_low,
        "estimated_cost_high": cost_high,
        "estimated_time_hours_low": time_min,
        "estimated_time_hours_high": time_max,
    }


def print_study_plan(conditions: list, n_scenarios: int, dry_run: bool, is_full_study: bool):
    """Print the study plan before execution."""
    estimates = estimate_cost_and_time(n_scenarios, conditions)

    print("\n" + "=" * 70)
    print("SIA-LLM ABLATION STUDY EXECUTION PLAN")
    print("=" * 70)

    if is_full_study:
        print("\n*** FULL STUDY MODE ***")
        print(f"Using all {n_scenarios} curated scenarios (same as baseline)")
    else:
        print("\n*** QUICK TEST MODE ***")
        print(f"Using {n_scenarios} of {STUDY_CONFIG['total_curated_scenarios']} scenarios")

    print(f"\nConditions to run: {len(conditions)}")
    for c in conditions:
        flags = PROFILES[c]
        features = flags.enabled_features() or "Baseline"
        print(f"  {c}: {features}")

    print(f"\nScenarios per condition: {n_scenarios}")
    print(f"Total API calls: {estimates['total_scenarios']} scenario generations")
    print(f"  - GPT-3.5 Turbo: {estimates['gpt35_scenarios']}")
    print(f"  - GPT-5 Nano: {estimates['gpt5_nano_scenarios']}")

    print(f"\nEstimated cost: ${estimates['estimated_cost_low']:.2f} - ${estimates['estimated_cost_high']:.2f}")
    print(f"Estimated time: {estimates['estimated_time_hours_low']:.1f} - {estimates['estimated_time_hours_high']:.1f} hours")

    if dry_run:
        print("\n*** DRY RUN MODE - No API calls will be made ***")

    print("=" * 70 + "\n")


def run_study(
    conditions: list,
    n_scenarios: int,
    output_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
    rerun_conditions: list = None,
    scenarios: list = None,
):
    """Run the ablation study."""
    runner = AblationRunner(output_dir=str(output_dir))

    # Load checkpoint if resuming
    checkpoint = load_checkpoint(output_dir) if resume else {"completed_conditions": []}
    completed = set(checkpoint["completed_conditions"])
    if resume and rerun_conditions:
        completed.difference_update(rerun_conditions)

    # Filter out already completed conditions
    remaining = [c for c in conditions if c not in completed]

    if not remaining:
        print("All conditions already completed!")
        return

    print(f"\nStarting study with {len(remaining)} conditions...")
    if completed:
        print(f"  (Resuming: {len(completed)} conditions already done)")

    results = {}
    start_time = time.time()

    for i, condition_id in enumerate(remaining):
        print(f"\n{'#' * 70}")
        print(f"# Condition {i+1}/{len(remaining)}: {condition_id}")
        print(f"{'#' * 70}")

        save_checkpoint(output_dir, list(completed), in_progress=condition_id)

        try:
            condition_dir = find_condition_dir(output_dir, condition_id) if resume else None
            result = runner.run_condition(
                condition_id=condition_id,
                n_scenarios=n_scenarios,
                random_state=STUDY_CONFIG["random_seed"],
                max_retries=STUDY_CONFIG["max_retries"],
                dry_run=dry_run,
                condition_dir=condition_dir,
                scenarios=scenarios,
            )
            results[condition_id] = result
            if result.success:
                completed.add(condition_id)
            else:
                print(
                    f"Condition {condition_id} has failed scenarios; "
                    "not marking as completed (resume will retry)."
                )

            # Save intermediate results
            save_checkpoint(output_dir, list(completed), in_progress=None)

            # Print progress
            elapsed = (time.time() - start_time) / 3600
            remaining_count = len(remaining) - i - 1
            if i > 0:
                avg_time = elapsed / (i + 1)
                eta = avg_time * remaining_count
                print(f"\nProgress: {len(completed)}/{len(conditions)} conditions")
                print(f"Elapsed: {elapsed:.1f}h, ETA: {eta:.1f}h")

        except KeyboardInterrupt:
            print("\n\nStudy interrupted! Progress saved.")
            print(f"Resume with: python {__file__} --resume")
            save_checkpoint(output_dir, list(completed), in_progress=None)
            sys.exit(1)
        except Exception as e:
            print(f"\nError in condition {condition_id}: {e}")
            print("Continuing with next condition...")
            save_checkpoint(output_dir, list(completed), in_progress=None)

    total_time = (time.time() - start_time) / 3600
    print(f"\n\nStudy completed in {total_time:.1f} hours")

    return results


def analyze_results(output_dir: Path):
    """Analyze and report results from completed study."""
    print("\n" + "=" * 70)
    print("ANALYZING RESULTS")
    print("=" * 70)

    # Find all condition directories
    condition_dirs = {}
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("C"):
            cond_id = d.name.split("_")[0]
            condition_dirs[cond_id] = d

    if not condition_dirs:
        print("No results found to analyze.")
        return

    print(f"\nFound {len(condition_dirs)} conditions: {sorted(condition_dirs.keys())}")

    # Calculate metrics for each condition
    condition_metrics = {}
    for cond_id, cond_dir in sorted(condition_dirs.items()):
        data_dir = cond_dir / "data"
        if data_dir.exists():
            scenarios = load_scenarios_from_directory(str(data_dir))
            if scenarios:
                metrics = calculate_experiment_metrics(scenarios, cond_id)
                condition_metrics[cond_id] = metrics
                print(f"\n{cond_id}: {len(scenarios)} scenarios")
                print(f"  Intention completion: {metrics.intention_completion_rate:.1%}")
                print(f"  Executable actions: {metrics.executable_action_rate:.1%}")
                print(f"  Avg dialogue lines: {metrics.mean_dialogue_lines:.1f}")

    # Compare to baseline
    if "C00" in condition_metrics:
        baseline = condition_metrics["C00"]
        print("\n" + "-" * 70)
        print("COMPARISON TO BASELINE (C00)")
        print("-" * 70)

        comparisons = []
        for cond_id, metrics in sorted(condition_metrics.items()):
            if cond_id == "C00":
                continue
            comparison = compare_conditions(baseline, metrics)
            comparisons.append(comparison)

            print(f"\n{cond_id} vs C00:")
            print(f"  Intention completion: {comparison.intention_completion_improvement:+.1f}%")
            print(f"  Executable actions: {comparison.executable_action_improvement:+.1f}%")
            print(f"  Dialogue lines: {comparison.dialogue_lines_improvement:+.1f}%")

        # Generate markdown table
        if comparisons:
            table = generate_comparison_table(comparisons)
            table_path = output_dir / "comparison_table.md"
            with open(table_path, "w") as f:
                f.write(table)
            print(f"\n\nComparison table saved to: {table_path}")

    # Save full metrics
    metrics_path = output_dir / "all_metrics.json"
    all_metrics = {
        cond_id: metrics.to_dict()
        for cond_id, metrics in condition_metrics.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Full metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SIA-LLM ablation study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 4 conditions × 5 scenarios (verify setup)"
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=None,
        help="Override number of scenarios (default: 40 for full, 5 for quick)"
    )
    parser.add_argument(
        "--condition", type=str,
        help="Run single condition only (e.g., C08)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Dry run without API calls"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help="Only analyze existing results"
    )
    parser.add_argument(
        "--output-dir", type=str,
        help="Output directory (default: auto-generated)"
    )
    parser.add_argument(
        "--dataset", type=str, choices=sorted(DATASET_PATHS.keys()),
        help="Dataset preset to use (default: curated)"
    )
    parser.add_argument(
        "--scenarios-path", type=str,
        help="Path to curated scenarios JSON (overrides --dataset/default)"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Determine if this is a full study or quick test
    is_full_study = not args.quick and not args.condition

    # Determine conditions to run
    if args.condition:
        conditions = [args.condition]
    elif args.quick:
        conditions = STUDY_CONFIG["quick_conditions"]
    else:
        conditions = STUDY_CONFIG["condition_order"]

    # Determine number of scenarios
    if args.n_scenarios:
        n_scenarios = args.n_scenarios
    elif args.quick:
        n_scenarios = STUDY_CONFIG["quick_n_scenarios"]
    else:
        n_scenarios = STUDY_CONFIG["total_curated_scenarios"]  # All 40

    # Determine scenarios path
    if args.scenarios_path:
        scenarios_path = args.scenarios_path
        dataset_name = "custom"
    else:
        dataset_name = args.dataset or "curated"
        scenarios_path = DATASET_PATHS.get(dataset_name, STUDY_CONFIG["curated_scenarios_path"])

    # Load and validate scenarios
    try:
        scenarios = load_curated_scenarios(n_scenarios, scenarios_path=scenarios_path)
        n_scenarios = len(scenarios)  # Actual count
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output directory
    if args.resume and args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Output directory does not exist: {output_dir}")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
    elif args.resume:
        output_dir = find_latest_study_dir()
        if not output_dir:
            print("No previous study found to resume.")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = setup_output_directory()

    # Analyze only mode
    if args.analyze_only:
        if args.resume or args.output_dir:
            analyze_results(output_dir)
        else:
            latest = find_latest_study_dir()
            if latest:
                analyze_results(latest)
            else:
                print("No study results found.")
        return

    # Print plan
    print_study_plan(conditions, n_scenarios, args.dry_run, is_full_study)

    # Confirmation
    if not args.yes and not args.dry_run:
        response = input("Proceed? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Save study config
    config_path = output_dir / "study_config.json"
    with open(config_path, "w") as f:
        json.dump({
            **STUDY_CONFIG,
            "conditions": conditions,
            "n_scenarios": n_scenarios,
            "is_full_study": is_full_study,
            "dry_run": args.dry_run,
            "dataset": dataset_name,
            "curated_scenarios_path": scenarios_path,
            "started_at": datetime.now().isoformat(),
            "scenarios_used": [s["title"] for s in scenarios],
        }, f, indent=2)

    # Run the study
    run_study(
        conditions=conditions,
        n_scenarios=n_scenarios,
        output_dir=output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        rerun_conditions=conditions if args.resume and args.condition else None,
        scenarios=scenarios,
    )

    # Analyze results
    analyze_results(output_dir)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
