"""
Baseline experiment runner for TASK-002.
Runs scenario generation and collects metrics for baseline comparison.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.feature_flags import FeatureFlags, get_profile
from SocialScenarioGPT import generate_scenario
from AutomaticEvaluation import (
    load_scenarios,
    count_artifacts_scenarios,
    count_initial_actions_available,
    count_reachable_intentions,
)


def load_rocstories_scenarios(n_scenarios: int = 20, random_state: int = 42) -> pd.DataFrame:
    """Load RocStories scenarios from the Dataset folder."""
    dataset_path = Path(__file__).parent.parent / "Dataset"

    df_2016 = pd.read_csv(dataset_path / "ROCStories__spring2016.csv")
    df_2017 = pd.read_csv(dataset_path / "ROCStories_winter2017.csv")

    df_rocstories = pd.concat([df_2016, df_2017], ignore_index=True)
    df_sample = df_rocstories.sample(n=n_scenarios, replace=False, random_state=random_state)

    return df_sample


def run_baseline_experiment(
    n_scenarios: int = 20,
    output_dir: str = "experiments/results",
    feature_profile: str = "baseline",
    random_state: int = 42,
    max_retries: int = 3,
) -> dict:
    """
    Run baseline experiment with the original system.

    Args:
        n_scenarios: Number of scenarios to generate
        output_dir: Directory to save results
        feature_profile: Feature flag profile to use
        random_state: Random seed for reproducibility
        max_retries: Max retries per scenario on failure

    Returns:
        Dictionary with experiment results and metrics
    """
    # Get feature flags
    flags = get_profile(feature_profile)

    # Create output directory
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_path / f"{feature_profile}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config
    config = {
        "feature_profile": feature_profile,
        "feature_flags": flags.to_dict(),
        "n_scenarios": n_scenarios,
        "random_state": random_state,
        "max_retries": max_retries,
        "timestamp": timestamp,
    }
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Running experiment: {feature_profile}")
    print(f"Feature flags: {flags.to_dict()}")
    print(f"Output directory: {experiment_dir}")
    print(f"Generating {n_scenarios} scenarios...")
    print("-" * 50)

    # Load scenarios
    df = load_rocstories_scenarios(n_scenarios, random_state)

    # Track timing and results
    scenario_times = []
    scenario_results = []
    failed_scenarios = []

    for idx, (_, row) in enumerate(df.iterrows()):
        scenario_description = " ".join([
            row['sentence1'], row['sentence2'], row['sentence3'],
            row['sentence4'], row['sentence5']
        ])
        scenario_name = f"exp_{feature_profile}_{row['storytitle']}"

        print(f"\n[{idx+1}/{n_scenarios}] Generating: {scenario_name}")

        start_time = time.time()
        success = False

        for retry in range(max_retries):
            try:
                generate_scenario(scenario_name, scenario_description)
                success = True
                break
            except Exception as e:
                print(f"  Retry {retry+1}/{max_retries} failed: {e}")
                continue

        elapsed_time = time.time() - start_time
        scenario_times.append(elapsed_time)

        if success:
            print(f"  Completed in {elapsed_time/60:.2f} minutes")
            scenario_results.append({
                "name": scenario_name,
                "description": scenario_description,
                "time_seconds": elapsed_time,
                "success": True,
            })
        else:
            print(f"  FAILED after {max_retries} retries")
            failed_scenarios.append(scenario_name)
            scenario_results.append({
                "name": scenario_name,
                "description": scenario_description,
                "time_seconds": elapsed_time,
                "success": False,
            })

    # Save individual scenario results
    with open(experiment_dir / "scenario_results.json", "w") as f:
        json.dump(scenario_results, f, indent=2)

    # Load generated scenarios and compute metrics
    print("\n" + "-" * 50)
    print("Computing metrics...")

    data_dir = Path(__file__).parent.parent / "Data"
    scenarios = load_scenarios(str(data_dir))

    # Filter to only experiment scenarios
    exp_scenarios = [s for s in scenarios if s.get("scenario_name", "").startswith(f"exp_{feature_profile}_")]

    # Compute metrics
    artifact_counts = count_artifacts_scenarios(exp_scenarios)
    initial_actions, conditions_in_kb = count_initial_actions_available(exp_scenarios)
    intentions_completed = count_reachable_intentions(exp_scenarios)

    # Count total intentions
    total_intentions = 0
    for scenario in exp_scenarios:
        try:
            for agent in scenario["agents"].keys():
                total_intentions += len(scenario["agents"][agent].get("intentions", {}))
        except:
            continue

    # Count total actions
    total_actions = artifact_counts["absolute"]["actions"]

    # Calculate key metrics
    metrics = {
        "scenarios_generated": len(exp_scenarios),
        "scenarios_failed": len(failed_scenarios),
        "total_generation_time_minutes": sum(scenario_times) / 60,
        "avg_generation_time_minutes": (sum(scenario_times) / len(scenario_times)) / 60 if scenario_times else 0,

        # Artifact counts
        "artifact_counts": artifact_counts,

        # Key paper metrics
        "total_intentions": total_intentions,
        "intentions_completed": intentions_completed,
        "intention_completion_rate": intentions_completed / total_intentions if total_intentions > 0 else 0,

        "total_actions": total_actions,
        "immediately_executable_actions": initial_actions,
        "immediately_executable_rate": initial_actions / total_actions if total_actions > 0 else 0,

        "avg_dialogue_lines": artifact_counts["mean"]["dialogue_lines"],

        # Additional metrics
        "conditions_in_knowledge_base": conditions_in_kb,
        "failed_scenarios": failed_scenarios,
    }

    # Save metrics
    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Profile: {feature_profile}")
    print(f"Scenarios generated: {metrics['scenarios_generated']}")
    print(f"Scenarios failed: {metrics['scenarios_failed']}")
    print(f"Total time: {metrics['total_generation_time_minutes']:.2f} minutes")
    print(f"Avg time per scenario: {metrics['avg_generation_time_minutes']:.2f} minutes")
    print()
    print("KEY METRICS (Paper Comparison):")
    print(f"  Intention completion rate: {metrics['intention_completion_rate']*100:.1f}% ({intentions_completed}/{total_intentions})")
    print(f"  Immediately executable: {metrics['immediately_executable_rate']*100:.1f}% ({initial_actions}/{total_actions})")
    print(f"  Avg dialogue lines: {metrics['avg_dialogue_lines']:.2f}")
    print()
    print("ARTIFACT COUNTS (Mean per scenario):")
    for key, val in artifact_counts["mean"].items():
        print(f"  {key}: {val:.2f}")
    print()
    print(f"Results saved to: {experiment_dir}")

    return metrics


def analyze_existing_data(data_dir: str = "Data") -> dict:
    """
    Analyze existing generated scenarios in the Data directory.
    Useful for evaluating pre-existing baseline data.
    """
    data_path = Path(__file__).parent.parent / data_dir
    scenarios = load_scenarios(str(data_path))

    # Filter to completed scenarios
    completed = [s for s in scenarios if s.get("last_ended") == "end"]

    print(f"Found {len(scenarios)} total scenarios, {len(completed)} completed")

    artifact_counts = count_artifacts_scenarios(completed)
    initial_actions, conditions_in_kb = count_initial_actions_available(completed)
    intentions_completed = count_reachable_intentions(completed)

    # Count total intentions
    total_intentions = 0
    for scenario in completed:
        try:
            for agent in scenario["agents"].keys():
                total_intentions += len(scenario["agents"][agent].get("intentions", {}))
        except:
            continue

    total_actions = artifact_counts["absolute"]["actions"]

    metrics = {
        "scenarios_analyzed": len(completed),
        "artifact_counts": artifact_counts,
        "total_intentions": total_intentions,
        "intentions_completed": intentions_completed,
        "intention_completion_rate": intentions_completed / total_intentions if total_intentions > 0 else 0,
        "total_actions": total_actions,
        "immediately_executable_actions": initial_actions,
        "immediately_executable_rate": initial_actions / total_actions if total_actions > 0 else 0,
        "avg_dialogue_lines": artifact_counts["mean"]["dialogue_lines"],
    }

    print("\n" + "=" * 50)
    print("EXISTING DATA ANALYSIS")
    print("=" * 50)
    print(f"Completed scenarios: {len(completed)}")
    print()
    print("KEY METRICS:")
    print(f"  Intention completion rate: {metrics['intention_completion_rate']*100:.1f}% ({intentions_completed}/{total_intentions})")
    print(f"  Immediately executable: {metrics['immediately_executable_rate']*100:.1f}% ({initial_actions}/{total_actions})")
    print(f"  Avg dialogue lines: {metrics['avg_dialogue_lines']:.2f}")
    print()
    print("ARTIFACT COUNTS (Mean per scenario):")
    for key, val in artifact_counts["mean"].items():
        print(f"  {key}: {val:.2f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--analyze-existing", action="store_true",
                        help="Analyze existing data instead of generating new scenarios")
    parser.add_argument("--n-scenarios", type=int, default=20,
                        help="Number of scenarios to generate (default: 20)")
    parser.add_argument("--profile", type=str, default="baseline",
                        help="Feature profile to use (default: baseline)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.analyze_existing:
        analyze_existing_data()
    else:
        run_baseline_experiment(
            n_scenarios=args.n_scenarios,
            feature_profile=args.profile,
            random_state=args.random_state,
        )
