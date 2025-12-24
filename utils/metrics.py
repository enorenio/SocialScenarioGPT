"""
Metrics calculation utilities for SIA-LLM experiments.
Refactored from AutomaticEvaluation.py with additional metrics.
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ArtifactCounts:
    """Counts of generated artifacts."""
    agents: int = 0
    beliefs: int = 0
    desires: int = 0
    intentions: int = 0
    actions: int = 0
    conditions: int = 0
    effects: int = 0
    emotion_before: int = 0
    emotion_after: int = 0
    dialogue_lines: int = 0
    speak_actions: int = 0
    speak_conditions: int = 0
    speak_effects: int = 0

    @property
    def total(self) -> int:
        return (
            self.agents + self.beliefs + self.desires + self.intentions +
            self.actions + self.conditions + self.effects +
            self.emotion_before + self.emotion_after +
            self.dialogue_lines + self.speak_actions +
            self.speak_conditions + self.speak_effects
        )


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario."""
    scenario_name: str
    completed: bool
    artifact_counts: ArtifactCounts
    total_intentions: int
    intentions_completed: int
    total_actions: int
    immediately_executable: int
    intention_completion_rate: float
    immediately_executable_rate: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["artifact_counts"] = asdict(self.artifact_counts)
        return d


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple scenarios."""
    num_scenarios: int
    num_completed: int
    total_intentions: int
    intentions_completed: int
    total_actions: int
    immediately_executable: int
    intention_completion_rate: float
    immediately_executable_rate: float
    avg_dialogue_lines: float
    artifact_counts_total: ArtifactCounts
    artifact_counts_mean: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["artifact_counts_total"] = asdict(self.artifact_counts_total)
        return d


def count_artifacts(scenario: Dict[str, Any]) -> ArtifactCounts:
    """Count all artifacts in a scenario."""
    counts = ArtifactCounts()

    agents = scenario.get("agents", {})
    counts.agents = len(agents)

    for agent_name, agent_data in agents.items():
        knowledge = agent_data.get("knowledge_base", [])
        for item in knowledge:
            if "DES(" in item:
                counts.desires += 1
            elif "BEL(" in item:
                counts.beliefs += 1

        counts.intentions += len(agent_data.get("intentions", {}))

        actions = agent_data.get("actions", {})
        counts.actions += len(actions)

        for action_name, action_data in actions.items():
            if isinstance(action_data, dict):
                counts.conditions += len(action_data.get("conditions", []))
                counts.effects += len(action_data.get("effects", []))
                counts.emotion_before += len(action_data.get("emotion_condition", []))
                counts.emotion_after += len(action_data.get("occ_emotion", []))

        speak_actions = agent_data.get("speak_actions", {})
        counts.speak_actions += len(speak_actions)

        for sp_name, sp_data in speak_actions.items():
            if isinstance(sp_data, dict):
                counts.speak_conditions += len(sp_data.get("conditions", []))
                counts.speak_effects += len(sp_data.get("effects", []))

    counts.dialogue_lines = len(scenario.get("dialogue_tree", []))

    return counts


def check_action_executable(
    conditions: List[str],
    knowledge_base: List[str],
) -> bool:
    """Check if an action's conditions are satisfied by the knowledge base."""
    return all(cond in knowledge_base for cond in conditions)


def apply_effects(
    knowledge_base: List[str],
    effects: List[str],
) -> List[str]:
    """Apply effects to update knowledge base."""
    # Parse into key-value pairs
    kb_dict = {}
    for item in knowledge_base:
        if "=" in item:
            key, val = item.split("=", 1)
            kb_dict[key.strip()] = val.strip()

    effect_dict = {}
    for item in effects:
        if "=" in item:
            key, val = item.split("=", 1)
            effect_dict[key.strip()] = val.strip()

    # Apply effects
    for key, val in effect_dict.items():
        kb_dict[key] = val

    return [f"{k}={v}" for k, v in kb_dict.items()]


def count_immediately_executable(scenario: Dict[str, Any]) -> Tuple[int, int]:
    """
    Count actions that are immediately executable from initial state.

    Returns:
        Tuple of (executable_count, total_actions)
    """
    executable = 0
    total = 0

    agents = scenario.get("agents", {})
    for agent_name, agent_data in agents.items():
        knowledge = agent_data.get("knowledge_base", [])
        actions = agent_data.get("actions", {})

        for action_name, action_data in actions.items():
            if not isinstance(action_data, dict):
                continue

            total += 1
            conditions = action_data.get("conditions", [])

            if check_action_executable(conditions, knowledge):
                executable += 1

    return executable, total


def count_reachable_intentions(scenario: Dict[str, Any]) -> Tuple[int, int]:
    """
    Count intentions that can be fully completed.

    Returns:
        Tuple of (completed_intentions, total_intentions)
    """
    completed = 0
    total = 0

    agents = scenario.get("agents", {})
    for agent_name, agent_data in agents.items():
        knowledge = list(agent_data.get("knowledge_base", []))
        intentions = agent_data.get("intentions", {})
        actions = agent_data.get("actions", {})

        for intention_name, intention_data in intentions.items():
            total += 1

            if not isinstance(intention_data, dict):
                continue

            action_plan = intention_data.get("action_plan", [])
            plan_completed = True
            current_kb = list(knowledge)

            for action_name in action_plan:
                action_data = actions.get(action_name, {})
                if not isinstance(action_data, dict):
                    continue

                conditions = action_data.get("conditions", [])
                effects = action_data.get("effects", [])

                if check_action_executable(conditions, current_kb):
                    current_kb = apply_effects(current_kb, effects)
                else:
                    plan_completed = False
                    break

            if plan_completed and len(action_plan) > 0:
                completed += 1

    return completed, total


def calculate_scenario_metrics(scenario: Dict[str, Any]) -> ScenarioMetrics:
    """Calculate all metrics for a single scenario."""
    artifact_counts = count_artifacts(scenario)
    immediately_exec, total_actions = count_immediately_executable(scenario)
    intentions_done, total_intentions = count_reachable_intentions(scenario)

    return ScenarioMetrics(
        scenario_name=scenario.get("scenario_name", "unknown"),
        completed=scenario.get("last_ended") == "end",
        artifact_counts=artifact_counts,
        total_intentions=total_intentions,
        intentions_completed=intentions_done,
        total_actions=total_actions,
        immediately_executable=immediately_exec,
        intention_completion_rate=intentions_done / total_intentions if total_intentions > 0 else 0,
        immediately_executable_rate=immediately_exec / total_actions if total_actions > 0 else 0,
    )


def calculate_aggregate_metrics(scenarios: List[Dict[str, Any]]) -> AggregateMetrics:
    """Calculate aggregate metrics across all scenarios."""
    total_counts = ArtifactCounts()
    total_intentions = 0
    intentions_completed = 0
    total_actions = 0
    immediately_executable = 0
    num_completed = 0

    for scenario in scenarios:
        metrics = calculate_scenario_metrics(scenario)

        if metrics.completed:
            num_completed += 1

        # Sum artifact counts
        total_counts.agents += metrics.artifact_counts.agents
        total_counts.beliefs += metrics.artifact_counts.beliefs
        total_counts.desires += metrics.artifact_counts.desires
        total_counts.intentions += metrics.artifact_counts.intentions
        total_counts.actions += metrics.artifact_counts.actions
        total_counts.conditions += metrics.artifact_counts.conditions
        total_counts.effects += metrics.artifact_counts.effects
        total_counts.emotion_before += metrics.artifact_counts.emotion_before
        total_counts.emotion_after += metrics.artifact_counts.emotion_after
        total_counts.dialogue_lines += metrics.artifact_counts.dialogue_lines
        total_counts.speak_actions += metrics.artifact_counts.speak_actions
        total_counts.speak_conditions += metrics.artifact_counts.speak_conditions
        total_counts.speak_effects += metrics.artifact_counts.speak_effects

        # Sum key metrics
        total_intentions += metrics.total_intentions
        intentions_completed += metrics.intentions_completed
        total_actions += metrics.total_actions
        immediately_executable += metrics.immediately_executable

    n = len(scenarios) if scenarios else 1

    # Calculate means
    mean_counts = {
        "agents": total_counts.agents / n,
        "beliefs": total_counts.beliefs / n,
        "desires": total_counts.desires / n,
        "intentions": total_counts.intentions / n,
        "actions": total_counts.actions / n,
        "conditions": total_counts.conditions / n,
        "effects": total_counts.effects / n,
        "emotion_before": total_counts.emotion_before / n,
        "emotion_after": total_counts.emotion_after / n,
        "dialogue_lines": total_counts.dialogue_lines / n,
        "speak_actions": total_counts.speak_actions / n,
        "speak_conditions": total_counts.speak_conditions / n,
        "speak_effects": total_counts.speak_effects / n,
    }

    return AggregateMetrics(
        num_scenarios=len(scenarios),
        num_completed=num_completed,
        total_intentions=total_intentions,
        intentions_completed=intentions_completed,
        total_actions=total_actions,
        immediately_executable=immediately_executable,
        intention_completion_rate=intentions_completed / total_intentions if total_intentions > 0 else 0,
        immediately_executable_rate=immediately_executable / total_actions if total_actions > 0 else 0,
        avg_dialogue_lines=mean_counts["dialogue_lines"],
        artifact_counts_total=total_counts,
        artifact_counts_mean=mean_counts,
    )


def load_scenarios(directory: str) -> List[Dict[str, Any]]:
    """Load all scenario JSON files from a directory."""
    scenarios = []
    dir_path = Path(directory)

    for filepath in dir_path.glob("*.json"):
        if filepath.name == "time_spent.json":
            continue

        try:
            with open(filepath) as f:
                scenarios.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")

    return scenarios


def print_metrics_report(metrics: AggregateMetrics):
    """Print a formatted metrics report."""
    print("=" * 60)
    print("METRICS REPORT")
    print("=" * 60)
    print(f"Scenarios: {metrics.num_scenarios} ({metrics.num_completed} completed)")
    print()
    print("KEY METRICS:")
    print(f"  Intention completion: {metrics.intention_completion_rate*100:.1f}% "
          f"({metrics.intentions_completed}/{metrics.total_intentions})")
    print(f"  Immediately executable: {metrics.immediately_executable_rate*100:.1f}% "
          f"({metrics.immediately_executable}/{metrics.total_actions})")
    print(f"  Avg dialogue lines: {metrics.avg_dialogue_lines:.2f}")
    print()
    print("ARTIFACT COUNTS (Mean per scenario):")
    for name, value in metrics.artifact_counts_mean.items():
        print(f"  {name}: {value:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    # Test with existing data
    scenarios = load_scenarios("Data")
    metrics = calculate_aggregate_metrics(scenarios)
    print_metrics_report(metrics)
