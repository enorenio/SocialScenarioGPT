"""
Metrics calculation for SIA-LLM scenarios.

Provides comprehensive metrics for individual scenarios and experiment runs,
including artifact counts, completion rates, and dialogue analysis.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario."""
    scenario_name: str

    # Artifact counts
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

    # Computed metrics
    total_artifacts: int = 0
    intention_completion_rate: float = 0.0
    executable_actions_rate: float = 0.0
    executable_actions_count: int = 0
    total_action_count: int = 0

    # Dialogue metrics
    dialogue_branch_points: int = 0
    dialogue_unique_paths: int = 0
    dialogue_styles_used: int = 0

    # Quality flags
    is_complete: bool = True
    has_dialogue: bool = True
    has_emotions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "artifact_counts": {
                "agents": self.agents,
                "beliefs": self.beliefs,
                "desires": self.desires,
                "intentions": self.intentions,
                "actions": self.actions,
                "conditions": self.conditions,
                "effects": self.effects,
                "emotion_before": self.emotion_before,
                "emotion_after": self.emotion_after,
                "dialogue_lines": self.dialogue_lines,
                "speak_actions": self.speak_actions,
                "total": self.total_artifacts,
            },
            "completion_metrics": {
                "intention_completion_rate": self.intention_completion_rate,
                "executable_actions_rate": self.executable_actions_rate,
                "executable_actions_count": self.executable_actions_count,
                "total_action_count": self.total_action_count,
            },
            "dialogue_metrics": {
                "lines": self.dialogue_lines,
                "branch_points": self.dialogue_branch_points,
                "unique_paths": self.dialogue_unique_paths,
                "styles_used": self.dialogue_styles_used,
            },
            "quality_flags": {
                "is_complete": self.is_complete,
                "has_dialogue": self.has_dialogue,
                "has_emotions": self.has_emotions,
            },
        }


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment run."""
    experiment_name: str
    timestamp: str = ""
    condition_id: str = ""

    # Counts
    total_scenarios: int = 0
    complete_scenarios: int = 0
    scenarios_with_dialogue: int = 0

    # Aggregated artifact counts (means)
    mean_agents: float = 0.0
    mean_beliefs: float = 0.0
    mean_desires: float = 0.0
    mean_intentions: float = 0.0
    mean_actions: float = 0.0
    mean_conditions: float = 0.0
    mean_effects: float = 0.0
    mean_dialogue_lines: float = 0.0
    mean_speak_actions: float = 0.0
    mean_total_artifacts: float = 0.0

    # Key metrics
    intention_completion_rate: float = 0.0
    executable_actions_rate: float = 0.0
    mean_dialogue_branch_points: float = 0.0
    mean_dialogue_paths: float = 0.0

    # Totals
    total_intentions: int = 0
    completed_intentions: int = 0
    total_actions: int = 0
    executable_actions: int = 0

    # Individual scenario metrics
    scenario_metrics: List[ScenarioMetrics] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "condition_id": self.condition_id,
            "summary": {
                "total_scenarios": self.total_scenarios,
                "complete_scenarios": self.complete_scenarios,
                "scenarios_with_dialogue": self.scenarios_with_dialogue,
            },
            "key_metrics": {
                "intention_completion_rate": round(self.intention_completion_rate, 4),
                "executable_actions_rate": round(self.executable_actions_rate, 4),
                "mean_dialogue_lines": round(self.mean_dialogue_lines, 2),
                "mean_dialogue_branch_points": round(self.mean_dialogue_branch_points, 2),
            },
            "artifact_counts_mean": {
                "agents": round(self.mean_agents, 2),
                "beliefs": round(self.mean_beliefs, 2),
                "desires": round(self.mean_desires, 2),
                "intentions": round(self.mean_intentions, 2),
                "actions": round(self.mean_actions, 2),
                "conditions": round(self.mean_conditions, 2),
                "effects": round(self.mean_effects, 2),
                "dialogue_lines": round(self.mean_dialogue_lines, 2),
                "speak_actions": round(self.mean_speak_actions, 2),
                "total": round(self.mean_total_artifacts, 2),
            },
            "totals": {
                "intentions": self.total_intentions,
                "completed_intentions": self.completed_intentions,
                "actions": self.total_actions,
                "executable_actions": self.executable_actions,
            },
        }

    def to_json(self, include_scenarios: bool = False) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        if include_scenarios:
            data["scenarios"] = [s.to_dict() for s in self.scenario_metrics]
        return json.dumps(data, indent=2)


def calculate_scenario_metrics(scenario: Dict[str, Any]) -> ScenarioMetrics:
    """
    Calculate metrics for a single scenario.

    Args:
        scenario: The scenario dictionary

    Returns:
        ScenarioMetrics with all calculated values
    """
    metrics = ScenarioMetrics(
        scenario_name=scenario.get("scenario_name", "unknown")
    )

    agents = scenario.get("agents", {})
    metrics.agents = len(agents)

    # Count beliefs, desires, intentions, actions
    for agent_name, agent_data in agents.items():
        if not isinstance(agent_data, dict):
            continue

        # Knowledge base (beliefs and desires)
        kb = agent_data.get("knowledge_base", [])
        for item in kb:
            if isinstance(item, str):
                if "DES(" in item:
                    metrics.desires += 1
                elif "BEL(" in item:
                    metrics.beliefs += 1

        # Intentions
        intentions = agent_data.get("intentions", {})
        metrics.intentions += len(intentions)

        # Actions
        actions = agent_data.get("actions", {})
        metrics.actions += len(actions)

        # Conditions and effects
        for action_name, action_data in actions.items():
            if isinstance(action_data, dict):
                metrics.conditions += len(action_data.get("conditions", []))
                metrics.effects += len(action_data.get("effects", []))
                metrics.emotion_before += len(action_data.get("emotion_condition", []))
                metrics.emotion_after += len(action_data.get("occ_emotion", []))

        # Speak actions
        speak_actions = agent_data.get("speak_actions", {})
        metrics.speak_actions += len(speak_actions)

    # Dialogue
    dialogue_tree = scenario.get("dialogue_tree", [])
    metrics.dialogue_lines = len(dialogue_tree)
    metrics.has_dialogue = len(dialogue_tree) > 0

    # Dialogue analysis (if available)
    try:
        from prompts.dialogue import analyze_dialogue
        if dialogue_tree:
            dialogue_metrics = analyze_dialogue(dialogue_tree)
            metrics.dialogue_branch_points = dialogue_metrics.branch_points
            metrics.dialogue_unique_paths = dialogue_metrics.approximate_paths
            metrics.dialogue_styles_used = len(dialogue_metrics.styles_used)
    except ImportError:
        pass

    # Total artifacts
    metrics.total_artifacts = (
        metrics.agents + metrics.beliefs + metrics.desires +
        metrics.intentions + metrics.actions + metrics.conditions +
        metrics.effects + metrics.emotion_before + metrics.emotion_after +
        metrics.dialogue_lines + metrics.speak_actions
    )

    # Quality flags
    metrics.has_emotions = metrics.emotion_before > 0 or metrics.emotion_after > 0
    metrics.is_complete = (
        metrics.agents > 0 and
        metrics.intentions > 0 and
        metrics.actions > 0
    )

    # Calculate executable actions rate
    metrics.total_action_count, metrics.executable_actions_count = \
        _count_executable_actions(scenario)
    if metrics.total_action_count > 0:
        metrics.executable_actions_rate = (
            metrics.executable_actions_count / metrics.total_action_count
        )

    # Calculate intention completion rate
    completed, total = _count_completable_intentions(scenario)
    metrics.intention_completion_rate = completed / total if total > 0 else 0.0

    return metrics


def _count_executable_actions(scenario: Dict[str, Any]) -> Tuple[int, int]:
    """Count total actions and immediately executable ones."""
    total = 0
    executable = 0

    agents = scenario.get("agents", {})
    for agent_name, agent_data in agents.items():
        if not isinstance(agent_data, dict):
            continue

        knowledge = set(agent_data.get("knowledge_base", []))
        actions = agent_data.get("actions", {})

        for action_name, action_data in actions.items():
            if not isinstance(action_data, dict):
                continue

            total += 1
            conditions = action_data.get("conditions", [])

            # Check if all conditions are in knowledge base
            if all(cond in knowledge for cond in conditions):
                executable += 1

    return total, executable


def _count_completable_intentions(scenario: Dict[str, Any]) -> Tuple[int, int]:
    """Count completable intentions using simulation."""
    completed = 0
    total = 0

    agents = scenario.get("agents", {})
    for agent_name, agent_data in agents.items():
        if not isinstance(agent_data, dict):
            continue

        knowledge = set(agent_data.get("knowledge_base", []))
        actions = agent_data.get("actions", {})
        intentions = agent_data.get("intentions", {})

        for intent_name, intent_data in intentions.items():
            if not isinstance(intent_data, dict):
                continue

            total += 1
            action_plan = intent_data.get("action_plan", [])

            # Simulate action execution
            current_knowledge = knowledge.copy()
            actions_completed = 0

            for action_name in action_plan:
                action = actions.get(action_name, {})
                if not isinstance(action, dict):
                    break

                conditions = action.get("conditions", [])
                effects = action.get("effects", [])

                # Check if action can execute
                if all(cond in current_knowledge for cond in conditions):
                    actions_completed += 1
                    # Apply effects (simplified)
                    for effect in effects:
                        current_knowledge.add(effect)
                else:
                    break

            # Intention complete if all actions executed
            if action_plan and actions_completed == len(action_plan):
                completed += 1

    return completed, total


def calculate_experiment_metrics(
    scenarios: List[Dict[str, Any]],
    experiment_name: str = "experiment",
    condition_id: str = "",
) -> ExperimentMetrics:
    """
    Calculate aggregated metrics for an experiment.

    Args:
        scenarios: List of scenario dictionaries
        experiment_name: Name for this experiment
        condition_id: Ablation condition identifier

    Returns:
        ExperimentMetrics with aggregated values
    """
    metrics = ExperimentMetrics(
        experiment_name=experiment_name,
        condition_id=condition_id,
    )

    if not scenarios:
        return metrics

    # Calculate per-scenario metrics
    scenario_metrics_list = []
    for scenario in scenarios:
        sm = calculate_scenario_metrics(scenario)
        scenario_metrics_list.append(sm)

    metrics.scenario_metrics = scenario_metrics_list
    metrics.total_scenarios = len(scenarios)

    # Aggregate counts
    n = len(scenarios)
    metrics.mean_agents = sum(s.agents for s in scenario_metrics_list) / n
    metrics.mean_beliefs = sum(s.beliefs for s in scenario_metrics_list) / n
    metrics.mean_desires = sum(s.desires for s in scenario_metrics_list) / n
    metrics.mean_intentions = sum(s.intentions for s in scenario_metrics_list) / n
    metrics.mean_actions = sum(s.actions for s in scenario_metrics_list) / n
    metrics.mean_conditions = sum(s.conditions for s in scenario_metrics_list) / n
    metrics.mean_effects = sum(s.effects for s in scenario_metrics_list) / n
    metrics.mean_dialogue_lines = sum(s.dialogue_lines for s in scenario_metrics_list) / n
    metrics.mean_speak_actions = sum(s.speak_actions for s in scenario_metrics_list) / n
    metrics.mean_total_artifacts = sum(s.total_artifacts for s in scenario_metrics_list) / n

    # Dialogue metrics
    metrics.mean_dialogue_branch_points = sum(
        s.dialogue_branch_points for s in scenario_metrics_list
    ) / n
    metrics.mean_dialogue_paths = sum(
        s.dialogue_unique_paths for s in scenario_metrics_list
    ) / n

    # Quality counts
    metrics.complete_scenarios = sum(1 for s in scenario_metrics_list if s.is_complete)
    metrics.scenarios_with_dialogue = sum(1 for s in scenario_metrics_list if s.has_dialogue)

    # Totals for rate calculations
    metrics.total_intentions = sum(s.intentions for s in scenario_metrics_list)
    metrics.total_actions = sum(s.total_action_count for s in scenario_metrics_list)
    metrics.executable_actions = sum(s.executable_actions_count for s in scenario_metrics_list)

    # Calculate overall rates from totals
    if metrics.total_actions > 0:
        metrics.executable_actions_rate = metrics.executable_actions / metrics.total_actions

    # Intention completion (need to recalculate from scenarios)
    total_completed = 0
    total_intentions = 0
    for scenario in scenarios:
        completed, total = _count_completable_intentions(scenario)
        total_completed += completed
        total_intentions += total

    metrics.completed_intentions = total_completed
    if total_intentions > 0:
        metrics.intention_completion_rate = total_completed / total_intentions

    return metrics


def load_scenarios_from_directory(
    directory: str,
    pattern: str = "test_*.json",
) -> List[Dict[str, Any]]:
    """
    Load all scenarios from a directory.

    Args:
        directory: Path to directory containing scenario JSON files
        pattern: Glob pattern for scenario files

    Returns:
        List of scenario dictionaries
    """
    scenarios = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return scenarios

    for file_path in sorted(dir_path.glob(pattern)):
        try:
            with open(file_path) as f:
                scenario = json.load(f)
                # Only include valid scenario objects (dicts with scenario_name or agents)
                if isinstance(scenario, dict) and ("scenario_name" in scenario or "agents" in scenario):
                    scenarios.append(scenario)
                # Skip arrays (FAtiMA export files) and other non-scenario files
        except (json.JSONDecodeError, IOError):
            continue

    return scenarios


def print_metrics_report(metrics: ExperimentMetrics) -> None:
    """Print a formatted metrics report."""
    print("=" * 70)
    print(f"METRICS REPORT: {metrics.experiment_name}")
    if metrics.condition_id:
        print(f"Condition: {metrics.condition_id}")
    print(f"Timestamp: {metrics.timestamp}")
    print("=" * 70)

    print(f"\nSCENARIOS:")
    print(f"  Total: {metrics.total_scenarios}")
    print(f"  Complete: {metrics.complete_scenarios}")
    print(f"  With dialogue: {metrics.scenarios_with_dialogue}")

    print(f"\nKEY METRICS:")
    print(f"  Intention completion rate: {metrics.intention_completion_rate:.1%}")
    print(f"  Executable actions rate: {metrics.executable_actions_rate:.1%}")
    print(f"  Mean dialogue lines: {metrics.mean_dialogue_lines:.1f}")
    print(f"  Mean dialogue branches: {metrics.mean_dialogue_branch_points:.2f}")

    print(f"\nARTIFACT COUNTS (mean per scenario):")
    print(f"  Agents: {metrics.mean_agents:.1f}")
    print(f"  Beliefs: {metrics.mean_beliefs:.1f}")
    print(f"  Desires: {metrics.mean_desires:.1f}")
    print(f"  Intentions: {metrics.mean_intentions:.1f}")
    print(f"  Actions: {metrics.mean_actions:.1f}")
    print(f"  Conditions: {metrics.mean_conditions:.1f}")
    print(f"  Effects: {metrics.mean_effects:.1f}")
    print(f"  Dialogue lines: {metrics.mean_dialogue_lines:.1f}")
    print(f"  Total artifacts: {metrics.mean_total_artifacts:.1f}")

    print(f"\nTOTALS:")
    print(f"  Intentions: {metrics.total_intentions} ({metrics.completed_intentions} completed)")
    print(f"  Actions: {metrics.total_actions} ({metrics.executable_actions} executable)")

    print("=" * 70)
