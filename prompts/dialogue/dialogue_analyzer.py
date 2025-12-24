"""
Dialogue analysis and metrics for SIA-LLM scenarios.

Provides utilities for:
1. Parsing dialogue trees
2. Analyzing state machine structure
3. Calculating dialogue metrics
4. Generating dialogue reports
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict


@dataclass
class DialogueLine:
    """Parsed dialogue line."""
    current_state: str
    next_state: str
    meaning: str
    style: str
    utterance: str
    speaker: Optional[str] = None

    @classmethod
    def parse(cls, line: str) -> Optional["DialogueLine"]:
        """Parse a dialogue line from string format."""
        # Format: <CurrentState, NextState, Meaning, Style, "UtteranceText">
        # or: [[CurrentState, NextState, Meaning, Style, "UtteranceText"]]
        line = line.strip()

        # Try both formats
        if line.startswith("<") and line.endswith(">"):
            line = line[1:-1]
        elif line.startswith("[[") and line.endswith("]]"):
            line = line[2:-2]
        else:
            return None

        # Split carefully to handle quoted utterance
        # Pattern: state, state, meaning, style, "utterance"
        match = re.match(
            r'([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*["\'](.+)["\']',
            line
        )

        if match:
            return cls(
                current_state=match.group(1).strip(),
                next_state=match.group(2).strip(),
                meaning=match.group(3).strip(),
                style=match.group(4).strip(),
                utterance=match.group(5).strip(),
            )

        # Try simpler split for edge cases
        parts = line.split(",")
        if len(parts) >= 5:
            # Last part is the utterance (may contain commas)
            utterance = ",".join(parts[4:]).strip().strip("\"'")
            return cls(
                current_state=parts[0].strip(),
                next_state=parts[1].strip(),
                meaning=parts[2].strip(),
                style=parts[3].strip(),
                utterance=utterance,
            )

        return None


@dataclass
class DialogueState:
    """A state in the dialogue state machine."""
    name: str
    outgoing: List[DialogueLine] = field(default_factory=list)
    incoming: List[DialogueLine] = field(default_factory=list)

    @property
    def is_start(self) -> bool:
        return self.name.lower() in ("start", "initial", "begin")

    @property
    def is_end(self) -> bool:
        return self.name.lower() in ("end", "final", "exit", "done")

    @property
    def branch_count(self) -> int:
        """Number of outgoing transitions (branches)."""
        return len(self.outgoing)


@dataclass
class DialogueGraph:
    """Dialogue state machine graph."""
    states: Dict[str, DialogueState] = field(default_factory=dict)
    lines: List[DialogueLine] = field(default_factory=list)

    @classmethod
    def from_dialogue_tree(cls, dialogue_tree: List[str]) -> "DialogueGraph":
        """Build graph from dialogue tree list."""
        graph = cls()

        for line_str in dialogue_tree:
            line = DialogueLine.parse(line_str)
            if line:
                graph.lines.append(line)

                # Ensure states exist
                if line.current_state not in graph.states:
                    graph.states[line.current_state] = DialogueState(
                        name=line.current_state
                    )
                if line.next_state not in graph.states:
                    graph.states[line.next_state] = DialogueState(
                        name=line.next_state
                    )

                # Add transitions
                graph.states[line.current_state].outgoing.append(line)
                graph.states[line.next_state].incoming.append(line)

        return graph

    def get_start_states(self) -> List[DialogueState]:
        """Get all start states."""
        return [s for s in self.states.values() if s.is_start or not s.incoming]

    def get_end_states(self) -> List[DialogueState]:
        """Get all end states."""
        return [s for s in self.states.values() if s.is_end or not s.outgoing]

    def get_branch_points(self) -> List[DialogueState]:
        """Get states with multiple outgoing transitions."""
        return [s for s in self.states.values() if s.branch_count > 1]

    def count_paths(self, max_depth: int = 20) -> int:
        """Count approximate number of unique paths through dialogue."""
        if not self.states:
            return 0

        starts = self.get_start_states()
        if not starts:
            starts = [list(self.states.values())[0]]

        total_paths = 0
        for start in starts:
            total_paths += self._count_paths_from(start.name, set(), max_depth)

        return total_paths

    def _count_paths_from(
        self, state_name: str, visited: Set[str], depth: int
    ) -> int:
        """Recursively count paths from a state."""
        if depth <= 0 or state_name in visited:
            return 0

        state = self.states.get(state_name)
        if not state:
            return 0

        if state.is_end or not state.outgoing:
            return 1

        visited = visited | {state_name}
        total = 0
        for line in state.outgoing:
            total += self._count_paths_from(line.next_state, visited, depth - 1)

        return max(total, 1)


@dataclass
class DialogueMetrics:
    """Metrics for dialogue analysis."""
    total_lines: int = 0
    unique_states: int = 0
    branch_points: int = 0
    max_branches: int = 0
    approximate_paths: int = 0
    styles_used: Set[str] = field(default_factory=set)
    meanings_used: Set[str] = field(default_factory=set)
    avg_utterance_length: float = 0.0
    has_start: bool = False
    has_end: bool = False
    speakers: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "unique_states": self.unique_states,
            "branch_points": self.branch_points,
            "max_branches": self.max_branches,
            "approximate_paths": self.approximate_paths,
            "styles_used": list(self.styles_used),
            "meanings_used": list(self.meanings_used),
            "avg_utterance_length": round(self.avg_utterance_length, 1),
            "has_start": self.has_start,
            "has_end": self.has_end,
            "speakers": self.speakers,
        }


def analyze_dialogue(
    dialogue_tree: List[str],
    speak_actions: Dict[str, Dict[str, Any]] = None,
) -> DialogueMetrics:
    """
    Analyze dialogue tree and return metrics.

    Args:
        dialogue_tree: List of dialogue line strings
        speak_actions: Optional dict mapping agent -> speak_actions

    Returns:
        DialogueMetrics with analysis results
    """
    graph = DialogueGraph.from_dialogue_tree(dialogue_tree)

    metrics = DialogueMetrics()
    metrics.total_lines = len(graph.lines)
    metrics.unique_states = len(graph.states)

    # Branch analysis
    branch_points = graph.get_branch_points()
    metrics.branch_points = len(branch_points)
    metrics.max_branches = max(
        (s.branch_count for s in graph.states.values()), default=0
    )
    metrics.approximate_paths = graph.count_paths()

    # Style and meaning analysis
    for line in graph.lines:
        metrics.styles_used.add(line.style)
        metrics.meanings_used.add(line.meaning)

    # Utterance analysis
    if graph.lines:
        total_length = sum(len(line.utterance) for line in graph.lines)
        metrics.avg_utterance_length = total_length / len(graph.lines)

    # Start/end detection
    metrics.has_start = len(graph.get_start_states()) > 0
    metrics.has_end = len(graph.get_end_states()) > 0

    # Speaker analysis from speak_actions
    if speak_actions:
        for agent_name, actions in speak_actions.items():
            metrics.speakers[agent_name] = len(actions)

    return metrics


def analyze_scenario_dialogue(scenario: Dict[str, Any]) -> DialogueMetrics:
    """Analyze dialogue from a full scenario dict."""
    dialogue_tree = scenario.get("dialogue_tree", [])

    # Collect speak_actions from all agents
    speak_actions = {}
    for agent_name, agent_data in scenario.get("agents", {}).items():
        if isinstance(agent_data, dict):
            sa = agent_data.get("speak_actions", {})
            if sa:
                speak_actions[agent_name] = sa

    return analyze_dialogue(dialogue_tree, speak_actions)


def print_dialogue_report(metrics: DialogueMetrics, scenario_name: str = ""):
    """Print a formatted dialogue analysis report."""
    print("=" * 50)
    if scenario_name:
        print(f"DIALOGUE ANALYSIS: {scenario_name}")
    else:
        print("DIALOGUE ANALYSIS")
    print("=" * 50)

    print(f"\nStructure:")
    print(f"  Total dialogue lines: {metrics.total_lines}")
    print(f"  Unique states: {metrics.unique_states}")
    print(f"  Branch points: {metrics.branch_points}")
    print(f"  Max branches at a state: {metrics.max_branches}")
    print(f"  Approximate paths: {metrics.approximate_paths}")

    print(f"\nQuality:")
    print(f"  Has start state: {'Yes' if metrics.has_start else 'No'}")
    print(f"  Has end state: {'Yes' if metrics.has_end else 'No'}")
    print(f"  Avg utterance length: {metrics.avg_utterance_length:.1f} chars")

    print(f"\nVariety:")
    print(f"  Styles used: {', '.join(metrics.styles_used) or 'None'}")
    print(f"  Meanings used: {len(metrics.meanings_used)} types")

    if metrics.speakers:
        print(f"\nSpeakers:")
        for agent, count in metrics.speakers.items():
            print(f"  {agent}: {count} speak actions")


def compare_dialogue_metrics(
    baseline: DialogueMetrics,
    improved: DialogueMetrics,
) -> Dict[str, Any]:
    """Compare two dialogue metrics and return improvement summary."""
    return {
        "lines_improvement": improved.total_lines - baseline.total_lines,
        "lines_ratio": (
            improved.total_lines / baseline.total_lines
            if baseline.total_lines > 0 else 0
        ),
        "states_improvement": improved.unique_states - baseline.unique_states,
        "branches_improvement": improved.branch_points - baseline.branch_points,
        "paths_improvement": improved.approximate_paths - baseline.approximate_paths,
        "variety_improvement": (
            len(improved.styles_used) - len(baseline.styles_used)
        ),
    }
