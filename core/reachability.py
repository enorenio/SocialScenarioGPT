"""
Symbolic reachability analysis for SIA-LLM scenarios.
Simulates action execution to check if intentions can be completed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from core.scenario_state import ScenarioState, Agent, Action
from core.verification import BeliefDesireParser


class ActionStatus(Enum):
    """Status of an action during simulation."""
    PENDING = "pending"
    EXECUTABLE = "executable"
    BLOCKED = "blocked"
    EXECUTED = "executed"


@dataclass
class KnowledgeState:
    """
    Represents the current state of an agent's knowledge.
    Used for simulating action execution.
    """
    beliefs: Dict[str, str] = field(default_factory=dict)  # key -> value
    desires: Dict[str, str] = field(default_factory=dict)  # key -> value

    def copy(self) -> "KnowledgeState":
        """Create a copy of this state."""
        new_state = KnowledgeState()
        new_state.beliefs = dict(self.beliefs)
        new_state.desires = dict(self.desires)
        return new_state

    def has(self, key: str, value: str) -> bool:
        """Check if a belief/desire exists with the given value."""
        if key.startswith("BEL("):
            return self.beliefs.get(key) == value
        elif key.startswith("DES("):
            return self.desires.get(key) == value
        return False

    def set(self, key: str, value: str):
        """Set a belief/desire value."""
        if key.startswith("BEL("):
            self.beliefs[key] = value
        elif key.startswith("DES("):
            self.desires[key] = value

    def check_condition(self, condition: str, parser: BeliefDesireParser) -> bool:
        """Check if a condition is satisfied."""
        parsed = parser.parse(condition)
        if not parsed:
            return False

        key = f"{parsed['type']}({parsed['agent']}, {parsed['property']})"
        value = parsed.get("value", "True")

        if parsed["type"] == "BEL":
            return self.beliefs.get(key) == value
        elif parsed["type"] == "DES":
            return self.desires.get(key) == value

        return False

    def apply_effect(self, effect: str, parser: BeliefDesireParser):
        """Apply an effect to update the state."""
        parsed = parser.parse(effect)
        if not parsed:
            return

        key = f"{parsed['type']}({parsed['agent']}, {parsed['property']})"
        value = parsed.get("value", "True")

        if parsed["type"] == "BEL":
            self.beliefs[key] = value
        elif parsed["type"] == "DES":
            self.desires[key] = value

    @classmethod
    def from_knowledge_base(cls, knowledge_base: List[str]) -> "KnowledgeState":
        """Create state from an agent's knowledge base."""
        state = cls()
        parser = BeliefDesireParser()

        for item in knowledge_base:
            parsed = parser.parse(item)
            if parsed:
                key = f"{parsed['type']}({parsed['agent']}, {parsed['property']})"
                value = parsed.get("value", "True")

                if parsed["type"] == "BEL":
                    state.beliefs[key] = value
                elif parsed["type"] == "DES":
                    state.desires[key] = value

        return state


@dataclass
class ActionNode:
    """Node in the action dependency graph."""
    name: str
    conditions: List[str]
    effects: List[str]
    required_beliefs: Set[str] = field(default_factory=set)
    provided_beliefs: Set[str] = field(default_factory=set)
    status: ActionStatus = ActionStatus.PENDING


@dataclass
class IntentionAnalysis:
    """Analysis result for a single intention."""
    intention_name: str
    action_plan: List[str]
    total_actions: int
    executable_actions: int
    blocked_actions: int
    completion_possible: bool
    blocking_conditions: List[str]
    execution_trace: List[str]

    @property
    def completion_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.executable_actions / self.total_actions


@dataclass
class AgentAnalysis:
    """Analysis result for a single agent."""
    agent_name: str
    intentions: List[IntentionAnalysis]
    total_intentions: int
    completable_intentions: int
    immediately_executable_actions: int
    total_actions: int

    @property
    def intention_completion_rate(self) -> float:
        if self.total_intentions == 0:
            return 0.0
        return self.completable_intentions / self.total_intentions

    @property
    def action_executability_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.immediately_executable_actions / self.total_actions


@dataclass
class ScenarioAnalysis:
    """Complete analysis for a scenario."""
    scenario_name: str
    agents: List[AgentAnalysis]
    total_intentions: int
    completable_intentions: int
    total_actions: int
    immediately_executable_actions: int

    @property
    def intention_completion_rate(self) -> float:
        if self.total_intentions == 0:
            return 0.0
        return self.completable_intentions / self.total_intentions

    @property
    def action_executability_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.immediately_executable_actions / self.total_actions


class ReachabilityAnalyzer:
    """
    Analyzes reachability of intentions in a scenario.

    Simulates action execution to determine:
    - Which actions are immediately executable
    - Which intentions can be completed
    - What conditions block action execution
    """

    def __init__(self, state: ScenarioState):
        self.state = state
        self.parser = BeliefDesireParser()

    def check_action_executable(
        self,
        action: Action,
        knowledge_state: KnowledgeState,
    ) -> Tuple[bool, List[str]]:
        """
        Check if an action is executable given current knowledge state.

        Returns:
            Tuple of (is_executable, list of blocking conditions)
        """
        blocking = []

        for condition in action.conditions:
            if not knowledge_state.check_condition(condition, self.parser):
                blocking.append(condition)

        return len(blocking) == 0, blocking

    def execute_action(
        self,
        action: Action,
        knowledge_state: KnowledgeState,
    ) -> KnowledgeState:
        """
        Execute an action and return the new knowledge state.
        """
        new_state = knowledge_state.copy()

        for effect in action.effects:
            new_state.apply_effect(effect, self.parser)

        return new_state

    def analyze_intention(
        self,
        agent: Agent,
        intention_name: str,
    ) -> IntentionAnalysis:
        """
        Analyze if an intention can be completed.

        Simulates executing the action plan step by step.
        """
        intention = agent.intentions.get(intention_name)
        if not intention:
            return IntentionAnalysis(
                intention_name=intention_name,
                action_plan=[],
                total_actions=0,
                executable_actions=0,
                blocked_actions=0,
                completion_possible=False,
                blocking_conditions=["Intention not found"],
                execution_trace=[],
            )

        # Start with initial knowledge state
        knowledge_state = KnowledgeState.from_knowledge_base(agent.knowledge_base)

        executable_count = 0
        blocked_count = 0
        all_blocking = []
        execution_trace = []
        completion_possible = True

        for step in intention.action_plan:
            # Check if this step is an action or a belief/desire assertion
            if step in agent.actions:
                action = agent.actions[step]
                is_executable, blocking = self.check_action_executable(
                    action, knowledge_state
                )

                if is_executable:
                    executable_count += 1
                    knowledge_state = self.execute_action(action, knowledge_state)
                    execution_trace.append(f"✓ {step}")
                else:
                    blocked_count += 1
                    completion_possible = False
                    all_blocking.extend(blocking)
                    execution_trace.append(f"✗ {step} (blocked by: {blocking})")
            else:
                # It might be a belief/desire in the plan (not an action)
                # Try to parse and apply it
                parsed = self.parser.parse(step)
                if parsed:
                    key = f"{parsed['type']}({parsed['agent']}, {parsed['property']})"
                    value = parsed.get("value", "True")

                    if parsed["type"] == "BEL":
                        # Check if this belief is already true
                        if knowledge_state.beliefs.get(key) == value:
                            execution_trace.append(f"~ {step} (already true)")
                        else:
                            execution_trace.append(f"? {step} (belief not in state)")
                    elif parsed["type"] == "DES":
                        if knowledge_state.desires.get(key) == value:
                            execution_trace.append(f"~ {step} (already true)")
                        else:
                            execution_trace.append(f"? {step} (desire not in state)")
                else:
                    # Unknown step type - might be action not defined
                    execution_trace.append(f"? {step} (action not found)")

        return IntentionAnalysis(
            intention_name=intention_name,
            action_plan=intention.action_plan,
            total_actions=len([s for s in intention.action_plan if s in agent.actions]),
            executable_actions=executable_count,
            blocked_actions=blocked_count,
            completion_possible=completion_possible and executable_count > 0,
            blocking_conditions=list(set(all_blocking)),
            execution_trace=execution_trace,
        )

    def analyze_agent(self, agent_name: str) -> AgentAnalysis:
        """Analyze all intentions for an agent."""
        agent = self.state.agents.get(agent_name)
        if not agent:
            return AgentAnalysis(
                agent_name=agent_name,
                intentions=[],
                total_intentions=0,
                completable_intentions=0,
                immediately_executable_actions=0,
                total_actions=0,
            )

        # Analyze each intention
        intention_analyses = []
        for intention_name in agent.intentions:
            analysis = self.analyze_intention(agent, intention_name)
            intention_analyses.append(analysis)

        # Count immediately executable actions (from initial state)
        initial_state = KnowledgeState.from_knowledge_base(agent.knowledge_base)
        immediately_executable = 0

        for action in agent.actions.values():
            is_executable, _ = self.check_action_executable(action, initial_state)
            if is_executable:
                immediately_executable += 1

        return AgentAnalysis(
            agent_name=agent_name,
            intentions=intention_analyses,
            total_intentions=len(intention_analyses),
            completable_intentions=sum(1 for i in intention_analyses if i.completion_possible),
            immediately_executable_actions=immediately_executable,
            total_actions=len(agent.actions),
        )

    def analyze_scenario(self) -> ScenarioAnalysis:
        """Analyze the entire scenario."""
        agent_analyses = []
        total_intentions = 0
        completable_intentions = 0
        total_actions = 0
        immediately_executable = 0

        for agent_name in self.state.agents:
            analysis = self.analyze_agent(agent_name)
            agent_analyses.append(analysis)

            total_intentions += analysis.total_intentions
            completable_intentions += analysis.completable_intentions
            total_actions += analysis.total_actions
            immediately_executable += analysis.immediately_executable_actions

        return ScenarioAnalysis(
            scenario_name=self.state.scenario_name,
            agents=agent_analyses,
            total_intentions=total_intentions,
            completable_intentions=completable_intentions,
            total_actions=total_actions,
            immediately_executable_actions=immediately_executable,
        )


def analyze_scenario_reachability(state: ScenarioState) -> ScenarioAnalysis:
    """
    Convenience function to analyze a scenario.

    Args:
        state: Scenario state to analyze

    Returns:
        ScenarioAnalysis with reachability results
    """
    analyzer = ReachabilityAnalyzer(state)
    return analyzer.analyze_scenario()


def print_analysis_report(analysis: ScenarioAnalysis):
    """Print a formatted analysis report."""
    print("=" * 70)
    print(f"REACHABILITY ANALYSIS: {analysis.scenario_name}")
    print("=" * 70)
    print()
    print(f"Total Intentions: {analysis.total_intentions}")
    print(f"Completable Intentions: {analysis.completable_intentions}")
    print(f"Intention Completion Rate: {analysis.intention_completion_rate*100:.1f}%")
    print()
    print(f"Total Actions: {analysis.total_actions}")
    print(f"Immediately Executable: {analysis.immediately_executable_actions}")
    print(f"Action Executability Rate: {analysis.action_executability_rate*100:.1f}%")
    print()

    for agent in analysis.agents:
        print("-" * 50)
        print(f"Agent: {agent.agent_name}")
        print(f"  Intentions: {agent.completable_intentions}/{agent.total_intentions} completable")
        print(f"  Actions: {agent.immediately_executable_actions}/{agent.total_actions} immediately executable")

        for intention in agent.intentions:
            status = "✓" if intention.completion_possible else "✗"
            print(f"\n  {status} {intention.intention_name}")
            print(f"    Actions: {intention.executable_actions}/{intention.total_actions} executable")
            if intention.blocking_conditions:
                print(f"    Blocked by: {intention.blocking_conditions[:3]}")
            if intention.execution_trace:
                print("    Trace:")
                for step in intention.execution_trace[:5]:
                    print(f"      {step}")
                if len(intention.execution_trace) > 5:
                    print(f"      ... and {len(intention.execution_trace) - 5} more steps")


if __name__ == "__main__":
    # Demo/test
    state = ScenarioState.from_file("Data/test_Brother.json")
    analysis = analyze_scenario_reachability(state)
    print_analysis_report(analysis)
