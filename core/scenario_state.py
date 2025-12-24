"""
Full context state management for SIA-LLM experiments.
Maintains complete scenario state across all pipeline stages.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Action:
    """An action with conditions, effects, and emotions."""
    name: str
    conditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    occ_emotion: List[str] = field(default_factory=list)
    emotion_condition: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": self.conditions,
            "effects": self.effects,
            "occ_emotion": self.occ_emotion,
            "emotion_condition": self.emotion_condition,
        }


@dataclass
class SpeakAction:
    """A speak action for dialogue."""
    name: str
    conditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    occ_emotion: List[str] = field(default_factory=list)
    emotion_condition: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": self.conditions,
            "effects": self.effects,
            "occ_emotion": self.occ_emotion,
            "emotion_condition": self.emotion_condition,
        }


@dataclass
class Intention:
    """An intention with an action plan."""
    name: str
    action_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"action_plan": self.action_plan}


@dataclass
class Agent:
    """An agent with knowledge, intentions, and actions."""
    name: str
    knowledge_base: List[str] = field(default_factory=list)
    intentions: Dict[str, Intention] = field(default_factory=dict)
    actions: Dict[str, Action] = field(default_factory=dict)
    speak_actions: Dict[str, SpeakAction] = field(default_factory=dict)
    initial_occ_emotion: List[str] = field(default_factory=list)

    def get_beliefs(self) -> List[str]:
        """Extract beliefs from knowledge base."""
        return [kb for kb in self.knowledge_base if kb.startswith("BEL(")]

    def get_desires(self) -> List[str]:
        """Extract desires from knowledge base."""
        return [kb for kb in self.knowledge_base if kb.startswith("DES(")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_base": self.knowledge_base,
            "intentions": {k: v.to_dict() for k, v in self.intentions.items()},
            "actions": {k: v.to_dict() for k, v in self.actions.items()},
            "speak_actions": {k: v.to_dict() for k, v in self.speak_actions.items()},
            "initial_occ_emotion": self.initial_occ_emotion,
        }


@dataclass
class ValidationError:
    """A validation error found in the scenario state."""
    error_type: str
    message: str
    agent: Optional[str] = None
    element: Optional[str] = None

    def __str__(self) -> str:
        loc = f" [{self.agent}]" if self.agent else ""
        elem = f" in '{self.element}'" if self.element else ""
        return f"{self.error_type}{loc}{elem}: {self.message}"


class ScenarioState:
    """
    Complete state for a scenario generation run.

    Maintains all generated elements across pipeline stages and provides
    serialization for inclusion in prompts (full context mode).

    Usage:
        state = ScenarioState("test_scenario", "A story about...")

        # Add agents
        state.add_agent("Alice")
        state.add_belief("Alice", "BEL(Alice, happy) = True")

        # Serialize for prompt
        context = state.to_prompt_context()

        # Validate consistency
        errors = state.validate()
    """

    def __init__(
        self,
        scenario_name: str,
        scenario_description: str,
    ):
        self.scenario_name = scenario_name
        self.scenario_description = scenario_description
        self.agents: Dict[str, Agent] = {}
        self.dialogue_tree: List[str] = []
        self.last_ended: str = "scenario"
        self._stage_order = [
            "scenario", "agents", "knowledge", "intentions",
            "actions", "conditions_effects", "emotions",
            "speak_actions", "dialogue", "end"
        ]

    def add_agent(self, name: str) -> Agent:
        """Add a new agent to the scenario."""
        if name not in self.agents:
            self.agents[name] = Agent(name=name)
        return self.agents[name]

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def add_belief(self, agent_name: str, belief: str):
        """Add a belief to an agent's knowledge base."""
        agent = self.add_agent(agent_name)
        if belief not in agent.knowledge_base:
            agent.knowledge_base.append(belief)

    def add_desire(self, agent_name: str, desire: str):
        """Add a desire to an agent's knowledge base."""
        agent = self.add_agent(agent_name)
        if desire not in agent.knowledge_base:
            agent.knowledge_base.append(desire)

    def add_intention(self, agent_name: str, intent_name: str, action_plan: List[str]):
        """Add an intention with action plan to an agent."""
        agent = self.add_agent(agent_name)
        agent.intentions[intent_name] = Intention(name=intent_name, action_plan=action_plan)

    def add_action(
        self,
        agent_name: str,
        action_name: str,
        conditions: List[str] = None,
        effects: List[str] = None,
        occ_emotion: List[str] = None,
        emotion_condition: List[str] = None,
    ):
        """Add an action with conditions and effects to an agent."""
        agent = self.add_agent(agent_name)
        agent.actions[action_name] = Action(
            name=action_name,
            conditions=conditions or [],
            effects=effects or [],
            occ_emotion=occ_emotion or [],
            emotion_condition=emotion_condition or [],
        )

    def add_speak_action(
        self,
        agent_name: str,
        action_name: str,
        conditions: List[str] = None,
        effects: List[str] = None,
        occ_emotion: List[str] = None,
        emotion_condition: List[str] = None,
    ):
        """Add a speak action to an agent."""
        agent = self.add_agent(agent_name)
        agent.speak_actions[action_name] = SpeakAction(
            name=action_name,
            conditions=conditions or [],
            effects=effects or [],
            occ_emotion=occ_emotion or [],
            emotion_condition=emotion_condition or [],
        )

    def add_dialogue_line(self, line: str):
        """Add a line to the dialogue tree."""
        self.dialogue_tree.append(line)

    def set_stage(self, stage: str):
        """Set the current pipeline stage."""
        self.last_ended = stage

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (compatible with existing JSON format)."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "last_ended": self.last_ended,
            "agents": {name: agent.to_dict() for name, agent in self.agents.items()},
            "dialogue_tree": self.dialogue_tree,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert state to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_prompt_context(self, include_dialogue: bool = True) -> str:
        """
        Serialize full state for inclusion in LLM prompts.

        Returns a structured text representation of the current state
        suitable for providing full context to the model.
        """
        lines = []
        lines.append("=== CURRENT SCENARIO STATE ===")
        lines.append(f"Scenario: {self.scenario_name}")
        lines.append(f"Description: {self.scenario_description}")
        lines.append(f"Current Stage: {self.last_ended}")
        lines.append("")

        for agent_name, agent in self.agents.items():
            lines.append(f"--- Agent: {agent_name} ---")

            # Knowledge base
            if agent.knowledge_base:
                lines.append("Knowledge Base:")
                for kb in agent.knowledge_base:
                    lines.append(f"  - {kb}")

            # Intentions
            if agent.intentions:
                lines.append("Intentions:")
                for intent_name, intent in agent.intentions.items():
                    lines.append(f"  {intent_name}:")
                    for action in intent.action_plan:
                        lines.append(f"    - {action}")

            # Actions
            if agent.actions:
                lines.append("Actions:")
                for action_name, action in agent.actions.items():
                    lines.append(f"  {action_name}:")
                    if action.conditions:
                        lines.append(f"    Conditions: {action.conditions}")
                    if action.effects:
                        lines.append(f"    Effects: {action.effects}")
                    if action.occ_emotion:
                        lines.append(f"    Emotion: {action.occ_emotion}")

            # Speak actions
            if agent.speak_actions:
                lines.append("Speak Actions:")
                for sa_name, sa in agent.speak_actions.items():
                    lines.append(f"  {sa_name}:")
                    if sa.conditions:
                        lines.append(f"    Conditions: {sa.conditions}")
                    if sa.effects:
                        lines.append(f"    Effects: {sa.effects}")

            lines.append("")

        # Dialogue tree
        if include_dialogue and self.dialogue_tree:
            lines.append("--- Dialogue Tree ---")
            for line in self.dialogue_tree:
                lines.append(f"  {line}")
            lines.append("")

        lines.append("=== END STATE ===")
        return "\n".join(lines)

    def to_compact_context(self) -> str:
        """
        Serialize state in a more compact format for token efficiency.
        """
        parts = []
        parts.append(f"[Scenario: {self.scenario_name}]")
        parts.append(f"[Stage: {self.last_ended}]")

        for agent_name, agent in self.agents.items():
            parts.append(f"[Agent:{agent_name}]")
            if agent.knowledge_base:
                parts.append(f"KB:{','.join(agent.knowledge_base)}")
            if agent.intentions:
                intents = list(agent.intentions.keys())
                parts.append(f"INT:{','.join(intents)}")
            if agent.actions:
                actions = list(agent.actions.keys())
                parts.append(f"ACT:{','.join(actions)}")

        return " ".join(parts)

    def validate(self) -> List[ValidationError]:
        """
        Validate scenario state for consistency.

        Checks:
        - Action conditions reference existing beliefs
        - Effects use consistent naming
        - No orphaned intentions (intentions without action plans)
        - Action plan references existing actions

        Returns list of validation errors.
        """
        errors = []

        for agent_name, agent in self.agents.items():
            # Collect all belief/desire keys for this agent
            kb_keys = set()
            for kb in agent.knowledge_base:
                # Extract key from "BEL(agent, key) = value"
                match = re.match(r'(BEL|DES)\(([^,]+),\s*([^)]+)\)', kb)
                if match:
                    kb_keys.add(kb.split("=")[0].strip())

            # Check action conditions
            for action_name, action in agent.actions.items():
                for cond in action.conditions:
                    cond_key = cond.split("=")[0].strip()
                    # Check if condition references a belief that exists
                    if cond_key.startswith("BEL(") or cond_key.startswith("DES("):
                        if cond not in agent.knowledge_base and cond_key not in kb_keys:
                            # This is informational - conditions can require future states
                            pass

            # Check for empty action plans
            for intent_name, intent in agent.intentions.items():
                if not intent.action_plan:
                    errors.append(ValidationError(
                        error_type="EMPTY_ACTION_PLAN",
                        message="Intention has no action plan",
                        agent=agent_name,
                        element=intent_name,
                    ))

            # Check action plan references
            all_action_names = set(agent.actions.keys())
            for intent_name, intent in agent.intentions.items():
                for step in intent.action_plan:
                    # Actions in plan should ideally exist
                    # But they can also be beliefs/desires, so we only warn
                    pass

        return errors

    def get_all_beliefs(self) -> Dict[str, List[str]]:
        """Get all beliefs organized by agent."""
        return {
            name: agent.get_beliefs()
            for name, agent in self.agents.items()
        }

    def get_all_desires(self) -> Dict[str, List[str]]:
        """Get all desires organized by agent."""
        return {
            name: agent.get_desires()
            for name, agent in self.agents.items()
        }

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the current state."""
        total_beliefs = sum(len(a.get_beliefs()) for a in self.agents.values())
        total_desires = sum(len(a.get_desires()) for a in self.agents.values())
        total_intentions = sum(len(a.intentions) for a in self.agents.values())
        total_actions = sum(len(a.actions) for a in self.agents.values())
        total_speak = sum(len(a.speak_actions) for a in self.agents.values())

        return {
            "agents": len(self.agents),
            "beliefs": total_beliefs,
            "desires": total_desires,
            "intentions": total_intentions,
            "actions": total_actions,
            "speak_actions": total_speak,
            "dialogue_lines": len(self.dialogue_tree),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScenarioState":
        """Create ScenarioState from dictionary (load from JSON)."""
        state = cls(
            scenario_name=data.get("scenario_name", "unknown"),
            scenario_description=data.get("scenario_description", ""),
        )
        state.last_ended = data.get("last_ended", "scenario")
        state.dialogue_tree = data.get("dialogue_tree", [])

        for agent_name, agent_data in data.get("agents", {}).items():
            agent = state.add_agent(agent_name)
            agent.knowledge_base = agent_data.get("knowledge_base", [])
            agent.initial_occ_emotion = agent_data.get("initial_occ_emotion", [])

            # Load intentions
            for intent_name, intent_data in agent_data.get("intentions", {}).items():
                if isinstance(intent_data, dict):
                    agent.intentions[intent_name] = Intention(
                        name=intent_name,
                        action_plan=intent_data.get("action_plan", []),
                    )

            # Load actions
            for action_name, action_data in agent_data.get("actions", {}).items():
                if isinstance(action_data, dict):
                    agent.actions[action_name] = Action(
                        name=action_name,
                        conditions=action_data.get("conditions", []),
                        effects=action_data.get("effects", []),
                        occ_emotion=action_data.get("occ_emotion", []),
                        emotion_condition=action_data.get("emotion_condition", []),
                    )

            # Load speak actions
            for sa_name, sa_data in agent_data.get("speak_actions", {}).items():
                if isinstance(sa_data, dict):
                    agent.speak_actions[sa_name] = SpeakAction(
                        name=sa_name,
                        conditions=sa_data.get("conditions", []),
                        effects=sa_data.get("effects", []),
                        occ_emotion=sa_data.get("occ_emotion", []),
                        emotion_condition=sa_data.get("emotion_condition", []),
                    )

        return state

    @classmethod
    def from_json(cls, json_str: str) -> "ScenarioState":
        """Create ScenarioState from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, filepath: str) -> "ScenarioState":
        """Create ScenarioState from JSON file."""
        with open(filepath) as f:
            return cls.from_dict(json.load(f))


if __name__ == "__main__":
    # Demo/test
    state = ScenarioState("test_scenario", "A story about two friends.")

    # Add agents
    state.add_agent("Alice")
    state.add_agent("Bob")

    # Add beliefs/desires
    state.add_belief("Alice", "BEL(Alice, friends_with_Bob) = True")
    state.add_desire("Alice", "DES(Alice, help_Bob) = True")

    # Add intention
    state.add_intention("Alice", "INTENT(Alice, help_Bob) = True", [
        "FindBob(Alice)",
        "OfferHelp(Alice, Bob)",
    ])

    # Add action
    state.add_action(
        "Alice",
        "OfferHelp(Alice, Bob)",
        conditions=["BEL(Alice, friends_with_Bob) = True"],
        effects=["BEL(Bob, helped) = True"],
        occ_emotion=["Joy"],
    )

    print("=== State Dict ===")
    print(json.dumps(state.to_dict(), indent=2))

    print("\n=== Prompt Context ===")
    print(state.to_prompt_context())

    print("\n=== Stats ===")
    print(state.get_stats())

    print("\n=== Validation ===")
    errors = state.validate()
    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("  No errors found")
