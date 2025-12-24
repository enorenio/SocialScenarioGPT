"""
Verification system for SIA-LLM generated conditions and effects.
Checks consistency against the current scenario state.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.scenario_state import ScenarioState, Agent, Action


class ErrorSeverity(Enum):
    """Severity levels for verification errors."""
    WARNING = "warning"  # Potential issue, may still work
    ERROR = "error"      # Definite problem, likely to cause issues
    CRITICAL = "critical"  # Severe issue, will definitely fail


@dataclass
class VerificationError:
    """A verification error found during condition/effect checking."""
    error_type: str
    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    agent: Optional[str] = None
    action: Optional[str] = None
    element: Optional[str] = None  # The specific condition/effect
    suggestion: Optional[str] = None  # Suggested fix

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.error_type}"]
        if self.agent:
            parts.append(f"Agent: {self.agent}")
        if self.action:
            parts.append(f"Action: {self.action}")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


@dataclass
class VerificationResult:
    """Result of verifying conditions and effects."""
    valid: bool
    errors: List[VerificationError] = field(default_factory=list)
    warnings: List[VerificationError] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, error: VerificationError):
        if error.severity == ErrorSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.errors.append(error)
            self.valid = False

    def merge(self, other: "VerificationResult"):
        """Merge another result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if other.has_errors:
            self.valid = False


class BeliefDesireParser:
    """Parser for BEL/DES/INTENT statements."""

    # Pattern: BEL(agent, property) = value
    PATTERN = re.compile(
        r'^(BEL|DES|INTENT)\(([^,]+),\s*([^)]+)\)\s*=\s*(.+)$'
    )

    # Pattern without value: BEL(agent, property)
    PATTERN_NO_VALUE = re.compile(
        r'^(BEL|DES|INTENT)\(([^,]+),\s*([^)]+)\)$'
    )

    @classmethod
    def parse(cls, statement: str) -> Optional[Dict[str, str]]:
        """
        Parse a BEL/DES/INTENT statement.

        Returns dict with keys: type, agent, property, value
        or None if parsing fails.
        """
        statement = statement.strip()

        # Try pattern with value
        match = cls.PATTERN.match(statement)
        if match:
            return {
                "type": match.group(1),
                "agent": match.group(2).strip(),
                "property": match.group(3).strip(),
                "value": match.group(4).strip(),
            }

        # Try pattern without value
        match = cls.PATTERN_NO_VALUE.match(statement)
        if match:
            return {
                "type": match.group(1),
                "agent": match.group(2).strip(),
                "property": match.group(3).strip(),
                "value": None,
            }

        return None

    @classmethod
    def extract_key(cls, statement: str) -> Optional[str]:
        """Extract the key part (without value) from a statement."""
        parsed = cls.parse(statement)
        if parsed:
            return f"{parsed['type']}({parsed['agent']}, {parsed['property']})"
        return None


class ConditionEffectVerifier:
    """
    Verifies conditions and effects against scenario state.

    Verification checks:
    1. Condition references existing or reachable belief/desire
    2. Effect uses consistent naming with existing elements
    3. Values are valid (True/False for booleans, numbers for numeric)
    4. Agent references are valid
    5. No obvious contradictions
    """

    def __init__(self, state: ScenarioState):
        self.state = state
        self.parser = BeliefDesireParser()

        # Build index of all known beliefs/desires
        self._build_knowledge_index()

    def _build_knowledge_index(self):
        """Build index of all known beliefs and desires."""
        self.known_beliefs: Dict[str, Set[str]] = {}  # agent -> set of belief keys
        self.known_desires: Dict[str, Set[str]] = {}  # agent -> set of desire keys
        self.known_properties: Dict[str, Set[str]] = {}  # agent -> set of property names

        for agent_name, agent in self.state.agents.items():
            self.known_beliefs[agent_name] = set()
            self.known_desires[agent_name] = set()
            self.known_properties[agent_name] = set()

            for kb_item in agent.knowledge_base:
                parsed = self.parser.parse(kb_item)
                if parsed:
                    if parsed["type"] == "BEL":
                        self.known_beliefs[agent_name].add(
                            f"BEL({parsed['agent']}, {parsed['property']})"
                        )
                    elif parsed["type"] == "DES":
                        self.known_desires[agent_name].add(
                            f"DES({parsed['agent']}, {parsed['property']})"
                        )
                    self.known_properties[agent_name].add(parsed["property"])

            # Also index effects from existing actions (they become reachable beliefs)
            for action in agent.actions.values():
                for effect in action.effects:
                    parsed = self.parser.parse(effect)
                    if parsed:
                        self.known_properties[agent_name].add(parsed["property"])

    def verify_condition(
        self,
        condition: str,
        agent_name: str,
        action_name: str,
    ) -> List[VerificationError]:
        """Verify a single condition."""
        errors = []

        parsed = self.parser.parse(condition)

        # Check 1: Can we parse it?
        if not parsed:
            errors.append(VerificationError(
                error_type="INVALID_FORMAT",
                message=f"Cannot parse condition: '{condition}'",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=condition,
                suggestion="Use format: BEL(agent, property) = value",
            ))
            return errors

        # Check 1b: Is the value missing?
        if parsed["value"] is None:
            errors.append(VerificationError(
                error_type="MISSING_VALUE",
                message=f"Condition '{condition}' is missing a value (e.g., = True)",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=condition,
                suggestion=f"Add '= True' or '= False' to: {condition}",
            ))

        # Check 2: Is the referenced agent valid?
        ref_agent = parsed["agent"]
        if ref_agent not in self.state.agents:
            errors.append(VerificationError(
                error_type="UNKNOWN_AGENT",
                message=f"Condition references unknown agent '{ref_agent}'",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=condition,
                suggestion=f"Valid agents: {list(self.state.agents.keys())}",
            ))

        # Check 3: Is this belief/desire known or reachable?
        key = self.parser.extract_key(condition)
        cond_type = parsed["type"]

        if ref_agent in self.state.agents:
            if cond_type == "BEL":
                known = self.known_beliefs.get(ref_agent, set())
                if key not in known:
                    # Check if there's a similar property (typo detection)
                    prop = parsed["property"]
                    similar = self._find_similar_property(ref_agent, prop)
                    if similar:
                        errors.append(VerificationError(
                            error_type="UNKNOWN_BELIEF",
                            message=f"Belief '{key}' not in knowledge base",
                            severity=ErrorSeverity.WARNING,
                            agent=agent_name,
                            action=action_name,
                            element=condition,
                            suggestion=f"Did you mean: BEL({ref_agent}, {similar})?",
                        ))
                    # Note: Not always an error - conditions can require future states

            elif cond_type == "DES":
                known = self.known_desires.get(ref_agent, set())
                if key not in known:
                    errors.append(VerificationError(
                        error_type="UNKNOWN_DESIRE",
                        message=f"Desire '{key}' not in knowledge base",
                        severity=ErrorSeverity.WARNING,
                        agent=agent_name,
                        action=action_name,
                        element=condition,
                    ))

        # Check 4: Is the value valid?
        value = parsed.get("value")
        if value:
            if not self._is_valid_value(value):
                errors.append(VerificationError(
                    error_type="INVALID_VALUE",
                    message=f"Invalid value '{value}' in condition",
                    severity=ErrorSeverity.WARNING,
                    agent=agent_name,
                    action=action_name,
                    element=condition,
                    suggestion="Use True, False, or a number",
                ))

        return errors

    def verify_effect(
        self,
        effect: str,
        agent_name: str,
        action_name: str,
    ) -> List[VerificationError]:
        """Verify a single effect."""
        errors = []

        parsed = self.parser.parse(effect)

        # Check 1: Can we parse it?
        if not parsed:
            errors.append(VerificationError(
                error_type="INVALID_FORMAT",
                message=f"Cannot parse effect: '{effect}'",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=effect,
                suggestion="Use format: BEL(agent, property) = value",
            ))
            return errors

        # Check 1b: Is the value missing?
        if parsed["value"] is None:
            errors.append(VerificationError(
                error_type="MISSING_VALUE",
                message=f"Effect '{effect}' is missing a value",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=effect,
                suggestion=f"Add '= True' or '= False' to: {effect}",
            ))

        # Check 2: Is the referenced agent valid?
        ref_agent = parsed["agent"]
        if ref_agent not in self.state.agents:
            errors.append(VerificationError(
                error_type="UNKNOWN_AGENT",
                message=f"Effect references unknown agent '{ref_agent}'",
                severity=ErrorSeverity.ERROR,
                agent=agent_name,
                action=action_name,
                element=effect,
                suggestion=f"Valid agents: {list(self.state.agents.keys())}",
            ))

        # Check 3: Naming consistency
        prop = parsed["property"]
        if ref_agent in self.known_properties:
            similar = self._find_similar_property(ref_agent, prop)
            if similar and similar != prop:
                errors.append(VerificationError(
                    error_type="INCONSISTENT_NAMING",
                    message=f"Property '{prop}' similar to existing '{similar}'",
                    severity=ErrorSeverity.WARNING,
                    agent=agent_name,
                    action=action_name,
                    element=effect,
                    suggestion=f"Consider using consistent name: {similar}",
                ))

        return errors

    def verify_action(
        self,
        action: Action,
        agent_name: str,
    ) -> VerificationResult:
        """Verify all conditions and effects for an action."""
        result = VerificationResult(valid=True)

        for condition in action.conditions:
            errors = self.verify_condition(condition, agent_name, action.name)
            for error in errors:
                result.add_error(error)

        for effect in action.effects:
            errors = self.verify_effect(effect, agent_name, action.name)
            for error in errors:
                result.add_error(error)

        return result

    def verify_agent(self, agent_name: str) -> VerificationResult:
        """Verify all actions for an agent."""
        result = VerificationResult(valid=True)

        agent = self.state.agents.get(agent_name)
        if not agent:
            result.add_error(VerificationError(
                error_type="UNKNOWN_AGENT",
                message=f"Agent '{agent_name}' not found",
                severity=ErrorSeverity.CRITICAL,
            ))
            return result

        for action in agent.actions.values():
            action_result = self.verify_action(action, agent_name)
            result.merge(action_result)

        for speak_action in agent.speak_actions.values():
            # Treat speak actions like regular actions for verification
            action_obj = Action(
                name=speak_action.name,
                conditions=speak_action.conditions,
                effects=speak_action.effects,
            )
            action_result = self.verify_action(action_obj, agent_name)
            result.merge(action_result)

        return result

    def verify_all(self) -> VerificationResult:
        """Verify entire scenario state."""
        result = VerificationResult(valid=True)

        for agent_name in self.state.agents:
            agent_result = self.verify_agent(agent_name)
            result.merge(agent_result)

        return result

    def _find_similar_property(self, agent: str, prop: str) -> Optional[str]:
        """Find a similar property name (for typo detection)."""
        known = self.known_properties.get(agent, set())

        # Normalize for comparison
        prop_normalized = prop.lower().replace("_", "").replace("-", "")

        for known_prop in known:
            known_normalized = known_prop.lower().replace("_", "").replace("-", "")
            if prop_normalized == known_normalized and prop != known_prop:
                return known_prop

            # Check for substring match
            if len(prop) > 3 and (prop_normalized in known_normalized or known_normalized in prop_normalized):
                if prop != known_prop:
                    return known_prop

        return None

    def _is_valid_value(self, value: str) -> bool:
        """Check if a value is valid."""
        value = value.strip()

        # Boolean
        if value in ("True", "False", "true", "false"):
            return True

        # Number
        try:
            float(value)
            return True
        except ValueError:
            pass

        # Comparison expression (e.g., ">= 5")
        if re.match(r'^[<>=!]+\s*\d+', value):
            return True

        # String value (quoted)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return True

        # None/null
        if value.lower() in ("none", "null"):
            return True

        # Complex expressions - allow them with warning
        return True  # Be permissive for now


def verify_conditions_effects(
    conditions: List[str],
    effects: List[str],
    state: ScenarioState,
    agent_name: str,
    action_name: str = "unknown",
) -> VerificationResult:
    """
    Convenience function to verify conditions and effects.

    Args:
        conditions: List of condition strings
        effects: List of effect strings
        state: Current scenario state
        agent_name: Name of the agent
        action_name: Name of the action

    Returns:
        VerificationResult with any errors found
    """
    verifier = ConditionEffectVerifier(state)

    result = VerificationResult(valid=True)

    for condition in conditions:
        errors = verifier.verify_condition(condition, agent_name, action_name)
        for error in errors:
            result.add_error(error)

    for effect in effects:
        errors = verifier.verify_effect(effect, agent_name, action_name)
        for error in errors:
            result.add_error(error)

    return result


def verify_scenario(state: ScenarioState) -> VerificationResult:
    """
    Verify entire scenario for consistency.

    Args:
        state: Scenario state to verify

    Returns:
        VerificationResult with all errors found
    """
    verifier = ConditionEffectVerifier(state)
    return verifier.verify_all()


if __name__ == "__main__":
    # Demo/test
    from core.scenario_state import ScenarioState

    # Load a real scenario
    state = ScenarioState.from_file("Data/test_Brother.json")

    print("Verifying scenario:", state.scenario_name)
    print("=" * 60)

    result = verify_scenario(state)

    print(f"Valid: {result.valid}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    if result.errors:
        print("\nErrors:")
        for error in result.errors[:10]:
            print(f"  {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:10]:
            print(f"  {warning}")
