"""
Error feedback formatting for LLM regeneration.
Converts verification errors into prompts for the LLM to fix issues.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.verification import VerificationError, VerificationResult, ErrorSeverity


@dataclass
class RegenerationRequest:
    """Request for LLM to regenerate conditions/effects."""
    action_name: str
    agent_name: str
    original_conditions: List[str]
    original_effects: List[str]
    errors: List[VerificationError]
    context: str  # Scenario context for reference

    def to_prompt(self) -> str:
        """Generate a prompt for the LLM to fix the errors."""
        return format_regeneration_prompt(self)


def format_errors_for_llm(
    errors: List[VerificationError],
    include_suggestions: bool = True,
) -> str:
    """
    Format verification errors as feedback for the LLM.

    Args:
        errors: List of verification errors
        include_suggestions: Whether to include fix suggestions

    Returns:
        Formatted error string for inclusion in prompt
    """
    if not errors:
        return ""

    lines = ["The following issues were found with the generated conditions/effects:"]
    lines.append("")

    for i, error in enumerate(errors, 1):
        severity_icon = {
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚫",
        }.get(error.severity, "•")

        lines.append(f"{i}. {severity_icon} {error.error_type}: {error.message}")

        if error.element:
            lines.append(f"   Problem element: {error.element}")

        if include_suggestions and error.suggestion:
            lines.append(f"   Suggestion: {error.suggestion}")

        lines.append("")

    return "\n".join(lines)


def format_regeneration_prompt(request: RegenerationRequest) -> str:
    """
    Format a complete regeneration prompt for the LLM.

    Args:
        request: Regeneration request with context and errors

    Returns:
        Complete prompt string
    """
    lines = []

    # Header
    lines.append("Please regenerate the conditions and effects for the following action.")
    lines.append("Fix the issues identified below while maintaining consistency with the scenario.")
    lines.append("")

    # Context
    lines.append("=== CONTEXT ===")
    lines.append(request.context)
    lines.append("")

    # Action info
    lines.append("=== ACTION TO FIX ===")
    lines.append(f"Agent: {request.agent_name}")
    lines.append(f"Action: {request.action_name}")
    lines.append("")

    # Original (problematic) output
    lines.append("=== ORIGINAL OUTPUT (with issues) ===")
    lines.append("Conditions:")
    for cond in request.original_conditions:
        lines.append(f"  - {cond}")
    lines.append("Effects:")
    for effect in request.original_effects:
        lines.append(f"  - {effect}")
    lines.append("")

    # Errors
    lines.append("=== ISSUES TO FIX ===")
    lines.append(format_errors_for_llm(request.errors))

    # Instructions
    lines.append("=== INSTRUCTIONS ===")
    lines.append("1. Fix all the issues listed above")
    lines.append("2. Ensure conditions reference existing beliefs/desires from the knowledge base")
    lines.append("3. Use consistent naming for properties (check existing properties)")
    lines.append("4. All statements must have values (e.g., BEL(agent, property) = True)")
    lines.append("5. Only reference agents that exist in the scenario")
    lines.append("")

    # Expected format
    lines.append("=== EXPECTED OUTPUT FORMAT ===")
    lines.append("Conditions:")
    lines.append("- [[BEL(agent, property) = value]]")
    lines.append("- [[DES(agent, property) = value]]")
    lines.append("")
    lines.append("Effects:")
    lines.append("- [[BEL(agent, property) = new_value]]")
    lines.append("- [[DES(agent, property) = new_value]]")
    lines.append("")

    lines.append("Please provide the corrected conditions and effects:")

    return "\n".join(lines)


def format_knowledge_base_context(
    agent_name: str,
    knowledge_base: List[str],
    all_agents: List[str],
) -> str:
    """
    Format knowledge base as context for error feedback.

    Args:
        agent_name: Name of the agent
        knowledge_base: Agent's knowledge base
        all_agents: List of all agent names in scenario

    Returns:
        Formatted context string
    """
    lines = []

    lines.append(f"Agent: {agent_name}")
    lines.append(f"All agents in scenario: {', '.join(all_agents)}")
    lines.append("")

    lines.append("Current knowledge base:")
    beliefs = [kb for kb in knowledge_base if kb.startswith("BEL(")]
    desires = [kb for kb in knowledge_base if kb.startswith("DES(")]

    if beliefs:
        lines.append("  Beliefs:")
        for b in beliefs:
            lines.append(f"    - {b}")

    if desires:
        lines.append("  Desires:")
        for d in desires:
            lines.append(f"    - {d}")

    return "\n".join(lines)


def create_regeneration_request(
    action_name: str,
    agent_name: str,
    conditions: List[str],
    effects: List[str],
    result: VerificationResult,
    state: Any,  # ScenarioState
) -> RegenerationRequest:
    """
    Create a regeneration request from verification result.

    Args:
        action_name: Name of the action with issues
        agent_name: Name of the agent
        conditions: Original conditions
        effects: Original effects
        result: Verification result with errors
        state: Scenario state for context

    Returns:
        RegenerationRequest object
    """
    # Build context from state
    agent = state.agents.get(agent_name)
    context = format_knowledge_base_context(
        agent_name,
        agent.knowledge_base if agent else [],
        list(state.agents.keys()),
    )

    # Filter errors for this action
    action_errors = [
        e for e in result.errors + result.warnings
        if e.action == action_name or e.action is None
    ]

    return RegenerationRequest(
        action_name=action_name,
        agent_name=agent_name,
        original_conditions=conditions,
        original_effects=effects,
        errors=action_errors,
        context=context,
    )


def format_fix_summary(
    original_errors: int,
    remaining_errors: int,
    attempt: int,
    max_attempts: int,
) -> str:
    """
    Format a summary of the fix attempt.

    Args:
        original_errors: Number of errors before fix
        remaining_errors: Number of errors after fix
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed

    Returns:
        Summary string
    """
    if remaining_errors == 0:
        return f"✅ All {original_errors} issues fixed on attempt {attempt}/{max_attempts}"
    elif remaining_errors < original_errors:
        fixed = original_errors - remaining_errors
        return f"⚠️ Fixed {fixed}/{original_errors} issues on attempt {attempt}/{max_attempts}, {remaining_errors} remaining"
    else:
        return f"❌ No improvement on attempt {attempt}/{max_attempts}, {remaining_errors} issues remain"


if __name__ == "__main__":
    # Demo
    from core.verification import VerificationError, ErrorSeverity

    errors = [
        VerificationError(
            error_type="MISSING_VALUE",
            message="Condition 'BEL(brother, fell_asleep)' is missing a value",
            severity=ErrorSeverity.ERROR,
            agent="brother",
            action="WAKE_UP(brother)",
            element="BEL(brother, fell_asleep)",
            suggestion="Add '= True' or '= False' to: BEL(brother, fell_asleep)",
        ),
        VerificationError(
            error_type="UNKNOWN_BELIEF",
            message="Belief 'BEL(brother, rest)' not in knowledge base",
            severity=ErrorSeverity.WARNING,
            agent="brother",
            action="SetAlarm(brother, None, 'in 8 hours')",
            element="BEL(brother, rest) = True",
            suggestion="Did you mean: DES(brother, rest)?",
        ),
    ]

    print("=== Error Feedback ===")
    print(format_errors_for_llm(errors))

    print("\n=== Knowledge Base Context ===")
    print(format_knowledge_base_context(
        "brother",
        [
            "BEL(brother, feeling_sick) = True",
            "BEL(brother, have_homework) = True",
            "DES(brother, do_homework) = True",
            "DES(brother, rest) = True",
        ],
        ["brother", "mom", "teacher"],
    ))
