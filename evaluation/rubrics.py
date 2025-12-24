"""
Evaluation rubrics for LLM-as-Judge system.

Based on the subjective evaluation dimensions from Antunes et al. (2023):
- Agent relevance
- Belief/desire coherence
- Action appropriateness
- Dialogue quality
- Emotional consistency
- Overall scenario quality
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class EvaluationDimension(Enum):
    """Evaluation dimensions matching the original paper."""
    AGENT_RELEVANCE = "agent_relevance"
    BELIEF_COHERENCE = "belief_coherence"
    DESIRE_APPROPRIATENESS = "desire_appropriateness"
    INTENTION_VALIDITY = "intention_validity"
    ACTION_FEASIBILITY = "action_feasibility"
    CONDITION_EFFECT_LOGIC = "condition_effect_logic"
    DIALOGUE_QUALITY = "dialogue_quality"
    DIALOGUE_NATURALNESS = "dialogue_naturalness"
    EMOTIONAL_CONSISTENCY = "emotional_consistency"
    OVERALL_COHERENCE = "overall_coherence"


@dataclass
class RubricLevel:
    """A single level in a Likert scale rubric."""
    score: int
    label: str
    description: str
    examples: List[str] = field(default_factory=list)


@dataclass
class EvaluationRubric:
    """Complete rubric for one evaluation dimension."""
    dimension: EvaluationDimension
    name: str
    description: str
    question: str  # The question to ask the LLM judge
    levels: List[RubricLevel]
    weight: float = 1.0  # Relative importance

    def get_prompt_section(self) -> str:
        """Generate the rubric section for LLM prompt."""
        lines = [
            f"### {self.name}",
            f"**Question:** {self.question}",
            "",
            "**Scoring Guide:**",
        ]

        for level in sorted(self.levels, key=lambda x: x.score):
            lines.append(f"  {level.score} - {level.label}: {level.description}")
            if level.examples:
                for ex in level.examples:
                    lines.append(f"      Example: {ex}")

        return "\n".join(lines)


# ============================================================================
# EVALUATION RUBRICS
# ============================================================================

AGENT_RELEVANCE_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.AGENT_RELEVANCE,
    name="Agent Relevance",
    description="Evaluates whether the generated agents are appropriate for the scenario",
    question="Are the agents appropriate and relevant to the scenario description?",
    weight=1.0,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Agents are irrelevant, missing key characters, or include nonsensical entities",
            examples=["Scenario about cooking but agents are 'Car' and 'Building'"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some agents relevant but key characters missing or extra irrelevant agents",
            examples=["Story mentions John and Mary but only 'Person1' is generated"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Main agents present but naming or roles could be improved",
            examples=["Correct agents but generic names like 'Agent1' instead of story names"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Agents match scenario well with appropriate names and implied roles",
            examples=["John and Mary from story are both agents with correct identities"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Agents perfectly match scenario with clear roles and relationships",
            examples=["All characters present, roles clear (e.g., 'John (customer)', 'Mary (clerk)')"],
        ),
    ],
)

BELIEF_COHERENCE_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.BELIEF_COHERENCE,
    name="Belief Coherence",
    description="Evaluates whether beliefs are logically consistent and grounded in the scenario",
    question="Are the agents' beliefs logically consistent and grounded in the scenario?",
    weight=1.2,  # Slightly higher weight - core BDI component
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Beliefs contradict each other or the scenario, or are nonsensical",
            examples=["BEL(John, hungry) = True and BEL(John, not_hungry) = True"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some beliefs valid but others contradict scenario or are irrelevant",
            examples=["Scenario about shopping but beliefs about flying airplanes"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Beliefs generally consistent but some gaps or minor inconsistencies",
            examples=["Missing beliefs about key scenario elements mentioned in text"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Beliefs well-grounded in scenario with few gaps",
            examples=["Beliefs cover main scenario facts, no contradictions"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Beliefs comprehensive, consistent, and capture nuanced scenario details",
            examples=["Beliefs capture both explicit facts and reasonable inferences"],
        ),
    ],
)

DESIRE_APPROPRIATENESS_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.DESIRE_APPROPRIATENESS,
    name="Desire Appropriateness",
    description="Evaluates whether desires are appropriate given the scenario and agent roles",
    question="Are the agents' desires appropriate for their roles and the scenario?",
    weight=1.2,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Desires irrelevant to scenario or contradict agent's apparent goals",
            examples=["Customer in store has desire to 'fly to moon'"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some desires relevant but key motivations missing or misattributed",
            examples=["Store clerk wants to buy items instead of help customer"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Main desires present but could better reflect scenario nuances",
            examples=["Generic 'complete task' instead of specific scenario goals"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Desires well-matched to scenario with clear agent motivations",
            examples=["Customer wants specific item, clerk wants to assist"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Desires capture complex motivations and potential conflicts",
            examples=["Competing desires that create interesting scenario dynamics"],
        ),
    ],
)

INTENTION_VALIDITY_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.INTENTION_VALIDITY,
    name="Intention Validity",
    description="Evaluates whether intentions logically follow from beliefs and desires",
    question="Do the intentions logically derive from the agents' beliefs and desires?",
    weight=1.0,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Intentions unrelated to beliefs/desires or logically impossible",
            examples=["Agent believes store is closed but intends to shop there"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some intentions valid but others don't follow from mental state",
            examples=["Intention to help but no desire or belief supporting it"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Intentions generally valid but derivation could be clearer",
            examples=["Intentions present but connection to beliefs/desires implicit"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Intentions clearly follow from beliefs and desires",
            examples=["Clear: believes X + desires Y → intends Z"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Intentions show sophisticated reasoning from mental state",
            examples=["Intentions account for multiple beliefs and prioritize desires"],
        ),
    ],
)

ACTION_FEASIBILITY_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.ACTION_FEASIBILITY,
    name="Action Feasibility",
    description="Evaluates whether actions are achievable and well-sequenced",
    question="Are the action plans feasible and properly sequenced to achieve intentions?",
    weight=1.5,  # Higher weight - directly affects intention completion
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Actions impossible, circular dependencies, or completely unsequenced",
            examples=["Action requires result of later action as precondition"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some actions feasible but many have unmet preconditions",
            examples=["First action requires conditions never established"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Most actions feasible but some gaps in action chains",
            examples=["70% of actions could execute in sequence"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Actions well-sequenced with most preconditions satisfiable",
            examples=["Clear progression from initial state to goal"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Actions form complete, executable plan from initial state to goal",
            examples=["100% of intentions completable through action sequences"],
        ),
    ],
)

CONDITION_EFFECT_LOGIC_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.CONDITION_EFFECT_LOGIC,
    name="Condition/Effect Logic",
    description="Evaluates the logical consistency of action preconditions and effects",
    question="Are action conditions and effects logically consistent and properly formatted?",
    weight=1.5,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Conditions/effects malformed, missing values, or logically impossible",
            examples=["BEL(Agent, property) without = Value", "Effect contradicts condition"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Many formatting errors or logical inconsistencies",
            examples=["50%+ of statements missing proper format"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Mostly correct format but some logical gaps",
            examples=["Effects don't fully capture action consequences"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Well-formatted with clear condition→effect relationships",
            examples=["Conditions check relevant beliefs, effects update appropriately"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Comprehensive, logically sound conditions and effects",
            examples=["Full causal chains, no missing values, effects propagate correctly"],
        ),
    ],
)

DIALOGUE_QUALITY_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.DIALOGUE_QUALITY,
    name="Dialogue Quality",
    description="Evaluates dialogue quantity, branching, and state machine structure",
    question="Does the dialogue have sufficient quantity, branching paths, and proper structure?",
    weight=1.0,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Minimal dialogue (1-3 lines), no branching, or broken structure",
            examples=["Only 2 dialogue lines with no state transitions"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Limited dialogue (4-6 lines), linear path, basic structure",
            examples=["5 lines, single path from start to end"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Moderate dialogue (7-10 lines), minimal branching",
            examples=["8 lines with one branch point"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Good dialogue (11-14 lines), multiple branches, clear structure",
            examples=["12 lines with 2-3 alternative paths"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Rich dialogue (15+ lines), meaningful branching, complete state machine",
            examples=["15+ lines, 3+ paths, covers positive/negative/neutral outcomes"],
        ),
    ],
)

DIALOGUE_NATURALNESS_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.DIALOGUE_NATURALNESS,
    name="Dialogue Naturalness",
    description="Evaluates whether dialogue sounds natural and character-appropriate",
    question="Does the dialogue sound natural and appropriate for each character?",
    weight=0.8,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Dialogue robotic, out of character, or nonsensical",
            examples=["All characters speak identically", "Unnatural phrasing"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Dialogue awkward or doesn't match character roles",
            examples=["Child speaks like adult professional"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Dialogue acceptable but generic, lacks personality",
            examples=["Functional but could be said by anyone"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Dialogue natural with some character-specific elements",
            examples=["Characters have distinct speech patterns"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Dialogue highly natural with clear character voices",
            examples=["Each character's personality evident in speech"],
        ),
    ],
)

EMOTIONAL_CONSISTENCY_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.EMOTIONAL_CONSISTENCY,
    name="Emotional Consistency",
    description="Evaluates whether emotions follow OCC model and scenario context",
    question="Are emotions consistent with the OCC model and appropriate for scenario events?",
    weight=0.8,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Emotions contradict events or use invalid OCC categories",
            examples=["Joy after negative event", "Made-up emotion names"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Some emotions valid but others mismatched to events",
            examples=["Neutral emotions for highly charged events"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Emotions generally appropriate but lack nuance",
            examples=["Only basic emotions used (joy, distress)"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Emotions well-matched with appropriate OCC categories",
            examples=["Gratitude for help, hope for future outcomes"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Rich emotional modeling with proper OCC emotion types",
            examples=["Distinguishes hope vs. satisfaction, uses compound emotions"],
        ),
    ],
)

OVERALL_COHERENCE_RUBRIC = EvaluationRubric(
    dimension=EvaluationDimension.OVERALL_COHERENCE,
    name="Overall Coherence",
    description="Evaluates the overall quality and coherence of the complete scenario",
    question="How coherent and well-integrated is the overall scenario?",
    weight=1.5,
    levels=[
        RubricLevel(
            score=1,
            label="Poor",
            description="Scenario fragmented, contradictory, or unusable for simulation",
            examples=["Components don't connect, would fail in FAtiMA"],
        ),
        RubricLevel(
            score=2,
            label="Below Average",
            description="Major coherence issues affecting usability",
            examples=["Some components work but overall flow broken"],
        ),
        RubricLevel(
            score=3,
            label="Average",
            description="Scenario functional but with notable gaps or inconsistencies",
            examples=["Would run but produce unexpected behaviors"],
        ),
        RubricLevel(
            score=4,
            label="Good",
            description="Scenario coherent with minor issues",
            examples=["Would run well with small adjustments"],
        ),
        RubricLevel(
            score=5,
            label="Excellent",
            description="Scenario fully coherent, ready for simulation",
            examples=["All components integrate seamlessly, clear narrative flow"],
        ),
    ],
)


# ============================================================================
# RUBRIC REGISTRY
# ============================================================================

EVALUATION_RUBRICS: Dict[EvaluationDimension, EvaluationRubric] = {
    EvaluationDimension.AGENT_RELEVANCE: AGENT_RELEVANCE_RUBRIC,
    EvaluationDimension.BELIEF_COHERENCE: BELIEF_COHERENCE_RUBRIC,
    EvaluationDimension.DESIRE_APPROPRIATENESS: DESIRE_APPROPRIATENESS_RUBRIC,
    EvaluationDimension.INTENTION_VALIDITY: INTENTION_VALIDITY_RUBRIC,
    EvaluationDimension.ACTION_FEASIBILITY: ACTION_FEASIBILITY_RUBRIC,
    EvaluationDimension.CONDITION_EFFECT_LOGIC: CONDITION_EFFECT_LOGIC_RUBRIC,
    EvaluationDimension.DIALOGUE_QUALITY: DIALOGUE_QUALITY_RUBRIC,
    EvaluationDimension.DIALOGUE_NATURALNESS: DIALOGUE_NATURALNESS_RUBRIC,
    EvaluationDimension.EMOTIONAL_CONSISTENCY: EMOTIONAL_CONSISTENCY_RUBRIC,
    EvaluationDimension.OVERALL_COHERENCE: OVERALL_COHERENCE_RUBRIC,
}


def get_rubric(dimension: EvaluationDimension) -> EvaluationRubric:
    """Get rubric for a specific dimension."""
    return EVALUATION_RUBRICS[dimension]


def list_dimensions() -> List[EvaluationDimension]:
    """List all evaluation dimensions."""
    return list(EVALUATION_RUBRICS.keys())


def get_all_rubrics_prompt() -> str:
    """Generate complete rubrics section for LLM prompt."""
    sections = []
    for dim in EvaluationDimension:
        rubric = EVALUATION_RUBRICS[dim]
        sections.append(rubric.get_prompt_section())
    return "\n\n".join(sections)


def calculate_weighted_average(
    scores: Dict[EvaluationDimension, int]
) -> float:
    """Calculate weighted average score across dimensions."""
    total_weight = 0.0
    weighted_sum = 0.0

    for dim, score in scores.items():
        rubric = EVALUATION_RUBRICS[dim]
        weighted_sum += score * rubric.weight
        total_weight += rubric.weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight
