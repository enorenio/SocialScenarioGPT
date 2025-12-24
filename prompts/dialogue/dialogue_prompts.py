"""
Improved dialogue generation prompts for SIA-LLM.

Key improvements over original:
1. Character personality integration
2. Multi-path dialogue trees (branching)
3. Emotional context awareness
4. Richer dialogue variations
5. Better state machine coverage

Baseline metrics to improve:
- Average dialogue lines: 5.30 per scenario
- Target: 15-20 lines per scenario
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class PersonalityTrait(Enum):
    """Big Five personality traits for character dialogue."""
    OPENNESS = "openness"  # Creative, curious vs conventional
    CONSCIENTIOUSNESS = "conscientiousness"  # Organized, dependable vs careless
    EXTRAVERSION = "extraversion"  # Outgoing, energetic vs reserved
    AGREEABLENESS = "agreeableness"  # Friendly, compassionate vs antagonistic
    NEUROTICISM = "neuroticism"  # Sensitive, nervous vs secure


class DialogueStyle(Enum):
    """Dialogue style markers."""
    FORMAL = "formal"
    CASUAL = "casual"
    EMOTIONAL = "emotional"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    TENSE = "tense"
    SUPPORTIVE = "supportive"
    CONFRONTATIONAL = "confrontational"


@dataclass
class CharacterPersonality:
    """Personality profile for a character."""
    name: str
    role: str = ""  # e.g., "protagonist", "helper", "authority figure"
    traits: Dict[str, float] = field(default_factory=dict)  # trait -> -1.0 to 1.0
    speech_patterns: List[str] = field(default_factory=list)  # e.g., "uses formal language"
    emotional_tendencies: List[str] = field(default_factory=list)  # e.g., "expresses worry"

    def to_prompt_context(self) -> str:
        """Generate prompt context for this personality."""
        lines = [f"Character: {self.name}"]
        if self.role:
            lines.append(f"Role: {self.role}")

        if self.traits:
            trait_desc = []
            for trait, value in self.traits.items():
                if value > 0.3:
                    trait_desc.append(f"high {trait}")
                elif value < -0.3:
                    trait_desc.append(f"low {trait}")
            if trait_desc:
                lines.append(f"Personality: {', '.join(trait_desc)}")

        if self.speech_patterns:
            lines.append(f"Speech style: {', '.join(self.speech_patterns)}")

        if self.emotional_tendencies:
            lines.append(f"Emotional tendencies: {', '.join(self.emotional_tendencies)}")

        return "\n".join(lines)


@dataclass
class DialogueContext:
    """Context for dialogue generation."""
    scenario_description: str
    agents: List[str]
    personalities: Dict[str, CharacterPersonality] = field(default_factory=dict)
    current_beliefs: Dict[str, List[str]] = field(default_factory=dict)
    current_emotions: Dict[str, str] = field(default_factory=dict)
    conversation_goals: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Generate full dialogue context for prompts."""
        lines = ["=== DIALOGUE CONTEXT ==="]
        lines.append(f"\nScenario: {self.scenario_description}")
        lines.append(f"\nParticipants: {', '.join(self.agents)}")

        for agent in self.agents:
            if agent in self.personalities:
                lines.append(f"\n{self.personalities[agent].to_prompt_context()}")

            if agent in self.current_emotions:
                lines.append(f"  Current emotion: {self.current_emotions[agent]}")

        if self.conversation_goals:
            lines.append(f"\nConversation goals:")
            for goal in self.conversation_goals:
                lines.append(f"  - {goal}")

        return "\n".join(lines)


def generate_personality_prompt(agent_name: str, scenario: str) -> str:
    """Generate prompt to create character personality."""
    return f"""
Let's create a personality profile for {agent_name} based on the scenario.

SCENARIO: {scenario}

STEP 1: Determine {agent_name}'s role in the scenario.
Think: Are they a protagonist, helper, authority, obstacle, or background character?

STEP 2: Assign personality traits (Big Five model).
Rate each trait from -1.0 (low) to +1.0 (high):
- Openness: Creative/curious vs conventional
- Conscientiousness: Organized/reliable vs spontaneous
- Extraversion: Outgoing/talkative vs reserved/quiet
- Agreeableness: Cooperative/kind vs competitive/blunt
- Neuroticism: Anxious/emotional vs calm/stable

STEP 3: Define speech patterns.
How does {agent_name} typically speak?
- Formal or casual?
- Uses specific phrases or vocabulary?
- Speaks at length or briefly?

STEP 4: Identify emotional tendencies.
What emotions does {agent_name} commonly express?

OUTPUT FORMAT:
{{
    "name": "{agent_name}",
    "role": "[role description]",
    "traits": {{
        "openness": [float -1 to 1],
        "conscientiousness": [float],
        "extraversion": [float],
        "agreeableness": [float],
        "neuroticism": [float]
    }},
    "speech_patterns": ["pattern1", "pattern2"],
    "emotional_tendencies": ["emotion1", "emotion2"]
}}
"""


def generate_dialogue_context(
    scenario: str,
    agents: List[str],
    beliefs: Dict[str, List[str]] = None,
    emotions: Dict[str, str] = None,
) -> DialogueContext:
    """Create a DialogueContext from scenario data."""
    return DialogueContext(
        scenario_description=scenario,
        agents=agents,
        current_beliefs=beliefs or {},
        current_emotions=emotions or {},
    )


# =============================================================================
# IMPROVED DIALOGUE PROMPTS
# =============================================================================

DIALOGUE_TREE_IMPROVED = """
Let's create a rich dialogue tree for this scenario with multiple conversation paths.

{context}

GOAL: Generate at least 12-15 dialogue lines with branching paths.

STEP 1: PLAN THE CONVERSATION STRUCTURE
Identify 3-4 major conversation topics/phases:
- Opening/greeting phase
- Main discussion phase(s) - what are characters trying to communicate?
- Resolution/closing phase

STEP 2: CREATE BRANCHING PATHS
For key decision points, create alternative responses:
- Positive path (agreement, cooperation)
- Negative path (disagreement, conflict)
- Neutral path (more questions, clarification)

STEP 3: MATCH DIALOGUE TO CHARACTER PERSONALITIES
Each character should speak in their own voice:
- Use vocabulary matching their personality
- Express emotions appropriate to their state
- Maintain consistent speech patterns

DIALOGUE FORMAT:
[[CurrentState, NextState, Meaning, Style, "UtteranceText"]]

STATE NAMING RULES:
- Start with "Start" state
- Use descriptive state names (e.g., "AskingHelp", "ShowingConcern", "Explaining")
- End paths with "End" state
- Create branches with states like "AgreeResponse", "DisagreeResponse"

STYLE OPTIONS: Formal, Casual, Emotional, Supportive, Tense, Friendly, Professional, Confrontational

REQUIRED STRUCTURE:
1. At least 2 opening variations
2. At least 3-4 main topic exchanges per character
3. At least 2 response alternatives for key moments
4. At least 2 closing variations

SELF-CHECK:
- Do I have at least 12 dialogue lines?
- Are there branching paths (not just linear)?
- Does each character have roughly equal speaking time?
- Do the utterances sound natural for each character's personality?
- Is there emotional progression in the conversation?

Generate the dialogue tree now:
"""

SPEAK_ACTIONS_IMPROVED = """
Let's determine which dialogue lines agent [[AGENT NAME]] will speak.

CONTEXT:
[[CONTEXT]]

ALL DIALOGUE LINES:
[[DIALOGUE_LINES]]

STEP 1: Identify [[AGENT NAME]]'s role in the conversation.
Are they initiating, responding, or both?

STEP 2: For each dialogue line, determine the speaker.
Consider:
- Who would naturally say this given the content?
- Does the utterance match [[AGENT NAME]]'s personality and role?
- Is there a clear "you" or "I" that identifies the speaker?

STEP 3: Assign lines to [[AGENT NAME]] only if they are the speaker.

FORMAT:
List each dialogue turn that [[AGENT NAME]] speaks:
[[CurrentState, NextState, Meaning, Style, "UtteranceText"]]

RULES:
- Each line should only be assigned to ONE agent
- Consider turn-taking (conversation alternates between speakers)
- Match emotional content to speaker's current state

OUTPUT only the lines that [[AGENT NAME]] speaks.
"""

SPEAK_CONDITIONS_EFFECTS_IMPROVED = """
Let's define when [[AGENT NAME]] would say "[[SPEAK ACTION]]" and what results from it.

UNDERSTANDING THE UTTERANCE:
"[[SPEAK ACTION]]"

STEP 1: ANALYZE THE UTTERANCE
- What is [[AGENT NAME]] trying to communicate?
- What mental state would lead to saying this?
- How might the listener react?

STEP 2: CONDITIONS (What must be true for [[AGENT NAME]] to say this?)
Think about:
- What must [[AGENT NAME]] believe about the situation?
- What goal/desire is this utterance serving?
- What emotional state would prompt this?
- What previous dialogue state are we in?

STEP 3: EFFECTS (What changes after saying this?)
Think about:
- What new beliefs might the listener form?
- How does [[AGENT NAME]]'s goal progress?
- Does [[AGENT NAME]]'s emotional state change?
- Are any desires satisfied or new ones created?

FORMAT:
Conditions:
[[BEL(Agent, property) = Value]]
[[DES(Agent, goal) = Value]]

Effects:
[[BEL(Agent, property) = NewValue]]
[[DES(Agent, goal) = NewValue]]

RULES:
1. EVERY statement MUST have "= Value" (True, False, or specific value)
2. Only reference agents that exist in the scenario
3. Effects should reflect the communication's impact
4. Consider BOTH speaker and listener effects

SELF-CHECK:
- Does every line end with "= Value"?
- Do conditions make sense for this utterance?
- Do effects capture both information transfer and emotional impact?

Write "Conditions:" then list, then "Effects:" then list.
"""

DIALOGUE_VARIATIONS = """
Let's create alternative phrasings for this dialogue line to add variety.

ORIGINAL LINE: [[ORIGINAL]]
SPEAKER: [[SPEAKER]]
CONTEXT: [[CONTEXT]]
EMOTIONAL STATE: [[EMOTION]]

Generate 3 alternative ways [[SPEAKER]] could express the same meaning:

VARIATION 1 (More formal):
"[utterance]"

VARIATION 2 (More emotional):
"[utterance]"

VARIATION 3 (More casual):
"[utterance]"

Each variation should:
- Convey the same core meaning
- Match [[SPEAKER]]'s personality
- Fit the emotional context
- Sound natural in conversation
"""

MULTI_PATH_DIALOGUE = """
Let's create a branching dialogue exchange for this conversation point.

CURRENT STATE: [[STATE]]
SPEAKER: [[SPEAKER]]
TOPIC: [[TOPIC]]
OTHER PARTICIPANT: [[OTHER]]

Generate responses for each path:

PATH 1 - POSITIVE (Agreement/Acceptance):
[[SPEAKER]] says: "[utterance expressing agreement]"
[[OTHER]] responds: "[positive response]"
Next state: [new state name]

PATH 2 - NEGATIVE (Disagreement/Resistance):
[[SPEAKER]] says: "[utterance expressing disagreement]"
[[OTHER]] responds: "[handling disagreement]"
Next state: [new state name]

PATH 3 - QUESTIONING (Needs more information):
[[SPEAKER]] says: "[utterance asking for clarification]"
[[OTHER]] responds: "[providing information]"
Next state: [new state name]

FORMAT each line as:
[[CurrentState, NextState, Meaning, Style, "UtteranceText"]]
"""

CHARACTER_DIALOGUE_STYLE = """
Let's establish the dialogue style for [[AGENT NAME]].

SCENARIO: [[SCENARIO]]
PERSONALITY: [[PERSONALITY]]
CURRENT EMOTION: [[EMOTION]]

Based on this character profile, define their dialogue style:

VOCABULARY LEVEL:
- Simple/everyday words OR sophisticated vocabulary?
- Technical jargon (if applicable)?
- Colloquialisms or formal language?

SENTENCE STRUCTURE:
- Short, punchy sentences OR long, elaborate ones?
- Questions frequently OR mostly statements?
- Complete sentences OR fragments?

EMOTIONAL EXPRESSION:
- Direct emotional statements ("I'm worried") OR implied emotions?
- Uses emphatic words (very, really, so)?
- Exclamations and interjections?

SOCIAL MARKERS:
- Polite forms (please, thank you)?
- Terms of address (names, titles, pet names)?
- Hedging ("I think", "maybe")?

UNIQUE SPEECH PATTERNS:
- Catchphrases or repeated expressions?
- Specific topics they always mention?
- Ways they start or end statements?

OUTPUT:
Provide 3 example utterances that demonstrate [[AGENT NAME]]'s speech style:
1. A greeting or opening line
2. An emotional statement
3. A request or suggestion
"""

# =============================================================================
# PROMPT REGISTRY
# =============================================================================

DIALOGUE_PROMPTS = {
    "dialogue_tree_improved": {
        "template": DIALOGUE_TREE_IMPROVED,
        "description": "Generate rich branching dialogue tree",
    },
    "speak_actions_improved": {
        "template": SPEAK_ACTIONS_IMPROVED,
        "description": "Assign dialogue lines to specific agent",
    },
    "speak_conditions_effects_improved": {
        "template": SPEAK_CONDITIONS_EFFECTS_IMPROVED,
        "description": "Define conditions and effects for speak actions",
    },
    "dialogue_variations": {
        "template": DIALOGUE_VARIATIONS,
        "description": "Generate alternative phrasings for dialogue",
    },
    "multi_path_dialogue": {
        "template": MULTI_PATH_DIALOGUE,
        "description": "Create branching dialogue paths",
    },
    "character_dialogue_style": {
        "template": CHARACTER_DIALOGUE_STYLE,
        "description": "Define character-specific dialogue style",
    },
    "personality_prompt": {
        "template": generate_personality_prompt("{agent_name}", "{scenario}"),
        "description": "Generate character personality profile",
    },
}
