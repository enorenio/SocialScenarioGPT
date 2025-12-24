# Improved dialogue generation prompts for SIA-LLM
from prompts.dialogue.dialogue_prompts import (
    DIALOGUE_PROMPTS,
    CharacterPersonality,
    DialogueContext,
    PersonalityTrait,
    DialogueStyle,
    generate_personality_prompt,
    generate_dialogue_context,
)
from prompts.dialogue.dialogue_analyzer import (
    DialogueLine,
    DialogueState,
    DialogueGraph,
    DialogueMetrics,
    analyze_dialogue,
    analyze_scenario_dialogue,
    print_dialogue_report,
    compare_dialogue_metrics,
)
