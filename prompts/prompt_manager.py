"""
Prompt management system for SIA-LLM experiments.
Supports original and enhanced Chain-of-Thought prompts with feature flag control.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class PromptStyle(Enum):
    """Style of prompts to use."""
    ORIGINAL = "original"  # Original paper prompts
    ENHANCED = "enhanced"  # Enhanced CoT prompts


@dataclass
class PromptTemplate:
    """A prompt template with metadata."""
    name: str
    template: str
    description: str = ""
    placeholders: List[str] = field(default_factory=list)
    style: PromptStyle = PromptStyle.ORIGINAL

    def fill(self, **kwargs) -> str:
        """Fill placeholders in the template."""
        result = self.template
        for key, value in kwargs.items():
            # Try both underscore and space versions (original prompts use spaces)
            placeholder_underscore = f"[[{key.upper()}]]"
            placeholder_space = f"[[{key.upper().replace('_', ' ')}]]"
            result = result.replace(placeholder_underscore, str(value))
            result = result.replace(placeholder_space, str(value))
        return result

    def get_placeholders(self) -> List[str]:
        """Extract all placeholders from template."""
        pattern = r'\[\[([^\]]+)\]\]'
        matches = re.findall(pattern, self.template)
        return list(set(matches))


class PromptManager:
    """
    Manages prompts for scenario generation.
    Supports switching between original and enhanced prompts via feature flags.
    """

    def __init__(self, use_enhanced: bool = False):
        """
        Initialize prompt manager.

        Args:
            use_enhanced: If True, use enhanced CoT prompts
        """
        self.use_enhanced = use_enhanced
        self._original_prompts: Dict[str, PromptTemplate] = {}
        self._enhanced_prompts: Dict[str, PromptTemplate] = {}

        # Load prompts
        self._load_original_prompts()
        self._load_enhanced_prompts()

    def _load_original_prompts(self):
        """Load original prompts from Constants/task_constants.py."""
        from Constants.task_constants import (
            TASK_DESCRIPTION,
            SCENARIO_DESCRIPTION_GENERATIVE_TASK,
            AGENT_TRANSLATION_TASK,
            BELIEFS_DESIRES_TRANSLATION_TASK,
            INTENTS_TRANSLATION_TASK,
            ACTION_PLAN_TRANSLATION_TASK,
            ACTION_TRANSLATION_TASK,
            CONDITIONS_EFFECTS_TASK,
            DIALOGUE_TREE_TASK,
            SPEAK_ACTION_TASK,
            SPEAK_CONDITIONS_EFFECTS,
            INITIAL_EMO_TASK,
            INITIAL_MOOD_TASK,
            ACTIONS_EMO_APPRAISAL,
            EMOTION_CONDITION_TASK,
            ACTION_MOOD,
        )

        self._original_prompts = {
            "task_description": PromptTemplate(
                name="task_description",
                template=TASK_DESCRIPTION,
                description="Main BDI architecture task description",
                style=PromptStyle.ORIGINAL,
            ),
            "scenario_description": PromptTemplate(
                name="scenario_description",
                template=SCENARIO_DESCRIPTION_GENERATIVE_TASK,
                description="Scenario description template",
                style=PromptStyle.ORIGINAL,
            ),
            "agents": PromptTemplate(
                name="agents",
                template=AGENT_TRANSLATION_TASK,
                description="Extract agents from scenario",
                style=PromptStyle.ORIGINAL,
            ),
            "beliefs_desires": PromptTemplate(
                name="beliefs_desires",
                template=BELIEFS_DESIRES_TRANSLATION_TASK,
                description="Generate beliefs and desires for agent",
                style=PromptStyle.ORIGINAL,
            ),
            "intentions": PromptTemplate(
                name="intentions",
                template=INTENTS_TRANSLATION_TASK,
                description="Generate intentions from beliefs/desires",
                style=PromptStyle.ORIGINAL,
            ),
            "action_plan": PromptTemplate(
                name="action_plan",
                template=ACTION_PLAN_TRANSLATION_TASK,
                description="Generate action sequence for intention",
                style=PromptStyle.ORIGINAL,
            ),
            "actions": PromptTemplate(
                name="actions",
                template=ACTION_TRANSLATION_TASK,
                description="Translate intentions to actions",
                style=PromptStyle.ORIGINAL,
            ),
            "conditions_effects": PromptTemplate(
                name="conditions_effects",
                template=CONDITIONS_EFFECTS_TASK,
                description="Generate conditions and effects for action",
                style=PromptStyle.ORIGINAL,
            ),
            "dialogue_tree": PromptTemplate(
                name="dialogue_tree",
                template=DIALOGUE_TREE_TASK,
                description="Generate dialogue state machine",
                style=PromptStyle.ORIGINAL,
            ),
            "speak_actions": PromptTemplate(
                name="speak_actions",
                template=SPEAK_ACTION_TASK,
                description="Get dialogue turns for agent",
                style=PromptStyle.ORIGINAL,
            ),
            "speak_conditions_effects": PromptTemplate(
                name="speak_conditions_effects",
                template=SPEAK_CONDITIONS_EFFECTS,
                description="Conditions/effects for speak actions",
                style=PromptStyle.ORIGINAL,
            ),
            "initial_emotion": PromptTemplate(
                name="initial_emotion",
                template=INITIAL_EMO_TASK,
                description="Initial OCC emotion for agent",
                style=PromptStyle.ORIGINAL,
            ),
            "initial_mood": PromptTemplate(
                name="initial_mood",
                template=INITIAL_MOOD_TASK,
                description="Initial mood value for agent",
                style=PromptStyle.ORIGINAL,
            ),
            "action_emotion": PromptTemplate(
                name="action_emotion",
                template=ACTIONS_EMO_APPRAISAL,
                description="Emotion after performing action",
                style=PromptStyle.ORIGINAL,
            ),
            "emotion_condition": PromptTemplate(
                name="emotion_condition",
                template=EMOTION_CONDITION_TASK,
                description="Emotion required before action",
                style=PromptStyle.ORIGINAL,
            ),
            "action_mood": PromptTemplate(
                name="action_mood",
                template=ACTION_MOOD,
                description="Mood required for action",
                style=PromptStyle.ORIGINAL,
            ),
        }

    def _load_enhanced_prompts(self):
        """Load enhanced CoT prompts."""
        from prompts.enhanced.cot_prompts import ENHANCED_PROMPTS

        for name, prompt_data in ENHANCED_PROMPTS.items():
            self._enhanced_prompts[name] = PromptTemplate(
                name=name,
                template=prompt_data["template"],
                description=prompt_data.get("description", ""),
                style=PromptStyle.ENHANCED,
            )

    def get(self, name: str, **kwargs) -> str:
        """
        Get a prompt by name, filled with provided values.

        Args:
            name: Prompt name
            **kwargs: Values for placeholders

        Returns:
            Filled prompt string
        """
        template = self.get_template(name)
        return template.fill(**kwargs)

    def get_template(self, name: str) -> PromptTemplate:
        """
        Get prompt template by name.
        Uses enhanced version if enabled and available.
        """
        if self.use_enhanced and name in self._enhanced_prompts:
            return self._enhanced_prompts[name]

        if name in self._original_prompts:
            return self._original_prompts[name]

        raise KeyError(f"Unknown prompt: {name}")

    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        all_names = set(self._original_prompts.keys())
        all_names.update(self._enhanced_prompts.keys())
        return sorted(all_names)

    def has_enhanced(self, name: str) -> bool:
        """Check if enhanced version exists for a prompt."""
        return name in self._enhanced_prompts

    def get_style(self, name: str) -> PromptStyle:
        """Get the style that would be used for a prompt."""
        if self.use_enhanced and name in self._enhanced_prompts:
            return PromptStyle.ENHANCED
        return PromptStyle.ORIGINAL


# Singleton instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager(use_enhanced: bool = False) -> PromptManager:
    """
    Get or create the prompt manager singleton.

    Args:
        use_enhanced: If True, use enhanced CoT prompts

    Returns:
        PromptManager instance
    """
    global _prompt_manager

    if _prompt_manager is None or _prompt_manager.use_enhanced != use_enhanced:
        _prompt_manager = PromptManager(use_enhanced=use_enhanced)

    return _prompt_manager
