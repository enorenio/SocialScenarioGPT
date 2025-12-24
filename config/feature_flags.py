"""
Feature flag system for SIA-LLM Enhancement Project.
Enables/disables features for ablation studies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class FeatureFlags:
    """Configuration for which features are enabled."""

    use_gpt4: bool = False
    full_context: bool = False
    verification_loop: bool = False
    cot_enhancement: bool = False
    dialogue_improvement: bool = False
    llm_judge: bool = False
    symbolic_verification: bool = False

    # Feature dependencies: feature -> list of required features
    _dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        "verification_loop": ["full_context"],
        "symbolic_verification": ["full_context"],
    }, repr=False)

    def __post_init__(self):
        """Validate dependencies after initialization."""
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if feature dependencies are not satisfied."""
        errors = []
        for feature, required in self._dependencies.items():
            if getattr(self, feature):
                for req in required:
                    if not getattr(self, req):
                        errors.append(f"'{feature}' requires '{req}' to be enabled")
        if errors:
            raise ValueError("Feature dependency errors:\n  - " + "\n  - ".join(errors))

    def enabled_features(self) -> Set[str]:
        """Return set of enabled feature names."""
        return {
            name for name in [
                "use_gpt4", "full_context", "verification_loop",
                "cot_enhancement", "dialogue_improvement", "llm_judge",
                "symbolic_verification"
            ]
            if getattr(self, name)
        }

    def to_dict(self) -> Dict[str, bool]:
        """Return dict of all feature flags."""
        return {
            "use_gpt4": self.use_gpt4,
            "full_context": self.full_context,
            "verification_loop": self.verification_loop,
            "cot_enhancement": self.cot_enhancement,
            "dialogue_improvement": self.dialogue_improvement,
            "llm_judge": self.llm_judge,
            "symbolic_verification": self.symbolic_verification,
        }


# Pre-defined ablation profiles for TASK-014
# Condition IDs match plan.md ablation study design
PROFILES = {
    # C00: Baseline (original system)
    "C00": FeatureFlags(),
    "baseline": FeatureFlags(),  # Alias for backwards compatibility

    # C01: GPT-4 only
    "C01": FeatureFlags(use_gpt4=True),
    "gpt4_only": FeatureFlags(use_gpt4=True),

    # C02: Full context only
    "C02": FeatureFlags(full_context=True),
    "full_context_only": FeatureFlags(full_context=True),

    # C03: CoT enhancement only
    "C03": FeatureFlags(cot_enhancement=True),
    "cot_only": FeatureFlags(cot_enhancement=True),

    # C04: Dialogue improvement only
    "C04": FeatureFlags(dialogue_improvement=True),
    "dialogue_only": FeatureFlags(dialogue_improvement=True),

    # C05: GPT-4 + Full context
    "C05": FeatureFlags(
        use_gpt4=True,
        full_context=True,
    ),
    "gpt4_full_context": FeatureFlags(
        use_gpt4=True,
        full_context=True,
    ),

    # C06: GPT-4 + Full context + Verification loop
    "C06": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        verification_loop=True,
    ),
    "gpt4_full_context_verification": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        verification_loop=True,
    ),

    # C07: GPT-4 + Full context + Verification + CoT
    "C07": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        verification_loop=True,
        cot_enhancement=True,
    ),

    # C08: Full system (all features)
    "C08": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        verification_loop=True,
        cot_enhancement=True,
        dialogue_improvement=True,
    ),
    "full_system": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        verification_loop=True,
        cot_enhancement=True,
        dialogue_improvement=True,
    ),

    # C09: Full minus verification (GPT-4 + Full context + CoT + Dialogue)
    "C09": FeatureFlags(
        use_gpt4=True,
        full_context=True,
        cot_enhancement=True,
        dialogue_improvement=True,
    ),

    # C10: Full minus full_context (GPT-4 + CoT + Dialogue)
    "C10": FeatureFlags(
        use_gpt4=True,
        cot_enhancement=True,
        dialogue_improvement=True,
    ),
}

# Ablation study condition groups for organized running
CONDITION_GROUPS = {
    "single_feature": ["C00", "C01", "C02", "C03", "C04"],  # Baseline + single features
    "combined": ["C05", "C06", "C07", "C08"],  # Progressive combinations
    "ablation": ["C08", "C09", "C10"],  # Full system minus one feature
    "all": ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10"],
}


def get_profile(name: str) -> FeatureFlags:
    """Get a pre-defined feature profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(PROFILES.keys())}")
    return PROFILES[name]
