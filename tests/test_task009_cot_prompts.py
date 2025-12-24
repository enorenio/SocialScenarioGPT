"""
Tests for TASK-009: Enhanced Chain-of-Thought Prompting
Tests the prompt management system and enhanced prompts.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts import PromptManager, PromptTemplate, get_prompt_manager
from prompts.prompt_manager import PromptStyle
from prompts.enhanced.cot_prompts import ENHANCED_PROMPTS


# ============================================================
# PromptTemplate Tests
# ============================================================

def test_prompt_template_init():
    """Test PromptTemplate initialization."""
    template = PromptTemplate(
        name="test",
        template="Hello [[NAME]]!",
        description="Test template",
        style=PromptStyle.ORIGINAL,
    )

    assert template.name == "test"
    assert template.template == "Hello [[NAME]]!"
    assert template.description == "Test template"
    assert template.style == PromptStyle.ORIGINAL

    print("✓ PromptTemplate initializes correctly")


def test_prompt_template_fill():
    """Test filling placeholders in template."""
    template = PromptTemplate(
        name="test",
        template="Hello [[NAME]], welcome to [[PLACE]]!",
    )

    filled = template.fill(name="Alice", place="Wonderland")

    assert filled == "Hello Alice, welcome to Wonderland!"

    print("✓ PromptTemplate.fill() works")


def test_prompt_template_fill_uppercase():
    """Test that placeholder filling uses uppercase keys."""
    template = PromptTemplate(
        name="test",
        template="Agent [[AGENT_NAME]] does [[ACTION]]",
    )

    # Keys are converted to uppercase for matching
    filled = template.fill(agent_name="Bob", action="run")

    assert "Bob" in filled
    assert "run" in filled
    assert "[[AGENT_NAME]]" not in filled

    print("✓ PromptTemplate handles uppercase placeholders correctly")


def test_prompt_template_get_placeholders():
    """Test extracting placeholders from template."""
    template = PromptTemplate(
        name="test",
        template="[[AGENT NAME]] has [[BELIEF]] and [[DESIRE]]",
    )

    placeholders = template.get_placeholders()

    assert "AGENT NAME" in placeholders
    assert "BELIEF" in placeholders
    assert "DESIRE" in placeholders
    assert len(placeholders) == 3

    print("✓ PromptTemplate.get_placeholders() works")


# ============================================================
# PromptManager Tests
# ============================================================

def test_prompt_manager_init_original():
    """Test PromptManager with original prompts."""
    pm = PromptManager(use_enhanced=False)

    assert pm.use_enhanced is False
    assert len(pm._original_prompts) > 0

    print("✓ PromptManager initializes with original prompts")


def test_prompt_manager_init_enhanced():
    """Test PromptManager with enhanced prompts."""
    pm = PromptManager(use_enhanced=True)

    assert pm.use_enhanced is True
    assert len(pm._enhanced_prompts) > 0

    print("✓ PromptManager initializes with enhanced prompts")


def test_prompt_manager_list_prompts():
    """Test listing available prompts."""
    pm = PromptManager(use_enhanced=False)
    prompts = pm.list_prompts()

    assert len(prompts) > 0
    assert "beliefs_desires" in prompts
    assert "conditions_effects" in prompts
    assert "agents" in prompts

    print(f"✓ PromptManager.list_prompts() returns {len(prompts)} prompts")


def test_prompt_manager_get_original():
    """Test getting original prompts."""
    pm = PromptManager(use_enhanced=False)

    template = pm.get_template("beliefs_desires")

    assert template is not None
    assert template.style == PromptStyle.ORIGINAL
    assert "[[AGENT NAME]]" in template.template

    print("✓ PromptManager returns original prompts when not enhanced")


def test_prompt_manager_get_enhanced():
    """Test getting enhanced prompts when enabled."""
    pm = PromptManager(use_enhanced=True)

    template = pm.get_template("beliefs_desires")

    assert template is not None
    assert template.style == PromptStyle.ENHANCED
    assert "step by step" in template.template.lower()

    print("✓ PromptManager returns enhanced prompts when enabled")


def test_prompt_manager_get_with_fill():
    """Test getting and filling a prompt."""
    pm = PromptManager(use_enhanced=False)

    filled = pm.get("beliefs_desires", agent_name="Alice")

    assert "Alice" in filled
    assert "[[AGENT NAME]]" not in filled

    print("✓ PromptManager.get() fills placeholders correctly")


def test_prompt_manager_has_enhanced():
    """Test checking for enhanced versions."""
    pm = PromptManager(use_enhanced=False)

    assert pm.has_enhanced("beliefs_desires") is True
    assert pm.has_enhanced("conditions_effects") is True
    assert pm.has_enhanced("nonexistent") is False

    print("✓ PromptManager.has_enhanced() works")


def test_prompt_manager_get_style():
    """Test getting the style that would be used."""
    pm_orig = PromptManager(use_enhanced=False)
    pm_enh = PromptManager(use_enhanced=True)

    assert pm_orig.get_style("beliefs_desires") == PromptStyle.ORIGINAL
    assert pm_enh.get_style("beliefs_desires") == PromptStyle.ENHANCED

    print("✓ PromptManager.get_style() returns correct style")


def test_get_prompt_manager_singleton():
    """Test singleton behavior of get_prompt_manager."""
    pm1 = get_prompt_manager(use_enhanced=False)
    pm2 = get_prompt_manager(use_enhanced=False)

    # Should be same instance
    assert pm1 is pm2

    # Different setting should create new instance
    pm3 = get_prompt_manager(use_enhanced=True)
    assert pm3 is not pm1
    assert pm3.use_enhanced is True

    print("✓ get_prompt_manager() singleton works correctly")


def test_prompt_manager_fallback_to_original():
    """Test fallback to original when enhanced not available."""
    pm = PromptManager(use_enhanced=True)

    # task_description might not have enhanced version
    # Should fall back to original gracefully
    template = pm.get_template("task_description")
    assert template is not None

    print("✓ PromptManager falls back to original when enhanced unavailable")


# ============================================================
# Enhanced Prompts Content Tests
# ============================================================

def test_enhanced_prompts_registry():
    """Test that enhanced prompts registry has expected entries."""
    assert "beliefs_desires" in ENHANCED_PROMPTS
    assert "conditions_effects" in ENHANCED_PROMPTS
    assert "intentions" in ENHANCED_PROMPTS
    assert "action_plan" in ENHANCED_PROMPTS

    print(f"✓ Enhanced prompts registry has {len(ENHANCED_PROMPTS)} prompts")


def test_enhanced_prompts_have_cot_elements():
    """Test that enhanced prompts contain CoT elements."""
    for name, data in ENHANCED_PROMPTS.items():
        template = data["template"].lower()

        # Should have step-by-step reasoning
        has_steps = "step" in template
        # Should have self-check
        has_check = "check" in template or "self-check" in template

        assert has_steps or has_check, f"{name} lacks CoT elements"

    print("✓ All enhanced prompts contain CoT elements")


def test_enhanced_prompts_have_format_section():
    """Test that most enhanced prompts include format guidance."""
    prompts_with_format = 0
    for name, data in ENHANCED_PROMPTS.items():
        template = data["template"]

        # Check for format guidance
        has_format = "FORMAT" in template or "format:" in template.lower()
        has_example = "Example" in template or "EXAMPLE" in template
        has_write = "Write" in template  # Instructions on what to write

        if has_format or has_example or has_write:
            prompts_with_format += 1

    # Most prompts should have format guidance (allow some flexibility)
    assert prompts_with_format >= len(ENHANCED_PROMPTS) * 0.7, \
        f"Only {prompts_with_format}/{len(ENHANCED_PROMPTS)} have format guidance"

    print(f"✓ {prompts_with_format}/{len(ENHANCED_PROMPTS)} enhanced prompts have format/example guidance")


def test_conditions_effects_enhanced_has_strict_rules():
    """Test that conditions_effects prompt has strict validation rules."""
    template = ENHANCED_PROMPTS["conditions_effects"]["template"]

    # Should emphasize the = Value requirement
    assert "= Value" in template or "= True" in template
    assert "ABSOLUTE RULES" in template or "RULES" in template
    assert "SELF-CHECK" in template

    print("✓ conditions_effects enhanced prompt has strict validation rules")


def test_enhanced_prompts_preserve_placeholders():
    """Test that enhanced prompts use same placeholders as original."""
    pm = PromptManager(use_enhanced=False)

    for name in ENHANCED_PROMPTS:
        if name in pm._original_prompts:
            orig = pm._original_prompts[name]
            enh_template = ENHANCED_PROMPTS[name]["template"]

            # Check key placeholders are preserved
            orig_placeholders = orig.get_placeholders()
            for ph in orig_placeholders:
                if ph in ["AGENT NAME", "ACTION", "INTENTION"]:
                    assert f"[[{ph}]]" in enh_template, \
                        f"Enhanced {name} missing placeholder [[{ph}]]"

    print("✓ Enhanced prompts preserve original placeholders")


# ============================================================
# Integration Tests
# ============================================================

def test_enhanced_longer_than_original():
    """Test that enhanced prompts are generally longer (more guidance)."""
    pm_orig = PromptManager(use_enhanced=False)
    pm_enh = PromptManager(use_enhanced=True)

    longer_count = 0
    for name in ENHANCED_PROMPTS:
        if name in pm_orig._original_prompts:
            orig_len = len(pm_orig._original_prompts[name].template)
            enh_len = len(ENHANCED_PROMPTS[name]["template"])

            if enh_len > orig_len:
                longer_count += 1

    # Most enhanced should be longer
    assert longer_count > len(ENHANCED_PROMPTS) // 2

    print(f"✓ {longer_count}/{len(ENHANCED_PROMPTS)} enhanced prompts are longer than original")


def test_prompt_workflow():
    """Test typical workflow of getting prompts for scenario generation."""
    pm = get_prompt_manager(use_enhanced=True)

    # Simulate workflow
    # 1. Get agents prompt
    agents_prompt = pm.get("agents")
    assert len(agents_prompt) > 0

    # 2. Get beliefs/desires for each agent
    bd_prompt = pm.get("beliefs_desires", agent_name="TestAgent")
    assert "TestAgent" in bd_prompt

    # 3. Get intentions
    int_prompt = pm.get("intentions", agent_name="TestAgent")
    assert "TestAgent" in int_prompt

    # 4. Get conditions/effects for action
    ce_prompt = pm.get("conditions_effects", action="TestAction", agent_name="TestAgent")
    assert "TestAction" in ce_prompt
    assert "TestAgent" in ce_prompt

    print("✓ Complete prompt workflow works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-009: Enhanced Chain-of-Thought Prompting Tests")
    print("=" * 60)

    # PromptTemplate tests
    test_prompt_template_init()
    test_prompt_template_fill()
    test_prompt_template_fill_uppercase()
    test_prompt_template_get_placeholders()

    # PromptManager tests
    test_prompt_manager_init_original()
    test_prompt_manager_init_enhanced()
    test_prompt_manager_list_prompts()
    test_prompt_manager_get_original()
    test_prompt_manager_get_enhanced()
    test_prompt_manager_get_with_fill()
    test_prompt_manager_has_enhanced()
    test_prompt_manager_get_style()
    test_get_prompt_manager_singleton()
    test_prompt_manager_fallback_to_original()

    # Enhanced prompts content tests
    test_enhanced_prompts_registry()
    test_enhanced_prompts_have_cot_elements()
    test_enhanced_prompts_have_format_section()
    test_conditions_effects_enhanced_has_strict_rules()
    test_enhanced_prompts_preserve_placeholders()

    # Integration tests
    test_enhanced_longer_than_original()
    test_prompt_workflow()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
