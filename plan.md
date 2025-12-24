```markdown
# SIA-LLM Enhancement Project: Comprehensive Task List

## Project Overview
Enhancement of the Antunes et al. (2023) "Prompting for Socially Intelligent Agents with ChatGPT" system to improve intention completion rates, scenario coherence, and evaluation scalability.

**Baseline Metrics (Original Paper):**
- Intention completion rate: 3% (11/369)
- Immediately executable actions: 25.9% (955/2756)
- Dialogue lines per scenario: 5.30 average
- Generation time: 32.82 minutes per scenario
- Human artifact generation: 1.5 artifacts/minute
- LLM artifact generation: 15.28 artifacts/minute

**Target Metrics:**
- Intention completion rate: 40-60%
- Dialogue lines per scenario: 15-20
- Maintain or improve relevance scores

---

## Architecture Design

### TASK-000: Feature Flag System Design ✅ DONE
**Description:** Simple configuration system for enabling/disabling features. Keep it minimal - a single Python file with a dataclass or dict.

**Deliverables:**
- `config/feature_flags.py` - Feature flags with dependency validation ✅

**Estimated Hours:** 2-3 hours
**Difficulty:** ⭐⭐ Low
**Impact:** ⭐⭐⭐⭐⭐ Critical (blocks all other tasks)
**Dependencies:** None
**Blocks:** All other implementation tasks

---

## Phase 1: Infrastructure & Baseline

### TASK-001: Repository Setup ✅ DONE
**Description:** Clone the original repository and set up development environment.

**Deliverables:**
- Cloned repository ✅
- `requirements.txt` with dependencies ✅
- `venv/` virtual environment created ✅
- `.env` file with `OPENAI_API_KEY` (user creates locally, not committed)

**Estimated Hours:** 1 hour
**Difficulty:** ⭐ Low
**Impact:** ⭐⭐⭐⭐⭐ Critical (foundational)
**Dependencies:** None
**Blocks:** All subsequent tasks

---

### TASK-002: Baseline Reproduction ✅ DONE
**Description:** Reproduce the original paper's results using the same RocStories scenarios and GPT-3.5-turbo. This establishes our own baseline for comparison and validates our understanding of the system. Must use the feature flag system with all enhancements disabled.

**Deliverables:**
- `experiments/baseline_runner.py` - Script to run baseline experiments ✅
- 42 scenarios from original repo (33 completed, in `Data/`) ✅
- Metrics matching original paper ✅:
  - Intention completion: **3.0% (11/369)** - EXACT MATCH
  - Immediately executable: **34.7% (955/2756)** - paper says 25.9% (arithmetic error in paper)
  - Avg dialogue lines: **5.43** (paper: 5.30) - within 2.5%
- Model used: `gpt-3.5-turbo` (same as paper)
- **Actual measured generation time: 2.52 min** (paper claimed 32.82 min - likely due to API improvements since 2023)
- `experiments/results/baseline_existing_data.json` - Baseline results ✅

**Estimated Hours:** 10-15 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (validation)
**Dependencies:** TASK-000, TASK-001
**Blocks:** All comparison experiments

---

### TASK-003: Comprehensive Logging and Metrics Collection System ✅ DONE
**Description:** Implement a structured logging system that captures all intermediate outputs, API calls, token usage, timing information, and quality metrics. Essential for debugging, ablation studies, and result analysis. Must log which features were active for each run.

**Deliverables:**
- `utils/logger.py` - Structured logging module ✅
  - `ExperimentLogger` class with step context managers
  - Logs API calls, tokens, timing, artifacts per step
  - Saves JSON logs to `experiments/logs/`
  - Tracks feature flags per run
- `utils/metrics.py` - Metrics calculation utilities ✅
  - `calculate_scenario_metrics()` - per-scenario metrics
  - `calculate_aggregate_metrics()` - cross-scenario metrics
  - `print_metrics_report()` - formatted output
- Log file format: JSON with scenario, steps, api_calls, final_state ✅

**Estimated Hours:** 8-10 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐⭐ High (enables analysis)
**Dependencies:** TASK-000, TASK-001
**Blocks:** TASK-013, TASK-014

---

### TASK-004: RocStories Dataset Preparation
**Description:** Prepare the RocStories dataset subset used in the original paper. If the exact subset is not available, create a comparable subset with similar characteristics (causal/temporal commonsense, everyday events). Ensure reproducibility.

**Deliverables:**
- `data/rocstories_scenarios.json` - Curated scenario descriptions
- `data/rocstories_metadata.json` - Metadata about each scenario
- Script to load and iterate scenarios
- Documentation of selection criteria

**Estimated Hours:** 4-6 hours
**Difficulty:** ⭐ Low
**Impact:** ⭐⭐⭐⭐ High (data foundation)
**Dependencies:** TASK-001
**Blocks:** TASK-002, all experiments

---

## Phase 2: Core Improvements

### TASK-005: GPT-4 Model Integration [FEATURE: use_gpt4]
**Description:** Integrate GPT-4-turbo (or GPT-4o) as an alternative model option. This requires handling the larger context window (128K tokens), adjusting prompts if necessary, and managing the different API parameters. Implement behind feature flag.

**Implementation Details:**
- Abstract model selection in configuration
- Handle different token limits
- Adjust temperature/top_p if needed
- Track cost differences (GPT-4 is more expensive)
- Implement graceful fallback

**Deliverables:**
- `models/model_factory.py` - Model abstraction layer
- Updated configuration for model selection
- Cost tracking per scenario
- Comparison documentation

**Feature Flag:** `use_gpt4`
**Dependencies:** None (standalone feature)

**Estimated Hours:** 6-8 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (major improvement expected)
**Dependencies:** TASK-000, TASK-001
**Blocks:** TASK-010

---

### TASK-006: Full Context State Management [FEATURE: full_context]
**Description:** Implement comprehensive state management that maintains ALL generated elements across the entire pipeline. Instead of passing truncated context, maintain a structured JSON state object that grows with each step and is fully included in subsequent prompts.

**Implementation Details:**
```python
class ScenarioState:
    scenario_description: str
    agents: List[Agent]
    knowledge_base: Dict[str, List[BeliefOrDesire]]
    intentions: Dict[str, List[Intention]]
    action_plans: Dict[str, List[ActionPlan]]
    conditions_effects: Dict[str, ConditionsEffects]
    emotions: Dict[str, EmotionLabels]
    dialogue_state_machine: DialogueStateMachine
    
    def to_prompt_context(self) -> str:
        """Serialize full state for inclusion in prompts"""
        
    def validate_consistency(self) -> List[Error]:
        """Check internal consistency"""
```

**Deliverables:**
- `core/scenario_state.py` - State management class
- `core/state_serializer.py` - Serialization for prompts
- Updated pipeline to use state object
- State validation utilities

**Feature Flag:** `full_context`
**Dependencies:** None (standalone feature)
**Required By:** `verification_loop`, `symbolic_verification`

**Estimated Hours:** 12-16 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (addresses core limitation)
**Dependencies:** TASK-000, TASK-001
**Blocks:** TASK-007, TASK-008

---

### TASK-007: Verification Loop for Conditions/Effects [FEATURE: verification_loop]
**Description:** Implement an iterative verification system that checks generated conditions and effects against the current knowledge base. If inconsistencies are found (e.g., condition references non-existent belief), provide error feedback to the LLM and request regeneration.

**Implementation Details:**
```python
def generate_with_verification(
    action: Action, 
    state: ScenarioState,
    max_retries: int = 3
) -> Tuple[Conditions, Effects]:
    for attempt in range(max_retries):
        conditions, effects = generate_conditions_effects(action, state)
        errors = verify_against_state(conditions, effects, state)
        
        if not errors:
            return conditions, effects
        
        # Construct error feedback prompt
        feedback = format_errors_for_llm(errors)
        conditions, effects = regenerate_with_feedback(
            action, state, feedback
        )
    
    # Log failure and return best attempt or flag for review
    return handle_verification_failure(action, attempts)
```

**Verification Checks:**
- Condition references existing belief in knowledge base
- Effect updates use consistent naming
- No circular dependencies
- Numeric values are valid
- Boolean values are consistent

**Deliverables:**
- `core/verification.py` - Verification logic
- `core/error_feedback.py` - Error formatting for LLM
- Verification rules specification
- Metrics on retry rates and success

**Feature Flag:** `verification_loop`
**Dependencies:** `full_context` (requires TASK-006)

**Estimated Hours:** 15-20 hours
**Difficulty:** ⭐⭐⭐⭐ Medium-High
**Impact:** ⭐⭐⭐⭐⭐ Critical (directly addresses 3% completion rate)
**Dependencies:** TASK-000, TASK-006
**Blocks:** None (can be evaluated independently)

---

### TASK-008: Symbolic Consistency Verification [FEATURE: symbolic_verification]
**Description:** Implement programmatic verification that parses the generated FAtiMA-compatible output and checks logical consistency without relying on LLM judgment. This includes checking if action plans can theoretically complete given initial state.

**Implementation Details:**
- Parse BEL(), DES(), INTENT() statements
- Build dependency graph of actions
- Simulate action execution to check reachability
- Identify dead-end states
- Calculate theoretical maximum completion rate

**Deliverables:**
- `core/symbolic_parser.py` - Parse FAtiMA format
- `core/dependency_graph.py` - Build action dependencies
- `core/reachability_analyzer.py` - Check plan feasibility
- Report generator for consistency analysis

**Feature Flag:** `symbolic_verification`
**Dependencies:** `full_context` (requires TASK-006)

**Estimated Hours:** 12-15 hours
**Difficulty:** ⭐⭐⭐⭐ Medium-High
**Impact:** ⭐⭐⭐⭐ High (enables deeper analysis)
**Dependencies:** TASK-000, TASK-006
**Blocks:** TASK-013

---

### TASK-009: Enhanced Chain-of-Thought Prompting [FEATURE: cot_enhancement]
**Description:** Improve the Chain-of-Thought prompting strategy by adding explicit reasoning steps, self-consistency checks within prompts, and structured output formats. Based on advances in CoT since the original paper (2023).

**Implementation Details:**
- Add "Let's think step by step" variants
- Include self-consistency checks in prompts
- Request explicit reasoning before output
- Use structured JSON output format
- Add examples (few-shot) where beneficial

**Prompt Improvements:**
```
Original: "Generate beliefs for agent {agent_name}..."

Enhanced: "Let's generate beliefs for {agent_name} step by step:
1. First, identify what {agent_name} knows based on the scenario
2. Consider what {agent_name} might infer from the situation
3. Think about {agent_name}'s relationships and knowledge of others
4. Now, list each belief in the format BEL(...)=Value

Before finalizing, verify:
- Each belief is grounded in the scenario
- No contradictory beliefs exist
- Beliefs use consistent naming

Output as JSON: {\"beliefs\": [...], \"reasoning\": \"...\"}"
```

**Deliverables:**
- `prompts/enhanced/` - Directory with improved prompts
- A/B comparison data with original prompts
- Documentation of prompt engineering decisions

**Feature Flag:** `cot_enhancement`
**Dependencies:** None (standalone feature)

**Estimated Hours:** 10-12 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐ High (improves reasoning quality)
**Dependencies:** TASK-000, TASK-001
**Blocks:** None

---

### TASK-010: Dialogue Generation Improvement [FEATURE: dialogue_improvement]
**Description:** Address the limited dialogue generation (5.30 lines average) by implementing improved prompting strategies, character-specific dialogue styles, and better state machine coverage.

**Implementation Details:**
- Generate dialogue per character with personality
- Ensure state machine coverage (multiple paths)
- Include emotional context in dialogue prompts
- Generate variations for same semantic content
- Add contextual appropriateness checks

**Improvements:**
- Current: Generic dialogue usable by any agent
- Target: Character-specific dialogue with personality markers
- Current: Linear state machine (few branches)
- Target: Branching dialogue trees with multiple paths

**Deliverables:**
- `prompts/dialogue/` - Improved dialogue prompts
- Character personality template
- Dialogue state machine visualizer
- Metrics on dialogue quantity and diversity

**Feature Flag:** `dialogue_improvement`
**Dependencies:** None (standalone, but benefits from `use_gpt4`)

**Estimated Hours:** 10-14 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐ Medium-High (addresses specific weakness)
**Dependencies:** TASK-000, TASK-001
**Blocks:** None

---

## Phase 3: Evaluation Infrastructure

### TASK-011: LLM-as-Judge Evaluation System [FEATURE: llm_judge]
**Description:** Implement an automated evaluation system using a separate LLM instance to judge generated scenarios. Design rubrics matching the original paper's subjective evaluation criteria (relevance, branching, logical errors).

**Implementation Details:**
```python
EVALUATION_RUBRIC = {
    "agent_relevance": {
        "description": "Are the generated agents appropriate for the scenario?",
        "scale": "1-5 Likert",
        "criteria": [...]
    },
    "belief_coherence": {
        "description": "Are beliefs logically consistent and grounded?",
        "scale": "1-5 Likert", 
        "criteria": [...]
    },
    # ... match all original paper dimensions
}

def evaluate_scenario(scenario: GeneratedScenario, rubric: dict) -> EvaluationResult:
    """Use GPT-4 to evaluate scenario against rubric"""
```

**Validation:**
- Compare LLM-judge scores with original human annotations
- Calculate correlation (target: r > 0.6)
- Identify dimensions where LLM-judge diverges

**Deliverables:**
- `evaluation/llm_judge.py` - LLM evaluation module
- `evaluation/rubrics.yaml` - Evaluation criteria
- Validation study comparing to human ratings
- Inter-rater reliability analysis (LLM vs human)

**Feature Flag:** `llm_judge`
**Dependencies:** None (standalone evaluation feature)

**Estimated Hours:** 12-16 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐ High (enables scalable evaluation)
**Dependencies:** TASK-000, TASK-001, TASK-002
**Blocks:** TASK-014

---

### TASK-012: Human Evaluation Interface
**Description:** Build a simple web interface for human evaluation of generated scenarios. This enables collecting new human ratings to validate LLM-as-judge and evaluate improved system.

**Implementation Details:**
- Display scenario description and generated content
- Present evaluation questions matching original paper
- Collect Likert scale ratings
- Track evaluator and scenario IDs
- Export results in analyzable format

**Tech Stack Suggestion:**
- Streamlit or Gradio for rapid development
- SQLite for result storage
- CSV/JSON export

**Deliverables:**
- `evaluation/human_interface/` - Web application
- Evaluation protocol documentation
- Data collection consent form (if needed)
- Results export utilities

**Estimated Hours:** 10-12 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐ Medium (validation, not core)
**Dependencies:** TASK-001
**Blocks:** None (optional but valuable)

---

### TASK-013: Automated Metrics Dashboard
**Description:** Create a dashboard that automatically calculates and visualizes all relevant metrics across experiment runs. Should support comparison between ablation study conditions.

**Metrics to Track:**
- Artifact counts (all 10 categories from Table 1)
- Intention completion rate
- Immediately executable actions percentage
- Dialogue coverage metrics
- Generation time and cost
- Verification retry rates (if enabled)
- LLM-judge scores (if enabled)

**Deliverables:**
- `analysis/dashboard.py` - Metrics dashboard (Streamlit/Plotly)
- `analysis/comparison.py` - Cross-condition comparison
- Automated report generation
- Visualization exports for paper

**Estimated Hours:** 8-10 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐⭐ High (essential for analysis)
**Dependencies:** TASK-003, TASK-008
**Blocks:** TASK-014

---

## Phase 4: Ablation Study

### TASK-014: Ablation Study Execution
**Description:** Systematically evaluate each feature and feature combination to determine individual and combined contributions to performance improvements.

**Study Design:**

| Condition ID | use_gpt4 | full_context | verification_loop | cot_enhancement | dialogue_improvement | Notes |
|--------------|----------|--------------|-------------------|-----------------|---------------------|-------|
| C00 | ❌ | ❌ | ❌ | ❌ | ❌ | Baseline (original) |
| C01 | ✅ | ❌ | ❌ | ❌ | ❌ | GPT-4 only |
| C02 | ❌ | ✅ | ❌ | ❌ | ❌ | Full context only |
| C03 | ❌ | ❌ | ❌ | ✅ | ❌ | CoT enhancement only |
| C04 | ❌ | ❌ | ❌ | ❌ | ✅ | Dialogue improvement only |
| C05 | ✅ | ✅ | ❌ | ❌ | ❌ | GPT-4 + Full context |
| C06 | ✅ | ✅ | ✅ | ❌ | ❌ | + Verification loop |
| C07 | ✅ | ✅ | ✅ | ✅ | ❌ | + CoT enhancement |
| C08 | ✅ | ✅ | ✅ | ✅ | ✅ | Full system |
| C09 | ✅ | ✅ | ❌ | ✅ | ✅ | Full minus verification |
| C10 | ✅ | ❌ | ❌ | ✅ | ✅ | Full minus full_context |

**Note:** `verification_loop` and `symbolic_verification` require `full_context` to be enabled. The ablation study must respect these dependencies.

**Scenarios per Condition:** Minimum 20, ideally 43 (matching original)

**Deliverables:**
- `experiments/ablation_runner.py` - Automated ablation execution
- `experiments/ablation_config.yaml` - All condition definitions
- Results for all conditions
- Statistical analysis (paired t-tests, effect sizes)
- Feature contribution analysis

**Estimated Hours:** 20-30 hours (including runtime)
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (answers research questions)
**Dependencies:** TASK-005 through TASK-013
**Blocks:** TASK-015

---

### TASK-015: Results Analysis and Reporting
**Description:** Comprehensive analysis of ablation study results. Determine which features contribute most to improvements, identify interaction effects, and prepare publication-quality figures and tables.

**Analysis Components:**
1. **Individual Feature Contribution:**
   - Compare each single-feature condition to baseline
   - Calculate effect size (Cohen's d)
   - Statistical significance testing

2. **Interaction Effects:**
   - Compare combined conditions to sum of individual effects
   - Identify synergistic combinations
   - Identify redundant combinations

3. **Cost-Benefit Analysis:**
   - Improvement per additional API cost
   - Improvement per additional generation time
   - Pareto frontier of cost vs. quality

4. **Failure Analysis:**
   - Categorize remaining failures in best condition
   - Identify systematic error patterns
   - Suggest future improvements

**Deliverables:**
- Comprehensive results document
- Publication-ready tables (LaTeX format)
- Publication-ready figures
- Statistical analysis notebook
- Recommendations summary

**Estimated Hours:** 15-20 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (research output)
**Dependencies:** TASK-014
**Blocks:** None (final task)

---

## Feature Dependency Graph

```
TASK-000 (Feature Flags)
    │
    ├── TASK-001 (Repo Setup)
    │       │
    │       ├── TASK-002 (Baseline)
    │       ├── TASK-003 (Logging)
    │       ├── TASK-004 (Data Prep)
    │       │
    │       ├── TASK-005 (GPT-4) [use_gpt4]
    │       │
    │       ├── TASK-006 (Full Context) [full_context]
    │       │       │
    │       │       ├── TASK-007 (Verification) [verification_loop]
    │       │       │       └── requires: full_context
    │       │       │
    │       │       └── TASK-008 (Symbolic) [symbolic_verification]
    │       │               └── requires: full_context
    │       │
    │       ├── TASK-009 (CoT) [cot_enhancement]
    │       │
    │       ├── TASK-010 (Dialogue) [dialogue_improvement]
    │       │
    │       ├── TASK-011 (LLM Judge) [llm_judge]
    │       │
    │       ├── TASK-012 (Human Eval Interface)
    │       │
    │       └── TASK-013 (Dashboard)
    │               │
    │               └── TASK-014 (Ablation Study)
    │                       │
    │                       └── TASK-015 (Analysis)
```

**Feature Dependencies (for runtime validation):**
```yaml
feature_dependencies:
  use_gpt4: []
  full_context: []
  verification_loop: [full_context]
  symbolic_verification: [full_context]
  cot_enhancement: []
  dialogue_improvement: []
  llm_judge: []
```

---

## Summary Table

| Task ID | Task Name | Hours | Difficulty | Impact | Phase |
|---------|-----------|-------|------------|--------|-------|
| TASK-000 | Feature Flag System | 2-3 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0 |
| TASK-001 | Repo Setup | 1 | ⭐ | ⭐⭐⭐⭐⭐ | 1 |
| TASK-002 | Baseline Reproduction | 10-15 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 1 |
| TASK-003 | Logging System | 8-10 | ⭐⭐ | ⭐⭐⭐⭐ | 1 |
| TASK-004 | Data Preparation | 4-6 | ⭐ | ⭐⭐⭐⭐ | 1 |
| TASK-005 | GPT-4 Integration | 6-8 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 2 |
| TASK-006 | Full Context State | 12-16 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2 |
| TASK-007 | Verification Loop | 15-20 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 2 |
| TASK-008 | Symbolic Verification | 12-15 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 2 |
| TASK-009 | CoT Enhancement | 10-12 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 2 |
| TASK-010 | Dialogue Improvement | 10-14 | ⭐⭐⭐ | ⭐⭐⭐ | 2 |
| TASK-011 | LLM-as-Judge | 12-16 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 3 |
| TASK-012 | Human Eval Interface | 10-12 | ⭐⭐ | ⭐⭐⭐ | 3 |
| TASK-013 | Metrics Dashboard | 8-10 | ⭐⭐ | ⭐⭐⭐⭐ | 3 |
| TASK-014 | Ablation Study | 20-30 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4 |
| TASK-015 | Results Analysis | 15-20 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 4 |

**Total Estimated Hours:** 168-224 hours (4-6 weeks full-time, 2-3 months part-time)

---

## Quick Start Path (Minimum Viable Improvement)

If time-constrained, prioritize:

1. **TASK-000** - Feature flags (required)
2. **TASK-001** - Repo setup
3. **TASK-002** - Baseline
4. **TASK-005** - GPT-4 integration
5. **TASK-006** - Full context state
6. **TASK-014** - Ablation (simplified: just C00, C01, C02, C05)
7. **TASK-015** - Analysis

**Minimum Path Hours:** ~60-80 hours
**Expected Outcome:** Intention completion 3% → 25-40%, publishable improvement

---

## Notes for Implementation

1. **All features must be implemented behind feature flags** from TASK-000
2. **Log active configuration** at the start of every experimental run
3. **Use consistent random seeds** for reproducibility
4. **Track API costs** - GPT-4 is significantly more expensive
5. **Save all intermediate states** for debugging and analysis
6. **Version control all prompts** - changes should be tracked
7. **Document any deviations** from original paper methodology
```