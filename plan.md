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

### TASK-004: RocStories Dataset Preparation ✅ DONE
**Description:** Prepare the RocStories dataset subset used in the original paper. If the exact subset is not available, create a comparable subset with similar characteristics (causal/temporal commonsense, everyday events). Ensure reproducibility.

**Deliverables:**
- `data_utils/rocstories_scenarios.json` - Curated 40 scenarios (matched with original paper) ✅
- `data_utils/rocstories_metadata.json` - Metadata and selection criteria ✅
- `data_utils/scenario_loader.py` - RocStoriesLoader class with sampling, iteration ✅
- Dataset found in `Dataset/` folder (98,161 total stories)

**Key Findings:**
- RocStories dataset was already in the repository (`Dataset/ROCStories__spring2016.csv`, `Dataset/ROCStories_winter2017.csv`)
- All 40 scenarios from original experiments matched with RocStories titles
- Created reusable loader with `sample()`, `get_by_title()`, `load_curated_scenarios()` methods

**Estimated Hours:** 4-6 hours
**Difficulty:** ⭐ Low
**Impact:** ⭐⭐⭐⭐ High (data foundation)
**Dependencies:** TASK-001
**Blocks:** TASK-002, all experiments

---

## Phase 2: Core Improvements

### TASK-005: GPT-4 Model Integration [FEATURE: use_gpt4] ✅ DONE
**Description:** Integrate GPT-4-turbo (or GPT-4o) as an alternative model option. This requires handling the larger context window (128K tokens), adjusting prompts if necessary, and managing the different API parameters. Implement behind feature flag.

**Deliverables:**
- `models/model_factory.py` - Model abstraction layer with cost tracking ✅
- `models/__init__.py` - Package exports ✅
- `ModelHandler` class - drop-in replacement for `OpenAIHandler` ✅
- `ModelFactory.from_feature_flags()` - creates model from FeatureFlags ✅
- Cost tracking per API call with `UsageStats` ✅

**Supported Models:**
| Model | Context | Input Cost | Output Cost |
|-------|---------|------------|-------------|
| gpt-3.5-turbo | 16K | $0.0005/1K | $0.0015/1K |
| gpt-4-turbo | 128K | $0.01/1K | $0.03/1K |
| gpt-4o | 128K | $0.0025/1K | $0.01/1K |
| gpt-4o-mini | 128K | $0.00015/1K | $0.0006/1K |

**Usage:**
```python
from models import get_model
from config.feature_flags import FeatureFlags

# Direct usage
model = get_model(use_gpt4=True)  # Returns GPT-4o

# From feature flags
flags = FeatureFlags(use_gpt4=True)
model = ModelFactory.from_feature_flags(flags)
```

**Feature Flag:** `use_gpt4`
**Dependencies:** None (standalone feature)

**Estimated Hours:** 6-8 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (major improvement expected)
**Dependencies:** TASK-000, TASK-001
**Blocks:** TASK-010

---

### TASK-006: Full Context State Management [FEATURE: full_context] ✅ DONE
**Description:** Implement comprehensive state management that maintains ALL generated elements across the entire pipeline. Instead of passing truncated context, maintain a structured JSON state object that grows with each step and is fully included in subsequent prompts.

**Deliverables:**
- `core/scenario_state.py` - Complete state management class ✅
- `core/__init__.py` - Package exports ✅
- State serialization methods:
  - `to_prompt_context()` - Full readable format (~17K chars for complex scenario)
  - `to_compact_context()` - Token-efficient format (~2.5K chars)
  - `to_dict()` / `to_json()` - JSON serialization
- Load from existing scenarios: `ScenarioState.from_file()` ✅
- Validation: `validate()` returns list of `ValidationError` ✅

**Key Classes:**
- `ScenarioState` - Main state container
- `Agent` - Agent with knowledge, intentions, actions
- `Action` / `SpeakAction` - Actions with conditions/effects
- `Intention` - Intention with action plan
- `ValidationError` - Structured error reporting

**Usage:**
```python
from core import ScenarioState

# Create new state
state = ScenarioState("test", "A story...")
state.add_agent("Alice")
state.add_belief("Alice", "BEL(Alice, happy) = True")

# Load existing
state = ScenarioState.from_file("Data/test_Brother.json")

# Get context for prompts
context = state.to_prompt_context()  # Full context
compact = state.to_compact_context()  # Token-efficient
```

**Feature Flag:** `full_context`
**Dependencies:** None (standalone feature)
**Required By:** `verification_loop`, `symbolic_verification`

**Estimated Hours:** 12-16 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐⭐ Critical (addresses core limitation)
**Dependencies:** TASK-000, TASK-001
**Blocks:** TASK-007, TASK-008

---

### TASK-007: Verification Loop for Conditions/Effects [FEATURE: verification_loop] ✅ DONE
**Description:** Implement an iterative verification system that checks generated conditions and effects against the current knowledge base. If inconsistencies are found (e.g., condition references non-existent belief), provide error feedback to the LLM and request regeneration.

**Deliverables:**
- `core/verification.py` - Complete verification logic ✅
  - `BeliefDesireParser` - Parses BEL/DES/INTENT statements
  - `ConditionEffectVerifier` - Verifies conditions/effects against state
  - `verify_scenario()` - Verifies entire scenario
  - `verify_conditions_effects()` - Convenience function
- `core/error_feedback.py` - Error formatting for LLM ✅
  - `format_errors_for_llm()` - Formats errors as feedback
  - `format_regeneration_prompt()` - Complete regeneration prompt
  - `create_regeneration_request()` - Creates request from result
- `tests/test_task007_verification.py` - 22 tests ✅

**Verification Checks Implemented:**
- ✅ Missing value detection (e.g., `BEL(x, y)` without `= True`)
- ✅ Invalid format detection
- ✅ Unknown agent references
- ✅ Unknown belief/desire warnings
- ✅ Inconsistent naming detection (typo detection)
- ✅ Value validation (True/False, numbers, strings)

**Usage:**
```python
from core import verify_scenario, verify_conditions_effects

# Verify entire scenario
result = verify_scenario(state)
if not result.valid:
    feedback = format_errors_for_llm(result.errors)

# Verify specific conditions/effects
result = verify_conditions_effects(conditions, effects, state, agent, action)
```

**Test Results on Real Scenario (test_Brother.json):**
- 11 errors found (missing values, format issues)
- 20 warnings (unknown beliefs, naming inconsistencies)

**Feature Flag:** `verification_loop`
**Dependencies:** `full_context` (requires TASK-006)

**Estimated Hours:** 15-20 hours
**Difficulty:** ⭐⭐⭐⭐ Medium-High
**Impact:** ⭐⭐⭐⭐⭐ Critical (directly addresses 3% completion rate)
**Dependencies:** TASK-000, TASK-006
**Blocks:** None (can be evaluated independently)

---

### TASK-008: Symbolic Consistency Verification [FEATURE: symbolic_verification] ✅ DONE
**Description:** Implement programmatic verification that parses the generated FAtiMA-compatible output and checks logical consistency without relying on LLM judgment. This includes checking if action plans can theoretically complete given initial state.

**Deliverables:**
- `core/reachability.py` - Complete reachability analysis system ✅
  - `KnowledgeState` - Simulates agent knowledge state
  - `ReachabilityAnalyzer` - Simulates action execution
  - `analyze_scenario_reachability()` - Convenience function
  - `print_analysis_report()` - Formatted output
- `tests/test_task008_reachability.py` - 21 tests ✅

**Key Classes:**
- `KnowledgeState` - Manages beliefs/desires for simulation
- `ActionNode` - Node in action dependency graph
- `IntentionAnalysis` - Analysis result for single intention
- `AgentAnalysis` - Analysis result for single agent
- `ScenarioAnalysis` - Complete scenario analysis
- `ReachabilityAnalyzer` - Main analysis engine

**Implementation Details:**
- ✅ Parse BEL(), DES(), INTENT() statements (reuses TASK-007 parser)
- ✅ Simulate action execution step-by-step
- ✅ Track effects propagation through action chains
- ✅ Identify blocking conditions for each action
- ✅ Calculate intention completion rates
- ✅ Generate execution traces for debugging

**Usage:**
```python
from core.reachability import analyze_scenario_reachability, print_analysis_report
from core.scenario_state import ScenarioState

state = ScenarioState.from_file("Data/test_Brother.json")
analysis = analyze_scenario_reachability(state)
print_analysis_report(analysis)
```

**Baseline Analysis Results (test_Brother.json):**
- Intention Completion Rate: **0%** (0/7 completable)
- Action Executability Rate: **36.8%** (14/38 immediately executable)
- Root causes: missing values, circular dependencies, unreachable conditions

**Feature Flag:** `symbolic_verification`
**Dependencies:** `full_context` (requires TASK-006)

**Estimated Hours:** 12-15 hours
**Difficulty:** ⭐⭐⭐⭐ Medium-High
**Impact:** ⭐⭐⭐⭐ High (enables deeper analysis)
**Dependencies:** TASK-000, TASK-006
**Blocks:** TASK-013

---

### TASK-009: Enhanced Chain-of-Thought Prompting [FEATURE: cot_enhancement] ✅ DONE
**Description:** Improve the Chain-of-Thought prompting strategy by adding explicit reasoning steps, self-consistency checks within prompts, and structured output formats. Based on advances in CoT since the original paper (2023).

**Deliverables:**
- `prompts/` - Prompt management package ✅
  - `prompts/prompt_manager.py` - PromptManager class with style switching
  - `prompts/enhanced/cot_prompts.py` - 13 enhanced CoT prompts
- `tests/test_task009_cot_prompts.py` - 21 tests ✅

**Key Improvements in Enhanced Prompts:**
1. **Step-by-step reasoning**: "STEP 1:", "STEP 2:", etc.
2. **Self-consistency checks**: "SELF-CHECK before output:" sections
3. **Format guidance**: Explicit FORMAT sections with examples
4. **Strict rules**: ABSOLUTE RULES for critical prompts (conditions_effects)
5. **Error prevention**: Examples of CORRECT and WRONG output

**Enhanced Prompts Created:**
| Prompt | Enhancement Focus |
|--------|------------------|
| agents | Step-by-step extraction, naming rules |
| beliefs_desires | Format validation, = Value requirement |
| intentions | Derivation from beliefs/desires |
| action_plan | Dependency checking, ordering |
| conditions_effects | **Strict validation**, explicit value requirement |
| dialogue_tree | Branching guidance, state machine design |
| speak_actions | Character assignment |
| speak_conditions_effects | Dialogue effect tracking |
| initial_emotion | OCC model guidance |
| initial_mood | Scale explanation |
| action_emotion | Post-action appraisal |
| emotion_condition | Pre-action requirement |
| action_mood | Mood threshold guidance |

**Usage:**
```python
from prompts import get_prompt_manager

# Original prompts
pm = get_prompt_manager(use_enhanced=False)
prompt = pm.get("beliefs_desires", agent_name="Alice")

# Enhanced CoT prompts
pm_enh = get_prompt_manager(use_enhanced=True)
prompt = pm_enh.get("conditions_effects", action="Help(A,B)", agent_name="A")
```

**Feature Flag:** `cot_enhancement`
**Dependencies:** None (standalone feature)

**Estimated Hours:** 10-12 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐ High (improves reasoning quality)
**Dependencies:** TASK-000, TASK-001
**Blocks:** None

---

### TASK-010: Dialogue Generation Improvement [FEATURE: dialogue_improvement] ✅ DONE
**Description:** Address the limited dialogue generation (5.30 lines average) by implementing improved prompting strategies, character-specific dialogue styles, and better state machine coverage.

**Deliverables:**
- `prompts/dialogue/` - Improved dialogue prompts package ✅
  - `dialogue_prompts.py` - 7 improved dialogue prompts with personality system
  - `dialogue_analyzer.py` - State machine analysis utilities
- `tests/test_task010_dialogue.py` - 34 tests ✅

**Key Components Created:**

**Character Personality System:**
- `CharacterPersonality` - Big Five traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- `PersonalityTrait` enum - Standardized trait names
- `DialogueStyle` enum - 8 dialogue styles (Formal, Casual, Emotional, Supportive, Tense, Friendly, Professional, Confrontational)
- `DialogueContext` - Full context for dialogue generation

**Dialogue Analysis System:**
- `DialogueLine` - Parse dialogue lines from both `<>` and `[[]]` formats
- `DialogueState` - State in dialogue state machine
- `DialogueGraph` - Full dialogue state machine graph
- `DialogueMetrics` - Metrics collection (lines, states, branches, paths, styles)
- `analyze_dialogue()` - Analyze dialogue tree and compute metrics
- `compare_dialogue_metrics()` - Compare baseline vs improved

**Improved Prompts:**
| Prompt | Purpose |
|--------|---------|
| dialogue_tree_improved | Generate rich branching dialogue (target: 12-15 lines) |
| speak_actions_improved | Assign lines to specific agents |
| speak_conditions_effects_improved | Define conditions/effects for speak actions |
| dialogue_variations | Generate alternative phrasings |
| multi_path_dialogue | Create positive/negative/questioning paths |
| character_dialogue_style | Define character speech patterns |
| personality_prompt | Generate Big Five personality profile |

**Baseline Analysis:**
- Existing scenarios: 6.62 dialogue lines average
- Target: 15-20 lines with branching paths
- New prompts emphasize: branching (2-3 paths), character personality, emotional progression

**Usage:**
```python
from prompts.dialogue import (
    CharacterPersonality, DialogueContext,
    analyze_dialogue, compare_dialogue_metrics,
    DIALOGUE_PROMPTS
)

# Create personality
personality = CharacterPersonality(
    name="Alice",
    traits={"openness": 0.6, "agreeableness": 0.8},
)

# Analyze dialogue
metrics = analyze_dialogue(dialogue_tree)
print(f"Lines: {metrics.total_lines}, Paths: {metrics.approximate_paths}")
```

**Feature Flag:** `dialogue_improvement`
**Dependencies:** None (standalone, but benefits from `use_gpt4`)

**Estimated Hours:** 10-14 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐ Medium-High (addresses specific weakness)
**Dependencies:** TASK-000, TASK-001
**Blocks:** None

---

## Phase 3: Evaluation Infrastructure

### TASK-011: LLM-as-Judge Evaluation System [FEATURE: llm_judge] ✅ DONE
**Description:** Implement an automated evaluation system using a separate LLM instance to judge generated scenarios. Design rubrics matching the original paper's subjective evaluation criteria (relevance, branching, logical errors).

**Deliverables:**
- `evaluation/` - Evaluation package ✅
  - `rubrics.py` - 10 evaluation rubrics with 1-5 Likert scales
  - `llm_judge.py` - LLM judge implementation
- `tests/test_task011_llm_judge.py` - 25 tests ✅

**Evaluation Dimensions (10 total):**
| Dimension | Weight | Focus Area |
|-----------|--------|------------|
| Agent Relevance | 1.0 | Are agents appropriate for scenario? |
| Belief Coherence | 1.2 | Are beliefs logically consistent? |
| Desire Appropriateness | 1.2 | Do desires match agent roles? |
| Intention Validity | 1.0 | Do intentions follow from BDI? |
| Action Feasibility | 1.5 | Can action plans execute? |
| Condition/Effect Logic | 1.5 | Are conditions/effects well-formed? |
| Dialogue Quality | 1.0 | Quantity, branching, structure |
| Dialogue Naturalness | 0.8 | Natural, character-appropriate speech |
| Emotional Consistency | 0.8 | OCC model compliance |
| Overall Coherence | 1.5 | Complete scenario quality |

**Key Classes:**
- `EvaluationDimension` - Enum of all 10 dimensions
- `EvaluationRubric` - Rubric with levels, question, weights
- `LLMJudge` - Main evaluation class
- `EvaluationResult` - Scores with reasoning per dimension
- `DimensionScore` - Score, reasoning, confidence per dimension

**Features:**
- Structured JSON output parsing
- Fallback to freeform response parsing
- Score clamping to 1-5 range
- Weighted average calculation
- Batch evaluation support
- Comparison between baseline and improved

**Usage:**
```python
from evaluation import LLMJudge, evaluate_scenario

# Evaluate single scenario
result = evaluate_scenario(scenario, model_name="gpt-4o")
print(f"Weighted Average: {result.weighted_average:.2f}/5")

# Compare baseline vs improved
comparison = compare_evaluations(baseline_results, improved_results)
print(f"Improvement: {comparison['overall']['improvement']}")
```

**Feature Flag:** `llm_judge`
**Dependencies:** None (standalone evaluation feature)

**Estimated Hours:** 12-16 hours
**Difficulty:** ⭐⭐⭐ Medium
**Impact:** ⭐⭐⭐⭐ High (enables scalable evaluation)
**Dependencies:** TASK-000, TASK-001, TASK-002
**Blocks:** TASK-014

---

### TASK-012: Human Evaluation Interface ✅ DONE
**Description:** Build a simple web interface for human evaluation of generated scenarios. This enables collecting new human ratings to validate LLM-as-judge and evaluate improved system.

**Deliverables:**
- `evaluation/human_interface/` - Complete evaluation package ✅
  - `app.py` - Streamlit web application
  - `data_manager.py` - SQLite storage and CSV/JSON export
  - `README.md` - Usage instructions
- `tests/test_task012_human_eval.py` - 16 tests ✅

**Features:**
- **Scenario Display**: Shows description, agents, beliefs, actions, dialogue
- **10 Evaluation Dimensions**: Same rubrics as LLM-as-Judge (TASK-011)
- **1-5 Likert Scale**: Radio buttons with expandable scoring guides
- **Progress Tracking**: Shows evaluated vs pending scenarios per evaluator
- **SQLite Storage**: Persistent database with update support
- **Export Options**: CSV and JSON export via sidebar buttons
- **Statistics Dashboard**: Total evaluations, unique evaluators, average scores

**Key Classes:**
- `HumanEvaluation` - Single evaluation with scores and weighted average
- `EvaluationDatabase` - SQLite wrapper with CRUD operations

**Usage:**
```bash
# Run the web interface
streamlit run evaluation/human_interface/app.py

# Opens browser at http://localhost:8501
```

**Programmatic Usage:**
```python
from evaluation.human_interface import (
    EvaluationDatabase, HumanEvaluation,
    export_to_csv, export_to_json
)

# Load evaluations
db = EvaluationDatabase("evaluations.db")
all_evals = db.get_all_evaluations()

# Export for analysis
export_to_csv(all_evals, "results.csv")
```

**Estimated Hours:** 10-12 hours
**Difficulty:** ⭐⭐ Low-Medium
**Impact:** ⭐⭐⭐ Medium (validation, not core)
**Dependencies:** TASK-001, TASK-011 (uses same rubrics)
**Blocks:** None (optional but valuable)

---

### TASK-013: Automated Metrics Dashboard ✅ DONE
**Description:** Create a dashboard that automatically calculates and visualizes all relevant metrics across experiment runs. Should support comparison between ablation study conditions.

**Deliverables:**
- `analysis/` - Metrics analysis package ✅
  - `metrics.py` - Metrics calculation classes and functions
  - `comparison.py` - Cross-condition comparison utilities
  - `dashboard.py` - Streamlit dashboard application
- `tests/test_task013_metrics.py` - 43 tests ✅

**Key Classes:**

**ScenarioMetrics:**
- Artifact counts (agents, beliefs, desires, intentions, actions, conditions, effects, emotions, dialogue, speak actions)
- Completion rates (intention completion, executable actions)
- Dialogue metrics (lines, branch points, unique paths, styles)
- Quality flags (is_complete, has_dialogue, has_emotions)

**ExperimentMetrics:**
- Aggregated means across all scenarios
- Total counts and completion rates
- Per-scenario breakdowns

**ConditionComparison:**
- Absolute differences between conditions
- Percentage improvements
- Detailed metric-by-metric comparison

**Functions:**
- `calculate_scenario_metrics(scenario)` - Metrics for single scenario
- `calculate_experiment_metrics(scenarios)` - Aggregated experiment metrics
- `compare_conditions(baseline, treatment)` - Compare two conditions
- `compare_multiple_conditions()` - Compare multiple treatments vs baseline
- `generate_comparison_table()` - Markdown comparison table
- `load_scenarios_from_directory()` - Load scenarios from JSON files

**Usage:**
```python
from analysis import (
    calculate_scenario_metrics, calculate_experiment_metrics,
    compare_conditions, generate_comparison_table,
)

# Calculate metrics
metrics = calculate_experiment_metrics(scenarios, "experiment_name")
print(f"Intention completion: {metrics.intention_completion_rate:.1%}")

# Compare conditions
comparison = compare_conditions(baseline, treatment)
print(f"Improvement: {comparison.intention_completion_improvement:+.1f}%")

# Run dashboard
# streamlit run analysis/dashboard.py
```

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