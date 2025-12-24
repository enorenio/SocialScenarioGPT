# Research Findings

## Finding 3: Baseline Dialogues Are Almost Entirely Linear

**Date:** 2024-12-24

**Context:** While implementing TASK-010 (Dialogue Generation Improvement), we analyzed the state machine structure of all 31 scenarios with dialogue.

**Quantitative Analysis:**

| Metric | Baseline | Target |
|--------|----------|--------|
| Avg dialogue lines | 7.16 | 15-20 |
| Avg branch points | 0.26 | 2-3 |
| Avg unique paths | 1.26 | 3-5 |

**Distribution:**
- 87% of scenarios have 4-9 dialogue lines
- 0% reach the 13+ line target
- Only 8/31 (26%) scenarios have any branching at all

**Implications:**
- The dialogue limitation is structural, not just quantitative
- Generated dialogues are simple linear sequences, not true state machines
- FAtiMA's branching capability is underutilized
- Improved prompts must explicitly require branching paths

---

## Finding 2: Paper's Intention Completion Metric Uses Lenient String Matching

**Date:** 2024-12-24

**Context:** While implementing TASK-008 (Symbolic Consistency Verification), we discovered the paper's completion analysis differs significantly from semantic verification.

**Paper's Method (`AutomaticEvaluation.py`):**
```python
if all(item in knowledge for item in conditions):
    # action is executable
```
- Exact string matching: condition must exist *verbatim* in knowledge base
- Silently skips missing actions (`try/except` everywhere)
- No semantic parsing of BEL/DES statements

**Our Method (`core/reachability.py`):**
- Parses `BEL(agent, property) = value` semantically
- Tracks key→value pairs for matching
- Reports blocking conditions explicitly

**Results on 33 Completed Scenarios:**

| Method | Completed | Total | Rate |
|--------|-----------|-------|------|
| Paper's method | 9 | 266 | 3.4% |
| Our method | 4 | 266 | 1.5% |

**Why Paper's Method Finds More:**
1. Exact string matching can accidentally match unrelated beliefs
2. Missing actions are silently skipped (counted as "not blocking")
3. No validation that parsed values match

**Implications:**
- Both methods confirm completion rate is very low (<5%)
- Paper's 3% (11/369) is slightly inflated by lenient matching
- Our stricter analysis reveals even fewer logically consistent scenarios
- The core finding stands: baseline generates inconsistent action plans

---

## Finding 1: Baseline Generation Time is 13x Faster Than Reported

**Date:** 2024-12-24

**Context:** While reproducing the baseline from Antunes et al. (2023) "Prompting for Socially Intelligent Agents with ChatGPT", we discovered a significant discrepancy in generation time.

**Paper's Claim:**
- Average generation time: 32.82 minutes per scenario

**Our Measurement:**
- Actual generation time: **2.52 minutes** per scenario
- Model used: `gpt-3.5-turbo` (same as paper)
- Scenario completed successfully with all pipeline stages

**Speedup Factor:** ~13x faster

**Likely Explanations:**
1. OpenAI API infrastructure improvements since 2023
2. Faster model inference on OpenAI's side
3. Original experiments may have hit rate limits causing delays
4. Possible batching/queuing delays in original experiments

**Implications:**
- Running full ablation studies will be significantly faster than planned
- 20 scenarios @ 2.5 min = ~50 minutes (vs. ~11 hours at paper's rate)
- More iterations and experiments are feasible within budget

**Verification:**
- Test scenario: "John went to the store to buy groceries..."
- Generated 2 agents (John, Mary)
- Produced 37 actions, 7 speak actions, 4 dialogue lines
- All pipeline stages completed (`last_ended: "end"`)
