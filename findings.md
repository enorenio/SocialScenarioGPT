# Research Findings

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
