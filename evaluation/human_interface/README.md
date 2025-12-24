# Human Evaluation Interface

A Streamlit web application for collecting human ratings of SIA-LLM generated scenarios.

## Purpose

This interface allows human evaluators to rate generated scenarios using the same 10-dimension rubrics as the LLM-as-Judge system. Human ratings can be used to:

1. **Validate LLM-as-Judge** - Compare human scores with automated scores
2. **Establish Ground Truth** - Collect reliable human judgments for analysis
3. **Inter-rater Reliability** - Measure consistency between evaluators

## Installation

Ensure Streamlit is installed:

```bash
pip install streamlit
```

## Running the Application

From the project root directory:

```bash
streamlit run evaluation/human_interface/app.py
```

This will open a browser window at `http://localhost:8501`.

## How to Use

### 1. Enter Your Evaluator ID
- Use a unique identifier (e.g., your initials)
- This tracks your progress and prevents accidental overwriting

### 2. Select a Scenario
- Scenarios marked with ✓ have already been rated by you
- You can re-evaluate scenarios to update your ratings

### 3. Review the Scenario
- Read the scenario description
- Expand agent details to see beliefs, desires, intentions, and actions
- Review the dialogue tree

### 4. Rate Each Dimension
- Use the 1-5 scale for each of 10 evaluation dimensions
- Click "View scoring guide" to see detailed rubric for each level
- Each dimension has specific criteria for what constitutes a 1 vs 5

### 5. Add Comments (Optional)
- Note any specific issues or observations
- Helpful for qualitative analysis

### 6. Submit
- Your evaluation is saved to the local SQLite database
- You can continue to the next scenario or re-evaluate others

## Evaluation Dimensions

| Dimension | Focus |
|-----------|-------|
| Agent Relevance | Are agents appropriate for scenario? |
| Belief Coherence | Are beliefs logically consistent? |
| Desire Appropriateness | Do desires match agent roles? |
| Intention Validity | Do intentions follow from BDI? |
| Action Feasibility | Can action plans execute? |
| Condition/Effect Logic | Are conditions/effects well-formed? |
| Dialogue Quality | Quantity, branching, structure |
| Dialogue Naturalness | Natural, character-appropriate speech |
| Emotional Consistency | OCC model compliance |
| Overall Coherence | Complete scenario quality |

## Data Storage

Evaluations are stored in:
- **SQLite Database**: `evaluation/human_interface/evaluations.db`

## Exporting Data

Use the sidebar buttons to export:
- **CSV**: Spreadsheet-compatible format with one row per evaluation
- **JSON**: Structured format for programmatic analysis

Exports are saved to: `evaluation/human_interface/exports/`

## Comparing with LLM-as-Judge

After collecting human evaluations, compare with LLM-as-Judge:

```python
from evaluation import LLMJudge, compare_evaluations
from evaluation.human_interface import EvaluationDatabase, export_to_json

# Load human evaluations
db = EvaluationDatabase("evaluation/human_interface/evaluations.db")
human_evals = db.get_all_evaluations()

# Run LLM evaluation on same scenarios
judge = LLMJudge()
# ... evaluate same scenarios ...

# Compare results
# (Convert human evaluations to EvaluationResult format for comparison)
```

## Tips for Evaluators

1. **Be Consistent** - Use the rubric descriptions to anchor your scores
2. **Take Notes** - Comments help explain edge cases
3. **Review Before Submitting** - Check your scores make sense together
4. **Rate Multiple Scenarios** - More data = better validation
