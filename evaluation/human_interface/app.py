"""
Human Evaluation Interface for SIA-LLM scenarios.

A Streamlit application for collecting human ratings of generated scenarios
using the same rubrics as the LLM-as-Judge system.

Run with: streamlit run evaluation/human_interface/app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rubrics import (
    EVALUATION_RUBRICS,
    EvaluationDimension,
    calculate_weighted_average,
)
from evaluation.human_interface.data_manager import (
    EvaluationDatabase,
    HumanEvaluation,
    export_to_csv,
    export_to_json,
)


# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = PROJECT_ROOT / "Data"
DB_PATH = PROJECT_ROOT / "evaluation" / "human_interface" / "evaluations.db"
EXPORT_DIR = PROJECT_ROOT / "evaluation" / "human_interface" / "exports"


# ============================================================================
# Helper Functions
# ============================================================================

def load_scenarios() -> dict:
    """Load all scenarios from Data directory."""
    scenarios = {}
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("test_*.json"):
            try:
                with open(f) as file:
                    scenario = json.load(file)
                    name = scenario.get("scenario_name", f.stem)
                    scenarios[name] = scenario
            except Exception as e:
                st.warning(f"Could not load {f.name}: {e}")
    return scenarios


def format_scenario_display(scenario: dict) -> None:
    """Display scenario details in a formatted way."""
    # Scenario description
    st.subheader("Scenario Description")
    st.write(scenario.get("scenario_description", "No description available"))

    # Agents
    agents = scenario.get("agents", {})
    if agents:
        st.subheader(f"Agents ({len(agents)})")
        for agent_name, agent_data in agents.items():
            with st.expander(f"Agent: {agent_name}"):
                # Knowledge base
                kb = agent_data.get("knowledge_base", [])
                if kb:
                    st.write("**Beliefs/Desires:**")
                    for item in kb[:10]:
                        st.write(f"- {item}")
                    if len(kb) > 10:
                        st.write(f"*...and {len(kb) - 10} more*")

                # Intentions
                intentions = agent_data.get("intentions", {})
                if intentions:
                    st.write(f"**Intentions:** {len(intentions)}")

                # Actions
                actions = agent_data.get("actions", {})
                if actions:
                    st.write(f"**Actions:** {len(actions)}")

    # Dialogue
    dialogue = scenario.get("dialogue_tree", [])
    if dialogue:
        st.subheader(f"Dialogue ({len(dialogue)} lines)")
        with st.expander("View Dialogue"):
            for i, line in enumerate(dialogue):
                st.write(f"{i+1}. {line}")


def get_score_label(score: int) -> str:
    """Get label for a score value."""
    labels = {
        1: "1 - Poor",
        2: "2 - Below Average",
        3: "3 - Average",
        4: "4 - Good",
        5: "5 - Excellent",
    }
    return labels.get(score, str(score))


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.set_page_config(
        page_title="SIA-LLM Human Evaluation",
        page_icon="📊",
        layout="wide",
    )

    st.title("SIA-LLM Human Evaluation Interface")
    st.markdown("""
    Evaluate generated social scenarios using the same rubrics as the LLM-as-Judge system.
    Your ratings help validate automated evaluation and improve scenario generation.
    """)

    # Initialize database
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = EvaluationDatabase(str(DB_PATH))

    # Sidebar
    with st.sidebar:
        st.header("Evaluator Settings")

        # Evaluator ID
        evaluator_id = st.text_input(
            "Your Evaluator ID",
            value=st.session_state.get("evaluator_id", ""),
            help="Enter a unique identifier (e.g., your initials)",
        )
        if evaluator_id:
            st.session_state["evaluator_id"] = evaluator_id

        st.divider()

        # Statistics
        st.header("Progress")
        stats = db.get_statistics()
        st.metric("Total Evaluations", stats["total_evaluations"])
        st.metric("Unique Evaluators", stats["unique_evaluators"])
        st.metric("Scenarios Rated", stats["unique_scenarios"])

        if evaluator_id:
            evaluated = db.get_evaluated_scenarios(evaluator_id)
            st.metric("Your Evaluations", len(evaluated))

        st.divider()

        # Export
        st.header("Export Data")
        if st.button("Export to CSV"):
            evaluations = db.get_all_evaluations()
            if evaluations:
                path = export_to_csv(evaluations, str(EXPORT_DIR / "evaluations.csv"))
                st.success(f"Exported to {path}")
            else:
                st.warning("No evaluations to export")

        if st.button("Export to JSON"):
            evaluations = db.get_all_evaluations()
            if evaluations:
                path = export_to_json(evaluations, str(EXPORT_DIR / "evaluations.json"))
                st.success(f"Exported to {path}")
            else:
                st.warning("No evaluations to export")

    # Main content
    if not evaluator_id:
        st.warning("Please enter your Evaluator ID in the sidebar to begin.")
        return

    # Load scenarios
    scenarios = load_scenarios()
    if not scenarios:
        st.error(f"No scenarios found in {DATA_DIR}")
        return

    # Scenario selection
    st.header("Select Scenario")

    # Show which scenarios are already evaluated
    evaluated_scenarios = set(db.get_evaluated_scenarios(evaluator_id))
    scenario_names = sorted(scenarios.keys())

    # Sort: unevaluated first, then evaluated
    unevaluated = [n for n in scenario_names if n not in evaluated_scenarios]
    evaluated = [n for n in scenario_names if n in evaluated_scenarios]

    # Build options with clear markers
    scenario_options = []
    for name in unevaluated:
        scenario_options.append(name)
    for name in evaluated:
        scenario_options.append(f"✅ {name}")

    # Show counts
    st.caption(f"{len(unevaluated)} remaining, {len(evaluated)} completed")

    selected_option = st.selectbox(
        "Choose a scenario to evaluate",
        scenario_options,
        index=0,
    )

    # Extract actual scenario name
    if selected_option.startswith("✅ "):
        selected_name = selected_option[2:]
        st.info("You have already rated this scenario. Your previous scores are loaded below - you can update them.")
    else:
        selected_name = selected_option

    scenario = scenarios.get(selected_name)
    if not scenario:
        st.error("Could not load selected scenario")
        return

    # Display scenario
    st.divider()
    format_scenario_display(scenario)

    # Evaluation form
    st.divider()
    st.header("Your Evaluation")

    # Load existing evaluation if any
    existing_eval = db.get_evaluation(evaluator_id, selected_name)

    # Create columns for better layout
    col1, col2 = st.columns(2)

    scores = {}
    dimensions = list(EvaluationDimension)

    # Split dimensions between columns
    mid = len(dimensions) // 2

    with col1:
        for dim in dimensions[:mid]:
            rubric = EVALUATION_RUBRICS[dim]

            st.subheader(rubric.name)
            st.caption(rubric.question)

            # Show rubric levels in expander
            with st.expander("View scoring guide"):
                for level in sorted(rubric.levels, key=lambda x: x.score):
                    st.write(f"**{level.score} - {level.label}:** {level.description}")

            # Get existing score or default
            default_score = 3
            if existing_eval and dim.value in existing_eval.scores:
                default_score = existing_eval.scores[dim.value]

            # Radio buttons for score
            score = st.radio(
                f"Score for {rubric.name}",
                options=[1, 2, 3, 4, 5],
                index=default_score - 1,
                format_func=get_score_label,
                horizontal=True,
                key=f"score_{dim.value}",
                label_visibility="collapsed",
            )
            scores[dim.value] = score
            st.divider()

    with col2:
        for dim in dimensions[mid:]:
            rubric = EVALUATION_RUBRICS[dim]

            st.subheader(rubric.name)
            st.caption(rubric.question)

            # Show rubric levels in expander
            with st.expander("View scoring guide"):
                for level in sorted(rubric.levels, key=lambda x: x.score):
                    st.write(f"**{level.score} - {level.label}:** {level.description}")

            # Get existing score or default
            default_score = 3
            if existing_eval and dim.value in existing_eval.scores:
                default_score = existing_eval.scores[dim.value]

            # Radio buttons for score
            score = st.radio(
                f"Score for {rubric.name}",
                options=[1, 2, 3, 4, 5],
                index=default_score - 1,
                format_func=get_score_label,
                horizontal=True,
                key=f"score_{dim.value}",
                label_visibility="collapsed",
            )
            scores[dim.value] = score
            st.divider()

    # Comments
    st.subheader("Additional Comments (Optional)")
    comments = st.text_area(
        "Any additional observations about this scenario",
        value=existing_eval.comments if existing_eval else "",
        height=100,
    )

    # Show weighted average preview
    dim_scores = {EvaluationDimension(k): v for k, v in scores.items()}
    weighted_avg = calculate_weighted_average(dim_scores)
    st.metric("Weighted Average Score", f"{weighted_avg:.2f} / 5.00")

    # Submit button
    st.divider()

    # Show different button text for new vs update
    is_update = existing_eval is not None
    button_text = "Update Evaluation" if is_update else "Submit Evaluation"

    if st.button(button_text, type="primary", use_container_width=True):
        evaluation = HumanEvaluation(
            evaluator_id=evaluator_id,
            scenario_name=selected_name,
            scores=scores,
            comments=comments,
        )

        db.save_evaluation(evaluation)

        if is_update:
            st.success("Evaluation updated!")
        else:
            st.success("Evaluation saved!")
            st.balloons()

        # Rerun to refresh sidebar stats and scenario list
        st.rerun()


if __name__ == "__main__":
    main()
