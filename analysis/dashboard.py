"""
Streamlit Dashboard for SIA-LLM Metrics Visualization.

Provides interactive visualization of experiment metrics
and cross-condition comparisons.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import (
    ScenarioMetrics,
    ExperimentMetrics,
    calculate_scenario_metrics,
    calculate_experiment_metrics,
    load_scenarios_from_directory,
    print_metrics_report,
)
from analysis.comparison import (
    ConditionComparison,
    compare_conditions,
    compare_multiple_conditions,
    generate_comparison_table,
    summarize_comparisons,
)


# Page configuration
st.set_page_config(
    page_title="SIA-LLM Metrics Dashboard",
    page_icon="📊",
    layout="wide",
)


def render_folder_browser(
    base_path: str = ".",
    key_prefix: str = "folder",
) -> Tuple[Optional[str], str]:
    """
    Render a simple folder browser in the sidebar.

    Args:
        base_path: Starting directory path
        key_prefix: Unique key prefix for Streamlit widgets

    Returns:
        Tuple of (selected_directory, file_pattern)
    """
    # Initialize session state for current path
    state_key = f"{key_prefix}_current_path"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(PROJECT_ROOT)

    current_path = Path(st.session_state[state_key])

    # Ensure path exists
    if not current_path.exists():
        current_path = PROJECT_ROOT
        st.session_state[state_key] = str(current_path)

    # Show current location
    st.sidebar.caption(f"📍 Current: `{current_path.name}/`")

    # Navigation: Go up button
    col1, col2 = st.sidebar.columns([1, 3])
    with col1:
        if st.button("⬆️ Up", key=f"{key_prefix}_up", help="Go to parent folder"):
            parent = current_path.parent
            if parent.exists():
                st.session_state[state_key] = str(parent)
                st.rerun()

    with col2:
        if st.button("🏠 Root", key=f"{key_prefix}_root", help="Go to project root"):
            st.session_state[state_key] = str(PROJECT_ROOT)
            st.rerun()

    # List subdirectories
    try:
        subdirs = sorted([
            d for d in current_path.iterdir()
            if d.is_dir() and not d.name.startswith('.') and d.name != '__pycache__' and d.name != 'venv'
        ])
    except PermissionError:
        subdirs = []
        st.sidebar.warning("Permission denied")

    # Show folders as clickable buttons
    if subdirs:
        st.sidebar.markdown("**📁 Folders:**")
        for subdir in subdirs[:15]:  # Limit to 15 folders
            # Count JSON files in this folder
            json_count = len(list(subdir.glob("*.json")))
            label = f"📂 {subdir.name}"
            if json_count > 0:
                label += f" ({json_count} json)"

            if st.sidebar.button(label, key=f"{key_prefix}_dir_{subdir.name}"):
                st.session_state[state_key] = str(subdir)
                st.rerun()

        if len(subdirs) > 15:
            st.sidebar.caption(f"... and {len(subdirs) - 15} more folders")
    else:
        st.sidebar.caption("No subfolders")

    # Show JSON files in current directory
    json_files = sorted(current_path.glob("*.json"))
    if json_files:
        st.sidebar.markdown(f"**📄 JSON files:** {len(json_files)}")
        with st.sidebar.expander("Preview files"):
            for f in json_files[:10]:
                st.caption(f"• {f.name}")
            if len(json_files) > 10:
                st.caption(f"... and {len(json_files) - 10} more")

    # File pattern input
    pattern = st.sidebar.text_input(
        "File Pattern",
        value="*.json",
        key=f"{key_prefix}_pattern",
        help="Glob pattern (e.g., test_*.json, *.json)",
    )

    # Experiment name
    experiment_name = st.sidebar.text_input(
        "Experiment Name",
        value=current_path.name or "Experiment",
        key=f"{key_prefix}_name",
    )

    return str(current_path), pattern, experiment_name


def load_experiment_data(directory: str, name: str) -> ExperimentMetrics:
    """Load experiment data from a directory."""
    scenarios = load_scenarios_from_directory(directory)
    return calculate_experiment_metrics(scenarios, experiment_name=name)


def render_metrics_overview(metrics: ExperimentMetrics):
    """Render the metrics overview section."""
    st.header("📊 Metrics Overview")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Scenarios",
            metrics.total_scenarios,
            f"{metrics.complete_scenarios} complete",
        )

    with col2:
        st.metric(
            "Intention Completion",
            f"{metrics.intention_completion_rate:.1%}",
            help="Rate of intentions that can be fully executed",
        )

    with col3:
        st.metric(
            "Executable Actions",
            f"{metrics.executable_actions_rate:.1%}",
            help="Rate of actions immediately executable",
        )

    with col4:
        st.metric(
            "Avg Dialogue Lines",
            f"{metrics.mean_dialogue_lines:.1f}",
            help="Mean dialogue lines per scenario",
        )


def render_artifact_chart(metrics: ExperimentMetrics):
    """Render artifact counts bar chart."""
    st.subheader("📦 Mean Artifact Counts")

    # Data for chart
    artifact_data = {
        "Agents": metrics.mean_agents,
        "Beliefs": metrics.mean_beliefs,
        "Desires": metrics.mean_desires,
        "Intentions": metrics.mean_intentions,
        "Actions": metrics.mean_actions,
        "Conditions": metrics.mean_conditions,
        "Effects": metrics.mean_effects,
        "Dialogue": metrics.mean_dialogue_lines,
        "Speak Acts": metrics.mean_speak_actions,
    }

    st.bar_chart(artifact_data)


def render_scenario_table(metrics: ExperimentMetrics):
    """Render individual scenario metrics table."""
    st.subheader("📋 Individual Scenarios")

    if not metrics.scenario_metrics:
        st.info("No scenario data available.")
        return

    # Convert to table data
    table_data = []
    for sm in metrics.scenario_metrics:
        table_data.append({
            "Scenario": sm.scenario_name,
            "Agents": sm.agents,
            "Intentions": sm.intentions,
            "Actions": sm.actions,
            "Dialogue": sm.dialogue_lines,
            "Branches": sm.dialogue_branch_points,
            "Complete": "✅" if sm.is_complete else "❌",
            "Has Dialogue": "✅" if sm.has_dialogue else "❌",
        })

    st.dataframe(table_data, use_container_width=True)


def render_comparison(
    baseline: ExperimentMetrics,
    treatment: ExperimentMetrics,
):
    """Render comparison between two conditions."""
    st.header("🔄 Condition Comparison")

    comparison = compare_conditions(baseline, treatment)

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Intention Completion",
            f"{comparison.intention_completion_improvement:+.1f}%",
            delta=f"{comparison.intention_completion_diff:+.4f}",
        )

    with col2:
        st.metric(
            "Executable Actions",
            f"{comparison.executable_actions_improvement:+.1f}%",
            delta=f"{comparison.executable_actions_diff:+.4f}",
        )

    with col3:
        st.metric(
            "Dialogue Lines",
            f"{comparison.dialogue_lines_improvement:+.1f}%",
            delta=f"{comparison.dialogue_lines_diff:+.1f}",
        )

    # Detailed comparison table
    st.subheader("Detailed Comparison")

    comparison_data = []
    for metric_name, data in comparison.metric_comparisons.items():
        comparison_data.append({
            "Metric": metric_name.replace("_", " ").title(),
            "Baseline": round(data["baseline"], 2),
            "Treatment": round(data["treatment"], 2),
            "Difference": round(data["diff"], 2),
            "Improvement": f"{data['improvement']:+.1f}%",
        })

    st.dataframe(comparison_data, use_container_width=True)

    # Comparison visualization
    st.subheader("📊 Comparison Chart")

    # Prepare data for chart - side by side bars
    chart_tab1, chart_tab2 = st.tabs(["Side by Side", "Improvement %"])

    with chart_tab1:
        # Side-by-side comparison of baseline vs treatment
        import pandas as pd

        metrics_for_chart = []
        for metric_name, data in comparison.metric_comparisons.items():
            metrics_for_chart.append({
                "Metric": metric_name.replace("_", " ").title(),
                "Baseline": data["baseline"],
                "Treatment": data["treatment"],
            })

        df = pd.DataFrame(metrics_for_chart)
        df = df.set_index("Metric")

        st.bar_chart(df, horizontal=True)

    with chart_tab2:
        # Improvement percentages
        improvement_data = {}
        for metric_name, data in comparison.metric_comparisons.items():
            display_name = metric_name.replace("_", " ").title()
            improvement_data[display_name] = data["improvement"]

        # Sort by improvement value for better visualization
        sorted_improvements = dict(sorted(improvement_data.items(), key=lambda x: x[1]))

        st.bar_chart(sorted_improvements, horizontal=True)

        # Color legend explanation
        st.caption("📈 Positive values = improvement, 📉 Negative values = decrease")


def render_multi_comparison(
    baseline: ExperimentMetrics,
    treatments: list,
):
    """Render comparison of multiple conditions."""
    st.header("📈 Multi-Condition Comparison")

    comparisons = compare_multiple_conditions(baseline, treatments)

    # Generate markdown table
    table_md = generate_comparison_table(comparisons)
    st.markdown(table_md)

    # Summary
    summary = summarize_comparisons(comparisons)
    st.subheader("Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Best Overall:** {summary.get('best_overall', 'N/A')}")
        st.write(f"**Comparisons:** {summary.get('num_comparisons', 0)}")

    with col2:
        intention_stats = summary.get("intention_completion", {})
        st.write(
            f"**Intention Completion Range:** "
            f"{intention_stats.get('min', 0):.1f}% to {intention_stats.get('max', 0):.1f}%"
        )


def main():
    """Main dashboard application."""
    st.title("🔬 SIA-LLM Metrics Dashboard")
    st.markdown("Automated analysis of scenario generation experiments")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Preset directories with scenario data
    PRESET_DIRECTORIES = {
        "Baseline Scenarios (Data/)": ("Data", "test_*.json", "Baseline (44 scenarios)"),
        "Test Data (Test_Data/)": ("Test_Data", "*.json", "Test Data (12 scenarios)"),
        "Custom Directory": (None, "*.json", None),
    }

    # Initialize session state for metrics
    if "baseline_metrics" not in st.session_state:
        st.session_state["baseline_metrics"] = None
    if "treatment_metrics" not in st.session_state:
        st.session_state["treatment_metrics"] = None

    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Baseline Scenarios (Data/)", "Test Data (Test_Data/)", "Custom Directory", "Upload JSON", "Demo Data"],
    )

    if data_source in ["Baseline Scenarios (Data/)", "Test Data (Test_Data/)", "Custom Directory"]:
        preset = PRESET_DIRECTORIES[data_source]

        if data_source == "Custom Directory":
            # Interactive folder browser
            st.sidebar.markdown("**📂 Browse Folders:**")
            data_dir, pattern, experiment_name = render_folder_browser(
                key_prefix="baseline"
            )
        else:
            # Use preset values
            data_dir = preset[0]
            pattern = preset[1]
            experiment_name = preset[2]
            st.sidebar.info(f"📁 {data_dir}/ ({pattern})")

            # Add explanation for Baseline preset
            if data_source == "Baseline Scenarios (Data/)":
                st.sidebar.caption(
                    "ℹ️ Uses `test_*.json` pattern to match original paper's 40 scenarios. "
                    "Use Custom Directory with `*.json` to include all 43 scenarios."
                )

        if st.sidebar.button("Load Data"):
            try:
                scenarios = load_scenarios_from_directory(data_dir, pattern)
                if scenarios:
                    st.session_state["baseline_metrics"] = calculate_experiment_metrics(
                        scenarios, experiment_name
                    )
                    st.sidebar.success(
                        f"Loaded {st.session_state['baseline_metrics'].total_scenarios} scenarios"
                    )
                else:
                    st.sidebar.warning("No scenarios found in directory")
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")

    elif data_source == "Upload JSON":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Scenarios JSON",
            type=["json"],
        )

        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    scenarios = data
                elif isinstance(data, dict) and "scenarios" in data:
                    scenarios = data["scenarios"]
                else:
                    scenarios = [data]

                # Filter to valid scenario objects
                scenarios = [s for s in scenarios if isinstance(s, dict) and ("scenario_name" in s or "agents" in s)]

                st.session_state["baseline_metrics"] = calculate_experiment_metrics(
                    scenarios,
                    experiment_name=uploaded_file.name,
                )
                st.sidebar.success(
                    f"Loaded {st.session_state['baseline_metrics'].total_scenarios} scenarios"
                )
            except Exception as e:
                st.sidebar.error(f"Error parsing JSON: {e}")

    else:  # Demo Data
        st.sidebar.info("Using demo data for visualization")

        # Create demo metrics
        demo_scenarios = [
            {
                "scenario_name": f"demo_scenario_{i}",
                "agents": {
                    f"Agent_{j}": {
                        "knowledge_base": [
                            f"BEL(fact_{k})" for k in range(3)
                        ] + [f"DES(goal_{k})" for k in range(2)],
                        "intentions": {
                            f"intent_{k}": {
                                "action_plan": [f"action_{k}"]
                            } for k in range(2)
                        },
                        "actions": {
                            f"action_{k}": {
                                "conditions": [f"BEL(fact_{k})"],
                                "effects": [f"BEL(result_{k})"],
                            } for k in range(3)
                        },
                    } for j in range(2)
                },
                "dialogue_tree": [
                    f'<S{k}, S{k+1}, Inform, Neutral, "Line {k}">'
                    for k in range(5 + i)
                ],
            }
            for i in range(5)
        ]

        st.session_state["baseline_metrics"] = calculate_experiment_metrics(
            demo_scenarios,
            experiment_name="Demo Experiment",
        )

    # Get metrics from session state
    metrics = st.session_state.get("baseline_metrics")

    # Show loaded status
    if metrics:
        st.sidebar.success(f"✓ Baseline: {metrics.experiment_name} ({metrics.total_scenarios} scenarios)")

    # Main content
    if metrics:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Overview", "Scenarios", "Export"])

        with tab1:
            render_metrics_overview(metrics)
            st.divider()
            render_artifact_chart(metrics)

        with tab2:
            render_scenario_table(metrics)

        with tab3:
            st.header("📥 Export Data")

            # JSON export
            if st.button("Export as JSON"):
                json_data = metrics.to_json(include_scenarios=True)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{metrics.experiment_name}_metrics.json",
                    mime="application/json",
                )

            # Show raw JSON
            with st.expander("View JSON"):
                st.json(metrics.to_dict())

    else:
        # Welcome message
        st.info(
            "👈 Select a data source from the sidebar to begin.\n\n"
            "You can:\n"
            "- Load scenarios from a directory\n"
            "- Upload a JSON file\n"
            "- Use demo data to explore the dashboard"
        )

    # Comparison mode
    st.sidebar.divider()
    st.sidebar.header("Comparison Mode")

    enable_comparison = st.sidebar.checkbox("Enable Condition Comparison")

    if enable_comparison:
        st.sidebar.markdown("**Select Treatment Dataset:**")

        # Treatment data source options
        treatment_options = [
            "Baseline Scenarios (Data/)",
            "Test Data (Test_Data/)",
            "Custom Directory",
        ]

        treatment_source = st.sidebar.selectbox(
            "Treatment Data",
            treatment_options,
            key="treatment_source",
        )

        # Get treatment directory settings
        if treatment_source == "Custom Directory":
            # Interactive folder browser for treatment
            st.sidebar.markdown("**📂 Browse Treatment Folders:**")
            treatment_dir, treatment_pattern, treatment_name = render_folder_browser(
                key_prefix="treatment"
            )
        else:
            preset = PRESET_DIRECTORIES[treatment_source]
            treatment_dir = preset[0]
            treatment_pattern = preset[1]
            treatment_name = preset[2]
            st.sidebar.caption(f"📁 {treatment_dir}/ ({treatment_pattern})")

        if st.sidebar.button("Load & Compare", key="compare_btn"):
            if not metrics:
                st.sidebar.error("First load a baseline dataset above")
            else:
                try:
                    treatment_scenarios = load_scenarios_from_directory(
                        treatment_dir, treatment_pattern
                    )
                    if treatment_scenarios:
                        treatment_metrics = calculate_experiment_metrics(
                            treatment_scenarios, treatment_name
                        )
                        st.session_state["treatment_metrics"] = treatment_metrics
                        st.sidebar.success(
                            f"Loaded {treatment_metrics.total_scenarios} treatment scenarios"
                        )
                    else:
                        st.sidebar.warning("No scenarios found in treatment directory")
                except Exception as e:
                    st.sidebar.error(f"Error loading treatment data: {e}")

        # Show treatment status
        treatment_metrics = st.session_state.get("treatment_metrics")
        if treatment_metrics:
            st.sidebar.success(
                f"✓ Treatment: {treatment_metrics.experiment_name} ({treatment_metrics.total_scenarios} scenarios)"
            )

        # Show comparison if we have both datasets
        if metrics and treatment_metrics:
            st.divider()
            render_comparison(metrics, treatment_metrics)


if __name__ == "__main__":
    main()
