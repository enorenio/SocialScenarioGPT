"""
Tests for TASK-012: Human Evaluation Interface
Tests the data manager component (SQLite storage and export).
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.human_interface.data_manager import (
    EvaluationDatabase,
    HumanEvaluation,
    export_to_csv,
    export_to_json,
)
from evaluation.rubrics import EvaluationDimension


# ============================================================
# HumanEvaluation Tests
# ============================================================

def test_human_evaluation_init():
    """Test HumanEvaluation initialization."""
    evaluation = HumanEvaluation(
        evaluator_id="test_user",
        scenario_name="test_scenario",
        scores={"agent_relevance": 4, "belief_coherence": 3},
        comments="Test comment",
    )

    assert evaluation.evaluator_id == "test_user"
    assert evaluation.scenario_name == "test_scenario"
    assert evaluation.scores["agent_relevance"] == 4
    assert evaluation.comments == "Test comment"
    assert evaluation.timestamp  # Should be auto-set

    print("✓ HumanEvaluation initializes correctly")


def test_human_evaluation_to_dict():
    """Test converting evaluation to dictionary."""
    evaluation = HumanEvaluation(
        evaluator_id="test_user",
        scenario_name="test_scenario",
        scores={"agent_relevance": 4},
    )

    d = evaluation.to_dict()

    assert d["evaluator_id"] == "test_user"
    assert d["scenario_name"] == "test_scenario"
    assert d["scores"]["agent_relevance"] == 4
    assert "timestamp" in d

    print("✓ HumanEvaluation.to_dict() works")


def test_human_evaluation_weighted_average():
    """Test weighted average calculation."""
    evaluation = HumanEvaluation(
        evaluator_id="test",
        scenario_name="test",
        scores={
            "agent_relevance": 4,  # weight 1.0
            "belief_coherence": 3,  # weight 1.2
            "action_feasibility": 5,  # weight 1.5
        },
    )

    avg = evaluation.get_weighted_average()

    # (4*1.0 + 3*1.2 + 5*1.5) / (1.0 + 1.2 + 1.5) = 15.1 / 3.7 ≈ 4.08
    assert 4.0 <= avg <= 4.2

    print(f"✓ HumanEvaluation.get_weighted_average() returns {avg:.2f}")


def test_human_evaluation_weighted_average_all_dimensions():
    """Test weighted average with all dimensions."""
    evaluation = HumanEvaluation(
        evaluator_id="test",
        scenario_name="test",
        scores={dim.value: 3 for dim in EvaluationDimension},
    )

    avg = evaluation.get_weighted_average()

    # All 3s should give approximately 3.0 (floating point)
    assert 2.99 <= avg <= 3.01

    print(f"✓ Weighted average of all 3s is {avg:.2f}")


# ============================================================
# EvaluationDatabase Tests
# ============================================================

def test_database_init():
    """Test database initialization."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)
        assert Path(db_path).exists()
        print("✓ EvaluationDatabase initializes correctly")
    finally:
        os.unlink(db_path)


def test_database_save_and_get():
    """Test saving and retrieving an evaluation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        evaluation = HumanEvaluation(
            evaluator_id="tester",
            scenario_name="test_scenario",
            scores={"agent_relevance": 4, "belief_coherence": 5},
            comments="Great scenario!",
        )

        # Save
        eval_id = db.save_evaluation(evaluation)
        assert eval_id > 0

        # Retrieve
        retrieved = db.get_evaluation("tester", "test_scenario")
        assert retrieved is not None
        assert retrieved.evaluator_id == "tester"
        assert retrieved.scores["agent_relevance"] == 4
        assert retrieved.scores["belief_coherence"] == 5
        assert retrieved.comments == "Great scenario!"

        print("✓ Database save and get work correctly")

    finally:
        os.unlink(db_path)


def test_database_update_evaluation():
    """Test updating an existing evaluation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # First evaluation
        eval1 = HumanEvaluation(
            evaluator_id="tester",
            scenario_name="test_scenario",
            scores={"agent_relevance": 3},
        )
        db.save_evaluation(eval1)

        # Update with new score
        eval2 = HumanEvaluation(
            evaluator_id="tester",
            scenario_name="test_scenario",
            scores={"agent_relevance": 5},
        )
        db.save_evaluation(eval2)

        # Should have updated score
        retrieved = db.get_evaluation("tester", "test_scenario")
        assert retrieved.scores["agent_relevance"] == 5

        print("✓ Database updates evaluations correctly")

    finally:
        os.unlink(db_path)


def test_database_get_nonexistent():
    """Test getting a non-existent evaluation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        result = db.get_evaluation("nobody", "nothing")
        assert result is None

        print("✓ Database returns None for non-existent evaluation")

    finally:
        os.unlink(db_path)


def test_database_get_all_evaluations():
    """Test getting all evaluations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # Add multiple evaluations
        for i in range(3):
            evaluation = HumanEvaluation(
                evaluator_id=f"tester_{i}",
                scenario_name=f"scenario_{i}",
                scores={"agent_relevance": i + 3},
            )
            db.save_evaluation(evaluation)

        # Get all
        all_evals = db.get_all_evaluations()
        assert len(all_evals) == 3

        print("✓ Database get_all_evaluations() works")

    finally:
        os.unlink(db_path)


def test_database_get_evaluated_scenarios():
    """Test getting list of evaluated scenarios for a user."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # User evaluates two scenarios
        for scenario in ["scenario_a", "scenario_b"]:
            evaluation = HumanEvaluation(
                evaluator_id="tester",
                scenario_name=scenario,
                scores={"agent_relevance": 4},
            )
            db.save_evaluation(evaluation)

        # Different user evaluates one
        db.save_evaluation(HumanEvaluation(
            evaluator_id="other_tester",
            scenario_name="scenario_c",
            scores={"agent_relevance": 3},
        ))

        # Check tester's scenarios
        tester_scenarios = db.get_evaluated_scenarios("tester")
        assert len(tester_scenarios) == 2
        assert "scenario_a" in tester_scenarios
        assert "scenario_b" in tester_scenarios
        assert "scenario_c" not in tester_scenarios

        print("✓ Database get_evaluated_scenarios() works")

    finally:
        os.unlink(db_path)


def test_database_get_evaluation_count():
    """Test getting evaluation count per scenario."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # Multiple users rate same scenario
        for user in ["user1", "user2", "user3"]:
            evaluation = HumanEvaluation(
                evaluator_id=user,
                scenario_name="popular_scenario",
                scores={"agent_relevance": 4},
            )
            db.save_evaluation(evaluation)

        # One user rates another
        db.save_evaluation(HumanEvaluation(
            evaluator_id="user1",
            scenario_name="other_scenario",
            scores={"agent_relevance": 3},
        ))

        counts = db.get_evaluation_count()
        assert counts["popular_scenario"] == 3
        assert counts["other_scenario"] == 1

        print("✓ Database get_evaluation_count() works")

    finally:
        os.unlink(db_path)


def test_database_get_statistics():
    """Test getting overall statistics."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # Add some evaluations
        db.save_evaluation(HumanEvaluation(
            evaluator_id="user1",
            scenario_name="scenario_a",
            scores={"agent_relevance": 4, "belief_coherence": 3},
        ))
        db.save_evaluation(HumanEvaluation(
            evaluator_id="user2",
            scenario_name="scenario_a",
            scores={"agent_relevance": 5, "belief_coherence": 4},
        ))
        db.save_evaluation(HumanEvaluation(
            evaluator_id="user1",
            scenario_name="scenario_b",
            scores={"agent_relevance": 3},
        ))

        stats = db.get_statistics()

        assert stats["total_evaluations"] == 3
        assert stats["unique_evaluators"] == 2
        assert stats["unique_scenarios"] == 2
        assert "agent_relevance" in stats["average_scores_by_dimension"]

        # Average of 4, 5, 3 for agent_relevance = 4.0
        assert stats["average_scores_by_dimension"]["agent_relevance"] == 4.0

        print("✓ Database get_statistics() works")

    finally:
        os.unlink(db_path)


# ============================================================
# Export Tests
# ============================================================

def test_export_to_csv():
    """Test exporting evaluations to CSV."""
    evaluations = [
        HumanEvaluation(
            evaluator_id="user1",
            scenario_name="scenario_a",
            scores={"agent_relevance": 4, "belief_coherence": 3},
            comments="Good",
        ),
        HumanEvaluation(
            evaluator_id="user2",
            scenario_name="scenario_b",
            scores={"agent_relevance": 5},
            comments="Excellent",
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        export_to_csv(evaluations, csv_path)

        # Check file exists and has content
        assert Path(csv_path).exists()
        with open(csv_path) as f:
            content = f.read()
            assert "evaluator_id" in content
            assert "user1" in content
            assert "user2" in content
            assert "agent_relevance" in content

        print("✓ export_to_csv() works")

    finally:
        os.unlink(csv_path)


def test_export_to_json():
    """Test exporting evaluations to JSON."""
    evaluations = [
        HumanEvaluation(
            evaluator_id="user1",
            scenario_name="scenario_a",
            scores={"agent_relevance": 4},
        ),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name

    try:
        export_to_json(evaluations, json_path)

        # Check file exists and has valid JSON
        assert Path(json_path).exists()
        with open(json_path) as f:
            data = json.load(f)
            assert "export_timestamp" in data
            assert data["total_evaluations"] == 1
            assert len(data["evaluations"]) == 1
            assert data["evaluations"][0]["evaluator_id"] == "user1"

        print("✓ export_to_json() works")

    finally:
        os.unlink(json_path)


def test_export_empty():
    """Test exporting empty evaluation list."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name

    try:
        export_to_csv([], csv_path)
        # Should not crash
        print("✓ export_to_csv() handles empty list")

    finally:
        if Path(csv_path).exists():
            os.unlink(csv_path)


# ============================================================
# Integration Tests
# ============================================================

def test_full_workflow():
    """Test complete evaluation workflow."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = EvaluationDatabase(db_path)

        # Simulate evaluation workflow
        # 1. User starts evaluating
        evaluator_id = "researcher_1"

        # 2. Check what's already evaluated (nothing)
        evaluated = db.get_evaluated_scenarios(evaluator_id)
        assert len(evaluated) == 0

        # 3. Submit first evaluation
        eval1 = HumanEvaluation(
            evaluator_id=evaluator_id,
            scenario_name="test_Brother",
            scores={dim.value: 3 for dim in EvaluationDimension},
            comments="Average scenario, needs improvement",
        )
        db.save_evaluation(eval1)

        # 4. Check progress
        evaluated = db.get_evaluated_scenarios(evaluator_id)
        assert len(evaluated) == 1

        # 5. Submit second evaluation with different scores
        eval2 = HumanEvaluation(
            evaluator_id=evaluator_id,
            scenario_name="test_Mummies",
            scores={
                "agent_relevance": 5,
                "belief_coherence": 4,
                "dialogue_quality": 5,
                "overall_coherence": 4,
            },
            comments="Well-structured scenario",
        )
        db.save_evaluation(eval2)

        # 6. Get statistics
        stats = db.get_statistics()
        assert stats["total_evaluations"] == 2
        assert stats["unique_evaluators"] == 1

        # 7. Export results
        all_evals = db.get_all_evaluations()
        assert len(all_evals) == 2

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json_path = f.name

        export_to_json(all_evals, json_path)

        with open(json_path) as f:
            exported = json.load(f)
            assert exported["total_evaluations"] == 2

        os.unlink(json_path)

        print("✓ Full evaluation workflow works")

    finally:
        os.unlink(db_path)


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("TASK-012: Human Evaluation Interface Tests")
    print("=" * 60)

    # HumanEvaluation tests
    test_human_evaluation_init()
    test_human_evaluation_to_dict()
    test_human_evaluation_weighted_average()
    test_human_evaluation_weighted_average_all_dimensions()

    # Database tests
    test_database_init()
    test_database_save_and_get()
    test_database_update_evaluation()
    test_database_get_nonexistent()
    test_database_get_all_evaluations()
    test_database_get_evaluated_scenarios()
    test_database_get_evaluation_count()
    test_database_get_statistics()

    # Export tests
    test_export_to_csv()
    test_export_to_json()
    test_export_empty()

    # Integration tests
    test_full_workflow()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
