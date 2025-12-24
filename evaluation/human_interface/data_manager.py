"""
Data management for human evaluation interface.

Handles SQLite storage and CSV/JSON export of evaluation results.
"""

import csv
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import rubrics from TASK-011
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.rubrics import EvaluationDimension, EVALUATION_RUBRICS


@dataclass
class HumanEvaluation:
    """A single human evaluation of a scenario."""
    evaluator_id: str
    scenario_name: str
    scores: Dict[str, int]  # dimension_name -> score (1-5)
    comments: str = ""
    timestamp: str = ""
    evaluation_id: Optional[int] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluator_id": self.evaluator_id,
            "scenario_name": self.scenario_name,
            "scores": self.scores,
            "comments": self.comments,
            "timestamp": self.timestamp,
        }

    def get_weighted_average(self) -> float:
        """Calculate weighted average using rubric weights."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dim_name, score in self.scores.items():
            try:
                dim = EvaluationDimension(dim_name)
                weight = EVALUATION_RUBRICS[dim].weight
                weighted_sum += score * weight
                total_weight += weight
            except (ValueError, KeyError):
                # Unknown dimension, use weight 1.0
                weighted_sum += score
                total_weight += 1.0

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class EvaluationDatabase:
    """SQLite database for storing human evaluations."""

    def __init__(self, db_path: str = "evaluations.db"):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluator_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                comments TEXT,
                timestamp TEXT NOT NULL,
                UNIQUE(evaluator_id, scenario_name)
            )
        """)

        # Scores table (one row per dimension per evaluation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL,
                dimension TEXT NOT NULL,
                score INTEGER NOT NULL,
                FOREIGN KEY (evaluation_id) REFERENCES evaluations(id),
                UNIQUE(evaluation_id, dimension)
            )
        """)

        conn.commit()
        conn.close()

    def save_evaluation(self, evaluation: HumanEvaluation) -> int:
        """
        Save an evaluation to the database.

        Args:
            evaluation: The evaluation to save

        Returns:
            The evaluation ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert or replace evaluation
            cursor.execute("""
                INSERT OR REPLACE INTO evaluations
                (evaluator_id, scenario_name, comments, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                evaluation.evaluator_id,
                evaluation.scenario_name,
                evaluation.comments,
                evaluation.timestamp,
            ))

            evaluation_id = cursor.lastrowid

            # Delete existing scores for this evaluation
            cursor.execute("""
                DELETE FROM scores WHERE evaluation_id = ?
            """, (evaluation_id,))

            # Insert scores
            for dimension, score in evaluation.scores.items():
                cursor.execute("""
                    INSERT INTO scores (evaluation_id, dimension, score)
                    VALUES (?, ?, ?)
                """, (evaluation_id, dimension, score))

            conn.commit()
            return evaluation_id

        finally:
            conn.close()

    def get_evaluation(
        self, evaluator_id: str, scenario_name: str
    ) -> Optional[HumanEvaluation]:
        """Get an existing evaluation if it exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, evaluator_id, scenario_name, comments, timestamp
                FROM evaluations
                WHERE evaluator_id = ? AND scenario_name = ?
            """, (evaluator_id, scenario_name))

            row = cursor.fetchone()
            if not row:
                return None

            eval_id, eval_id_str, scenario, comments, timestamp = row

            # Get scores
            cursor.execute("""
                SELECT dimension, score FROM scores
                WHERE evaluation_id = ?
            """, (eval_id,))

            scores = {dim: score for dim, score in cursor.fetchall()}

            return HumanEvaluation(
                evaluation_id=eval_id,
                evaluator_id=eval_id_str,
                scenario_name=scenario,
                scores=scores,
                comments=comments,
                timestamp=timestamp,
            )

        finally:
            conn.close()

    def get_all_evaluations(self) -> List[HumanEvaluation]:
        """Get all evaluations from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, evaluator_id, scenario_name, comments, timestamp
                FROM evaluations
                ORDER BY timestamp DESC
            """)

            evaluations = []
            for row in cursor.fetchall():
                eval_id, evaluator_id, scenario, comments, timestamp = row

                # Get scores for this evaluation
                cursor.execute("""
                    SELECT dimension, score FROM scores
                    WHERE evaluation_id = ?
                """, (eval_id,))

                scores = {dim: score for dim, score in cursor.fetchall()}

                evaluations.append(HumanEvaluation(
                    evaluation_id=eval_id,
                    evaluator_id=evaluator_id,
                    scenario_name=scenario,
                    scores=scores,
                    comments=comments,
                    timestamp=timestamp,
                ))

            return evaluations

        finally:
            conn.close()

    def get_evaluated_scenarios(self, evaluator_id: str) -> List[str]:
        """Get list of scenarios already evaluated by this evaluator."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT scenario_name FROM evaluations
                WHERE evaluator_id = ?
            """, (evaluator_id,))

            return [row[0] for row in cursor.fetchall()]

        finally:
            conn.close()

    def get_evaluation_count(self) -> Dict[str, int]:
        """Get count of evaluations per scenario."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT scenario_name, COUNT(*) as count
                FROM evaluations
                GROUP BY scenario_name
            """)

            return {row[0]: row[1] for row in cursor.fetchall()}

        finally:
            conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about evaluations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Total evaluations
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evals = cursor.fetchone()[0]

            # Unique evaluators
            cursor.execute("SELECT COUNT(DISTINCT evaluator_id) FROM evaluations")
            unique_evaluators = cursor.fetchone()[0]

            # Unique scenarios
            cursor.execute("SELECT COUNT(DISTINCT scenario_name) FROM evaluations")
            unique_scenarios = cursor.fetchone()[0]

            # Average scores per dimension
            cursor.execute("""
                SELECT dimension, AVG(score) as avg_score
                FROM scores
                GROUP BY dimension
            """)
            avg_scores = {row[0]: round(row[1], 2) for row in cursor.fetchall()}

            return {
                "total_evaluations": total_evals,
                "unique_evaluators": unique_evaluators,
                "unique_scenarios": unique_scenarios,
                "average_scores_by_dimension": avg_scores,
            }

        finally:
            conn.close()


def export_to_csv(
    evaluations: List[HumanEvaluation],
    output_path: str,
) -> str:
    """
    Export evaluations to CSV format.

    Args:
        evaluations: List of evaluations to export
        output_path: Path for output CSV file

    Returns:
        Path to the created file
    """
    if not evaluations:
        return output_path

    # Get all dimension names
    all_dimensions = set()
    for eval in evaluations:
        all_dimensions.update(eval.scores.keys())
    dimensions = sorted(all_dimensions)

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = [
            'evaluation_id', 'evaluator_id', 'scenario_name',
            'weighted_average', 'timestamp', 'comments'
        ] + dimensions
        writer.writerow(header)

        # Data rows
        for eval in evaluations:
            row = [
                eval.evaluation_id,
                eval.evaluator_id,
                eval.scenario_name,
                round(eval.get_weighted_average(), 2),
                eval.timestamp,
                eval.comments,
            ]
            # Add dimension scores
            for dim in dimensions:
                row.append(eval.scores.get(dim, ''))
            writer.writerow(row)

    return output_path


def export_to_json(
    evaluations: List[HumanEvaluation],
    output_path: str,
) -> str:
    """
    Export evaluations to JSON format.

    Args:
        evaluations: List of evaluations to export
        output_path: Path for output JSON file

    Returns:
        Path to the created file
    """
    data = {
        "export_timestamp": datetime.now().isoformat(),
        "total_evaluations": len(evaluations),
        "evaluations": [
            {
                **eval.to_dict(),
                "weighted_average": round(eval.get_weighted_average(), 2),
            }
            for eval in evaluations
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path
