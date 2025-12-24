"""
LLM-as-Judge evaluation system for SIA-LLM scenarios.

Uses a separate LLM instance to evaluate generated scenarios against
rubrics matching the original paper's subjective evaluation criteria.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from evaluation.rubrics import (
    EVALUATION_RUBRICS,
    EvaluationDimension,
    EvaluationRubric,
    get_all_rubrics_prompt,
    calculate_weighted_average,
)


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: EvaluationDimension
    score: int  # 1-5 Likert scale
    reasoning: str  # LLM's explanation for the score
    confidence: float = 1.0  # Optional confidence estimate


@dataclass
class EvaluationResult:
    """Complete evaluation result for a scenario."""
    scenario_name: str
    scores: Dict[EvaluationDimension, DimensionScore]
    weighted_average: float
    raw_response: str = ""
    model_used: str = ""
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "scores": {
                dim.value: {
                    "score": score.score,
                    "reasoning": score.reasoning,
                    "confidence": score.confidence,
                }
                for dim, score in self.scores.items()
            },
            "weighted_average": round(self.weighted_average, 2),
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
        }

    def get_summary(self) -> str:
        """Get a summary of the evaluation."""
        lines = [
            f"Evaluation: {self.scenario_name}",
            f"Weighted Average: {self.weighted_average:.2f}/5.00",
            "",
            "Dimension Scores:",
        ]
        for dim in EvaluationDimension:
            if dim in self.scores:
                score = self.scores[dim]
                rubric = EVALUATION_RUBRICS[dim]
                lines.append(f"  {rubric.name}: {score.score}/5")
        return "\n".join(lines)


class LLMJudge:
    """
    LLM-as-Judge for evaluating generated scenarios.

    Uses structured prompting to get consistent evaluations across
    multiple dimensions using a 1-5 Likert scale.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        dimensions: Optional[List[EvaluationDimension]] = None,
    ):
        """
        Initialize the LLM judge.

        Args:
            model_name: Model to use for evaluation (default: gpt-4o)
            dimensions: Which dimensions to evaluate (default: all)
        """
        self.model_name = model_name
        self.dimensions = dimensions or list(EvaluationDimension)
        self._model = None

    def _get_model(self):
        """Lazy load the model handler."""
        if self._model is None:
            from models import get_model
            self._model = get_model(model_name=self.model_name)
        return self._model

    def _format_scenario_for_evaluation(
        self, scenario: Dict[str, Any]
    ) -> str:
        """Format scenario data for LLM evaluation."""
        lines = []

        # Scenario description
        lines.append("## SCENARIO DESCRIPTION")
        lines.append(scenario.get("scenario_description", "No description"))
        lines.append("")

        # Agents
        lines.append("## AGENTS")
        agents = scenario.get("agents", {})
        for agent_name, agent_data in agents.items():
            lines.append(f"\n### Agent: {agent_name}")

            # Knowledge base (beliefs/desires)
            kb = agent_data.get("knowledge_base", [])
            if kb:
                lines.append("\n**Knowledge Base (Beliefs/Desires):**")
                for item in kb[:20]:  # Limit to first 20
                    lines.append(f"  - {item}")
                if len(kb) > 20:
                    lines.append(f"  ... and {len(kb) - 20} more")

            # Intentions
            intentions = agent_data.get("intentions", {})
            if intentions:
                lines.append(f"\n**Intentions:** {len(intentions)}")
                for intent_name, intent_data in list(intentions.items())[:5]:
                    action_plan = intent_data.get("action_plan", [])
                    lines.append(f"  - {intent_name}: {len(action_plan)} actions")

            # Actions (sample)
            actions = agent_data.get("actions", {})
            if actions:
                lines.append(f"\n**Actions:** {len(actions)} total")
                for action_name, action_data in list(actions.items())[:3]:
                    conds = action_data.get("conditions", [])
                    effs = action_data.get("effects", [])
                    lines.append(f"  - {action_name}")
                    lines.append(f"    Conditions: {len(conds)}, Effects: {len(effs)}")

            # Speak actions
            speak_actions = agent_data.get("speak_actions", {})
            if speak_actions:
                lines.append(f"\n**Speak Actions:** {len(speak_actions)}")

        # Dialogue tree
        dialogue = scenario.get("dialogue_tree", [])
        if dialogue:
            lines.append("\n## DIALOGUE TREE")
            lines.append(f"Total lines: {len(dialogue)}")
            for i, line in enumerate(dialogue[:10]):
                lines.append(f"  {i+1}. {line}")
            if len(dialogue) > 10:
                lines.append(f"  ... and {len(dialogue) - 10} more lines")

        return "\n".join(lines)

    def _build_evaluation_prompt(
        self,
        scenario_text: str,
        dimensions: List[EvaluationDimension],
    ) -> str:
        """Build the evaluation prompt for the LLM."""
        # Get rubrics for requested dimensions
        rubric_sections = []
        for dim in dimensions:
            rubric = EVALUATION_RUBRICS[dim]
            rubric_sections.append(rubric.get_prompt_section())

        rubrics_text = "\n\n".join(rubric_sections)

        # Build dimension list for output format
        dim_list = ", ".join(f'"{dim.value}"' for dim in dimensions)

        prompt = f"""You are an expert evaluator for social simulation scenarios designed for the FAtiMA Toolkit (BDI architecture).

Your task is to evaluate a generated scenario against specific quality dimensions using a 1-5 Likert scale.

## EVALUATION RUBRICS

{rubrics_text}

## SCENARIO TO EVALUATE

{scenario_text}

## INSTRUCTIONS

1. Carefully read the scenario and all its components
2. For each dimension, assign a score from 1-5 based on the rubric
3. Provide brief reasoning (1-2 sentences) for each score
4. Be objective and consistent with the rubric criteria

## OUTPUT FORMAT

Respond with a valid JSON object containing your evaluation:

```json
{{
  "evaluations": [
    {{
      "dimension": "[dimension_name]",
      "score": [1-5],
      "reasoning": "[brief explanation]"
    }}
  ]
}}
```

Evaluate these dimensions: {dim_list}

Provide your evaluation now:"""

        return prompt

    def _parse_evaluation_response(
        self,
        response: str,
        dimensions: List[EvaluationDimension],
    ) -> Dict[EvaluationDimension, DimensionScore]:
        """Parse the LLM response into structured scores."""
        scores = {}

        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            # If no JSON found, try to parse line by line
            return self._parse_freeform_response(response, dimensions)

        try:
            data = json.loads(json_match.group())
            evaluations = data.get("evaluations", [])

            for eval_item in evaluations:
                dim_name = eval_item.get("dimension", "")
                score_val = eval_item.get("score", 3)
                reasoning = eval_item.get("reasoning", "")

                # Find matching dimension
                for dim in dimensions:
                    if dim.value == dim_name or dim.name.lower() == dim_name.lower():
                        # Validate score range
                        score_val = max(1, min(5, int(score_val)))
                        scores[dim] = DimensionScore(
                            dimension=dim,
                            score=score_val,
                            reasoning=reasoning,
                        )
                        break

        except json.JSONDecodeError:
            return self._parse_freeform_response(response, dimensions)

        # Fill in missing dimensions with default scores
        for dim in dimensions:
            if dim not in scores:
                scores[dim] = DimensionScore(
                    dimension=dim,
                    score=3,  # Default to average
                    reasoning="Could not parse score from response",
                    confidence=0.5,
                )

        return scores

    def _parse_freeform_response(
        self,
        response: str,
        dimensions: List[EvaluationDimension],
    ) -> Dict[EvaluationDimension, DimensionScore]:
        """Parse a non-JSON response by looking for patterns."""
        scores = {}
        response_lower = response.lower()

        for dim in dimensions:
            rubric = EVALUATION_RUBRICS[dim]
            dim_name = rubric.name.lower()

            # Look for pattern like "Agent Relevance: 4" or "agent_relevance: 4/5"
            patterns = [
                rf'{dim_name}[:\s]+(\d)',
                rf'{dim.value}[:\s]+(\d)',
                rf'{dim_name}.*?(\d)/5',
            ]

            score_val = 3  # Default
            reasoning = "Parsed from freeform response"

            for pattern in patterns:
                match = re.search(pattern, response_lower)
                if match:
                    score_val = int(match.group(1))
                    score_val = max(1, min(5, score_val))
                    break

            scores[dim] = DimensionScore(
                dimension=dim,
                score=score_val,
                reasoning=reasoning,
                confidence=0.7,
            )

        return scores

    def evaluate(
        self,
        scenario: Dict[str, Any],
        dimensions: Optional[List[EvaluationDimension]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single scenario.

        Args:
            scenario: The scenario dict to evaluate
            dimensions: Which dimensions to evaluate (default: self.dimensions)

        Returns:
            EvaluationResult with scores for each dimension
        """
        dims = dimensions or self.dimensions
        scenario_name = scenario.get("scenario_name", "unknown")

        # Format scenario for evaluation
        scenario_text = self._format_scenario_for_evaluation(scenario)

        # Build prompt
        prompt = self._build_evaluation_prompt(scenario_text, dims)

        # Call LLM
        model = self._get_model()
        response = model.request(prompt)

        # Parse response
        scores = self._parse_evaluation_response(response, dims)

        # Calculate weighted average
        score_dict = {dim: s.score for dim, s in scores.items()}
        weighted_avg = calculate_weighted_average(score_dict)

        return EvaluationResult(
            scenario_name=scenario_name,
            scores=scores,
            weighted_average=weighted_avg,
            raw_response=response,
            model_used=self.model_name,
            tokens_used=getattr(model, 'last_tokens_used', 0),
        )

    def evaluate_batch(
        self,
        scenarios: List[Dict[str, Any]],
        dimensions: Optional[List[EvaluationDimension]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple scenarios."""
        results = []
        for scenario in scenarios:
            result = self.evaluate(scenario, dimensions)
            results.append(result)
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_scenario(
    scenario: Dict[str, Any],
    model_name: str = "gpt-4o",
    dimensions: Optional[List[EvaluationDimension]] = None,
) -> EvaluationResult:
    """
    Evaluate a single scenario using LLM-as-Judge.

    Args:
        scenario: The scenario dict to evaluate
        model_name: Model to use for evaluation
        dimensions: Which dimensions to evaluate (default: all)

    Returns:
        EvaluationResult with scores
    """
    judge = LLMJudge(model_name=model_name, dimensions=dimensions)
    return judge.evaluate(scenario)


def evaluate_scenarios_batch(
    scenarios: List[Dict[str, Any]],
    model_name: str = "gpt-4o",
    dimensions: Optional[List[EvaluationDimension]] = None,
) -> List[EvaluationResult]:
    """Evaluate multiple scenarios."""
    judge = LLMJudge(model_name=model_name, dimensions=dimensions)
    return judge.evaluate_batch(scenarios, dimensions)


def compare_evaluations(
    baseline_results: List[EvaluationResult],
    improved_results: List[EvaluationResult],
) -> Dict[str, Any]:
    """
    Compare evaluation results between baseline and improved scenarios.

    Returns summary statistics for each dimension.
    """
    comparison = {
        "n_baseline": len(baseline_results),
        "n_improved": len(improved_results),
        "dimensions": {},
    }

    for dim in EvaluationDimension:
        baseline_scores = [
            r.scores[dim].score
            for r in baseline_results
            if dim in r.scores
        ]
        improved_scores = [
            r.scores[dim].score
            for r in improved_results
            if dim in r.scores
        ]

        if baseline_scores and improved_scores:
            baseline_avg = sum(baseline_scores) / len(baseline_scores)
            improved_avg = sum(improved_scores) / len(improved_scores)

            comparison["dimensions"][dim.value] = {
                "baseline_avg": round(baseline_avg, 2),
                "improved_avg": round(improved_avg, 2),
                "improvement": round(improved_avg - baseline_avg, 2),
                "improvement_pct": round(
                    (improved_avg - baseline_avg) / baseline_avg * 100, 1
                ) if baseline_avg > 0 else 0,
            }

    # Overall comparison
    baseline_overall = [r.weighted_average for r in baseline_results]
    improved_overall = [r.weighted_average for r in improved_results]

    if baseline_overall and improved_overall:
        comparison["overall"] = {
            "baseline_avg": round(sum(baseline_overall) / len(baseline_overall), 2),
            "improved_avg": round(sum(improved_overall) / len(improved_overall), 2),
        }
        comparison["overall"]["improvement"] = round(
            comparison["overall"]["improved_avg"] - comparison["overall"]["baseline_avg"],
            2,
        )

    return comparison


def print_evaluation_report(result: EvaluationResult):
    """Print a formatted evaluation report."""
    print("=" * 60)
    print(f"EVALUATION REPORT: {result.scenario_name}")
    print("=" * 60)
    print(f"Model: {result.model_used}")
    print(f"Weighted Average: {result.weighted_average:.2f}/5.00")
    print()

    print("DIMENSION SCORES:")
    print("-" * 60)

    for dim in EvaluationDimension:
        if dim in result.scores:
            score = result.scores[dim]
            rubric = EVALUATION_RUBRICS[dim]

            # Score bar visualization
            bar = "█" * score.score + "░" * (5 - score.score)

            print(f"\n{rubric.name}:")
            print(f"  Score: [{bar}] {score.score}/5")
            print(f"  Reasoning: {score.reasoning[:100]}...")

    print()
    print("=" * 60)
