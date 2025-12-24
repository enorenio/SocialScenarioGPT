"""
Structured logging system for SIA-LLM experiments.
Captures API calls, token usage, timing, and intermediate states.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.feature_flags import FeatureFlags


@dataclass
class APICallLog:
    """Log entry for a single API call."""
    timestamp: str
    step_name: str
    model: str
    prompt_messages: List[Dict[str, str]]
    response_content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class StepLog:
    """Log entry for a pipeline step."""
    step_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    api_calls: int
    total_tokens: int
    artifacts_generated: Dict[str, int]
    success: bool
    error: Optional[str] = None


@dataclass
class ScenarioLog:
    """Complete log for a scenario generation run."""
    scenario_name: str
    scenario_description: str
    feature_flags: Dict[str, bool]
    model: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_seconds: float = 0.0
    total_api_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    steps: List[StepLog] = field(default_factory=list)
    api_calls: List[APICallLog] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None


class ExperimentLogger:
    """
    Logger for tracking experiment runs.

    Usage:
        logger = ExperimentLogger("experiments/logs", flags)
        logger.start_scenario("test_scenario", "Description...")

        with logger.step("get_agents"):
            # do work
            logger.log_api_call(...)

        logger.end_scenario(final_state, success=True)
    """

    def __init__(
        self,
        log_dir: str = "experiments/logs",
        feature_flags: Optional[FeatureFlags] = None,
        model: str = "gpt-3.5-turbo",
        console_level: int = logging.INFO,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.feature_flags = feature_flags or FeatureFlags()
        self.model = model

        self.current_scenario: Optional[ScenarioLog] = None
        self.current_step: Optional[str] = None
        self._step_start_time: Optional[float] = None
        self._step_api_calls: int = 0
        self._step_tokens: int = 0
        self._step_artifacts: Dict[str, int] = {}

        # Setup Python logger for console output
        self._setup_console_logger(console_level)

    def _setup_console_logger(self, level: int):
        """Setup console logging."""
        self.console = logging.getLogger("experiment")
        self.console.setLevel(level)

        if not self.console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S"
            ))
            self.console.addHandler(handler)

    def start_scenario(self, name: str, description: str):
        """Start logging a new scenario."""
        self.current_scenario = ScenarioLog(
            scenario_name=name,
            scenario_description=description,
            feature_flags=self.feature_flags.to_dict(),
            model=self.model,
            start_time=datetime.now().isoformat(),
        )
        self.console.info(f"Starting scenario: {name}")
        self.console.info(f"Features: {self.feature_flags.enabled_features()}")

    def end_scenario(
        self,
        final_state: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """End logging for current scenario and save."""
        if not self.current_scenario:
            return

        self.current_scenario.end_time = datetime.now().isoformat()
        self.current_scenario.final_state = final_state
        self.current_scenario.success = success
        self.current_scenario.error = error

        # Calculate totals
        start = datetime.fromisoformat(self.current_scenario.start_time)
        end = datetime.fromisoformat(self.current_scenario.end_time)
        self.current_scenario.total_duration_seconds = (end - start).total_seconds()

        self.current_scenario.total_api_calls = len(self.current_scenario.api_calls)
        self.current_scenario.total_prompt_tokens = sum(
            c.prompt_tokens for c in self.current_scenario.api_calls
        )
        self.current_scenario.total_completion_tokens = sum(
            c.completion_tokens for c in self.current_scenario.api_calls
        )
        self.current_scenario.total_tokens = sum(
            c.total_tokens for c in self.current_scenario.api_calls
        )

        # Save log
        self._save_scenario_log()

        status = "SUCCESS" if success else f"FAILED: {error}"
        self.console.info(
            f"Scenario complete: {status} | "
            f"Duration: {self.current_scenario.total_duration_seconds:.1f}s | "
            f"API calls: {self.current_scenario.total_api_calls} | "
            f"Tokens: {self.current_scenario.total_tokens}"
        )

        self.current_scenario = None

    def step(self, step_name: str):
        """Context manager for logging a pipeline step."""
        return _StepContext(self, step_name)

    def _start_step(self, step_name: str):
        """Internal: start a step."""
        self.current_step = step_name
        self._step_start_time = time.time()
        self._step_api_calls = 0
        self._step_tokens = 0
        self._step_artifacts = {}
        self.console.debug(f"  Step: {step_name}")

    def _end_step(self, success: bool = True, error: Optional[str] = None):
        """Internal: end current step."""
        if not self.current_step or not self.current_scenario:
            return

        duration = time.time() - self._step_start_time

        step_log = StepLog(
            step_name=self.current_step,
            start_time=datetime.fromtimestamp(self._step_start_time).isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=duration,
            api_calls=self._step_api_calls,
            total_tokens=self._step_tokens,
            artifacts_generated=self._step_artifacts,
            success=success,
            error=error,
        )
        self.current_scenario.steps.append(step_log)

        self.current_step = None

    def log_api_call(
        self,
        prompt_messages: List[Dict[str, str]],
        response_content: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_seconds: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Log an API call."""
        if not self.current_scenario:
            return

        call_log = APICallLog(
            timestamp=datetime.now().isoformat(),
            step_name=self.current_step or "unknown",
            model=self.model,
            prompt_messages=prompt_messages,
            response_content=response_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            duration_seconds=duration_seconds,
            success=success,
            error=error,
        )
        self.current_scenario.api_calls.append(call_log)

        self._step_api_calls += 1
        self._step_tokens += call_log.total_tokens

    def log_artifacts(self, artifact_type: str, count: int):
        """Log generated artifacts for current step."""
        self._step_artifacts[artifact_type] = count

    def _save_scenario_log(self):
        """Save scenario log to JSON file."""
        if not self.current_scenario:
            return

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_scenario.scenario_name}_{timestamp}.json"
        filepath = self.log_dir / filename

        # Convert to dict (handle dataclasses)
        log_dict = asdict(self.current_scenario)

        with open(filepath, "w") as f:
            json.dump(log_dict, f, indent=2, default=str)

        self.console.debug(f"Log saved: {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current scenario."""
        if not self.current_scenario:
            return {}

        return {
            "scenario_name": self.current_scenario.scenario_name,
            "duration_seconds": self.current_scenario.total_duration_seconds,
            "api_calls": self.current_scenario.total_api_calls,
            "total_tokens": self.current_scenario.total_tokens,
            "steps_completed": len(self.current_scenario.steps),
            "success": self.current_scenario.success,
        }


class _StepContext:
    """Context manager for step logging."""

    def __init__(self, logger: ExperimentLogger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.error: Optional[str] = None

    def __enter__(self):
        self.logger._start_step(self.step_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self.logger._end_step(success=success, error=error)
        return False  # Don't suppress exceptions
