"""
HTTP client for the Paper Formatter environment.
Use this when the environment is running as a separate server
(e.g., HuggingFace Space or Docker container).

Usage:
    client = PaperFormatterClient("https://your-space.hf.space")
    obs = client.reset("task_easy")
    result = client.step("set_column_layout", {"columns": 2})
    state = client.state()
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

from models import PaperObservation, EpisodeState, StepResult, PaperAction, ActionType


class PaperFormatterClient:
    """HTTP client wrapping the Paper Formatter OpenEnv server."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str = "task_easy") -> PaperObservation:
        r = httpx.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return PaperObservation.model_validate(r.json())

    def step(self, action_type: str, parameters: Dict[str, Any] = {}) -> StepResult:
        r = httpx.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "parameters": parameters},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    def state(self) -> EpisodeState:
        r = httpx.get(f"{self.base_url}/state", timeout=self.timeout)
        r.raise_for_status()
        return EpisodeState.model_validate(r.json())

    def tasks(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base_url}/tasks", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def action_space(self) -> Dict[str, Any]:
        r = httpx.get(f"{self.base_url}/action_space", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        """No-op for HTTP client (stateless server)."""
        pass
