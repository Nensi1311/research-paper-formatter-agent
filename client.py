"""
client.py — ScholarEnv OpenEnv client.

Implements the EnvClient[ActT, ObsT, StateT] interface required by the
OpenEnv checklist.  Provides typed access to the ScholarEnv server over
HTTP/WebSocket.

Usage:
    from client import ScholarEnvClient
    from models import ScholarAction

    with ScholarEnvClient(base_url="http://localhost:7860") as env:
        result = env.reset()
        obs = result.observation
        result = env.step(ScholarAction(
            task="claim_evidence_audit",
            action_type="query_section",
            section_name="abstract",
        ))
        print(result.reward)

Authors: Nensi Pansuriya · Krushna Parmar · Ishita Bhojani
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# OpenEnv client base — graceful fallback for local dev
try:
    from openenv.core import EnvClient
    from openenv.core.client_types import StepResult
    _HAS_OPENENV = True
except ImportError:
    _HAS_OPENENV = False
    class EnvClient:  # type: ignore
        """Fallback stub when openenv-core is not installed."""
        def __init__(self, base_url: str = "http://localhost:7860", **kwargs):
            self.base_url = base_url
        def __enter__(self):  return self
        def __exit__(self, *a): pass

    class StepResult:  # type: ignore
        def __init__(self, observation, reward=None, done=False):
            self.observation = observation
            self.reward = reward
            self.done = done

from models import (
    ScholarAction,
    ScholarObservation,
    FormattingAction,
    CitationAction,
    AnyAction,
)


# ── Minimal State model (used by client) ─────────────────────────────────────

from pydantic import BaseModel, Field

class ScholarState(BaseModel):
    """Episode metadata returned by GET /state."""
    episode_id:        Optional[str]   = None
    task_id:           str             = ""
    paper_id:          str             = ""
    step_count:        int             = 0
    max_steps:         int             = 6
    status:            str             = "idle"
    cumulative_score:  float           = 0.0
    curriculum:        Dict[str, Any]  = Field(default_factory=dict)


# ── Main client ───────────────────────────────────────────────────────────────

class ScholarEnvClient(EnvClient if _HAS_OPENENV else object):  # type: ignore
    """
    Typed client for the ScholarEnv HTTP/WebSocket server.

    Implements OpenEnv EnvClient[ScholarAction, ScholarObservation, ScholarState]
    following the same pattern as kube-sre-gym (1st place, SF round).
    """

    def __init__(self, base_url: str = "http://localhost:7860", **kwargs):
        if _HAS_OPENENV:
            kwargs.setdefault("message_timeout_s", 120.0)
            super().__init__(base_url=base_url, **kwargs)
        else:
            self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None

    # ── OpenEnv required methods ──────────────────────────────────────────────

    def _step_payload(self, action) -> Dict:
        """Serialise action to wire format."""
        if hasattr(action, "model_dump"):
            return action.model_dump(exclude_none=True)
        return dict(action)

    def _parse_result(self, payload: Dict) -> StepResult:
        """Deserialise step response to typed result."""
        obs_data = payload.get("observation", {})
        try:
            obs = ScholarObservation(**obs_data)
        except Exception:
            obs = ScholarObservation(
                task_id=obs_data.get("task_id", ""),
                task_description=obs_data.get("task_description", ""),
                paper_id=obs_data.get("paper_id", ""),
            )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ScholarState:
        """Deserialise state response."""
        try:
            return ScholarState(**payload)
        except Exception:
            return ScholarState()

    # ── Convenience helpers ───────────────────────────────────────────────────

    def reset_task(self, task_id: str = "claim_evidence_audit") -> StepResult:
        """Reset environment with a specific task."""
        import requests
        r = requests.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=20,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def query_section(self, task: str, section_name: str) -> StepResult:
        """Navigate to a paper section."""
        return self.step(ScholarAction(
            task=task,
            action_type="query_section",
            section_name=section_name,
        ))

    def check_table(self, task: str, table_id: str) -> StepResult:
        """Inspect a specific table."""
        return self.step(ScholarAction(
            task=task,
            action_type="check_table",
            table_id=table_id,
        ))

    def submit_findings(self, task: str, findings: list) -> StepResult:
        """Submit audit findings for scoring."""
        return self.step(ScholarAction(
            task=task,
            action_type="submit_findings",
            findings=findings,
        ))


# ── Direct HTTP fallback (when openenv-core not installed) ────────────────────

class ScholarEnvHTTPClient:
    """
    Pure-requests fallback client.  No openenv-core dependency.
    Used in Colab training where openenv-core may not be installed.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "claim_evidence_audit",
              session_id: Optional[str] = None) -> Dict:
        import requests
        payload = {"task_id": task_id}
        if session_id:
            payload["session_id"] = session_id
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=20)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict, session_id: Optional[str] = None) -> Dict:
        import requests
        if session_id:
            action = {**action, "session_id": session_id}
        r = requests.post(f"{self.base_url}/step", json=action, timeout=20)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict:
        import requests
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()
