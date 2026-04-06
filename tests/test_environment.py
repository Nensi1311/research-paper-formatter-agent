#!/usr/bin/env python3
"""
Test suite for the Research Paper Formatter OpenEnv environment.

Tests:
  1. reset() returns valid PaperObservation
  2. step() returns StepResult with reward in [0,1]
  3. state() returns full EpisodeState
  4. All 3 tasks have deterministic graders
  5. Episode terminates correctly
  6. Reward provides partial progress signals
  7. Action space coverage
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import PaperFormatterEnv
from models import PaperAction, ActionType, PaperObservation, EpisodeState, StepResult
from grader import compute_reward
from tasks import list_tasks, get_task


def test_reset_returns_valid_observation():
    for task_id in list_tasks():
        env = PaperFormatterEnv(task_id=task_id)
        obs = env.reset()
        assert isinstance(obs, PaperObservation), f"reset() must return PaperObservation for {task_id}"
        assert obs.paper_id, "paper_id must be non-empty"
        assert obs.target_format, "target_format must be set"
        assert isinstance(obs.sections, list), "sections must be a list"
        assert isinstance(obs.issues, list), "issues must be a list"
        assert 0.0 <= obs.compliance_score <= 1.0, "compliance_score must be in [0,1]"
        assert not obs.done, "episode must not start as done"
        env.close()
    print("✓ test_reset_returns_valid_observation")


def test_step_returns_valid_result():
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    action = PaperAction(action_type=ActionType.SET_COLUMN_LAYOUT, parameters={"columns": 2})
    result = env.step(action)
    assert isinstance(result, StepResult), "step() must return StepResult"
    assert isinstance(result.observation, PaperObservation), "observation must be PaperObservation"
    assert 0.0 <= result.reward <= 1.0, f"reward must be in [0,1], got {result.reward}"
    assert isinstance(result.done, bool), "done must be bool"
    assert isinstance(result.info, dict), "info must be dict"
    env.close()
    print("✓ test_step_returns_valid_result")


def test_state_returns_episode_state():
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    s = env.state()
    assert isinstance(s, EpisodeState), "state() must return EpisodeState"
    assert s.steps_taken == 0, "steps_taken should be 0 after reset"
    env.close()
    print("✓ test_state_returns_episode_state")


def test_all_three_tasks_have_graders():
    task_ids = list_tasks()
    assert len(task_ids) >= 3, f"Must have >= 3 tasks, found {len(task_ids)}"
    difficulties = [get_task(t).difficulty for t in task_ids]
    assert "easy" in difficulties, "Must have an easy task"
    assert "medium" in difficulties, "Must have a medium task"
    assert "hard" in difficulties, "Must have a hard task"
    print("✓ test_all_three_tasks_have_graders")


def test_grader_deterministic():
    """Same state → same reward, every time."""
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    state = env.state()
    r1 = compute_reward(state)
    r2 = compute_reward(state)
    assert r1.total == r2.total, "Grader must be deterministic"
    env.close()
    print("✓ test_grader_deterministic")


def test_grader_score_in_range():
    for task_id in list_tasks():
        env = PaperFormatterEnv(task_id=task_id)
        env.reset()
        state = env.state()
        reward = compute_reward(state)
        assert 0.0 <= reward.total <= 1.0, f"Reward out of range for {task_id}: {reward.total}"
        # Check sub-scores
        assert 0.0 <= reward.section_structure_score <= 1.0
        assert 0.0 <= reward.reference_format_score <= 1.0
        assert 0.0 <= reward.abstract_compliance_score <= 1.0
        assert 0.0 <= reward.author_format_score <= 1.0
        assert 0.0 <= reward.layout_score <= 1.0
        assert 0.0 <= reward.citation_style_score <= 1.0
        env.close()
    print("✓ test_grader_score_in_range")


def test_partial_progress_reward_increases():
    """Fixing one issue should increase the reward."""
    env = PaperFormatterEnv(task_id="task_easy")
    obs = env.reset()
    initial_score = obs.compliance_score

    # Fix column layout (task_easy paper starts at 1 column, needs 2)
    action = PaperAction(action_type=ActionType.SET_COLUMN_LAYOUT, parameters={"columns": 2})
    result = env.step(action)
    post_score = result.observation.compliance_score

    assert post_score >= initial_score, (
        f"Score should not decrease after fixing an issue: {initial_score:.3f} → {post_score:.3f}"
    )
    env.close()
    print(f"✓ test_partial_progress_reward_increases ({initial_score:.3f} → {post_score:.3f})")


def test_episode_terminates_on_submit():
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    action = PaperAction(action_type=ActionType.SUBMIT, parameters={})
    result = env.step(action)
    assert result.done, "Episode must be done after submit"
    env.close()
    print("✓ test_episode_terminates_on_submit")


def test_episode_terminates_on_max_steps():
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    # Exhaust all steps
    for _ in range(env.task_spec.max_steps):
        if env.state().done:
            break
        action = PaperAction(action_type=ActionType.FORMAT_CITATIONS, parameters={"style": "numeric"})
        result = env.step(action)
    assert result.done, "Episode must end after max_steps"
    env.close()
    print("✓ test_episode_terminates_on_max_steps")


def test_invalid_action_handled_gracefully():
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()
    # Try to rename a section that doesn't exist
    action = PaperAction(
        action_type=ActionType.RENAME_SECTION,
        parameters={"old_name": "NONEXISTENT", "new_name": "Something"}
    )
    result = env.step(action)
    assert result.info.get("error") is not None, "Error should be reported in info"
    assert not result.done, "Invalid action should not end episode"
    env.close()
    print("✓ test_invalid_action_handled_gracefully")


def test_hard_task_is_harder_than_easy():
    """Hard task should have more open issues than easy task at start."""
    env_easy = PaperFormatterEnv(task_id="task_easy")
    obs_easy = env_easy.reset()
    easy_issues = len(obs_easy.issues)
    env_easy.close()

    env_hard = PaperFormatterEnv(task_id="task_hard")
    obs_hard = env_hard.reset()
    hard_issues = len(obs_hard.issues)
    env_hard.close()

    assert hard_issues >= easy_issues, (
        f"Hard task ({hard_issues} issues) should have >= issues than easy ({easy_issues} issues)"
    )
    print(f"✓ test_hard_task_is_harder_than_easy (easy={easy_issues} issues, hard={hard_issues} issues)")


def test_full_easy_episode():
    """Run a scripted optimal solution for task_easy and verify success."""
    env = PaperFormatterEnv(task_id="task_easy")
    env.reset()

    # Optimal sequence for task_easy: NeurIPS → IEEE
    actions = [
        PaperAction(action_type=ActionType.SET_ABSTRACT_WORD_LIMIT, parameters={"limit": 150}),
        PaperAction(action_type=ActionType.SET_COLUMN_LAYOUT, parameters={"columns": 2}),
        PaperAction(action_type=ActionType.FORMAT_CITATIONS, parameters={"style": "numeric"}),
        PaperAction(action_type=ActionType.FORMAT_REFERENCES, parameters={"style": "IEEE"}),
        PaperAction(action_type=ActionType.FORMAT_AUTHOR_LIST, parameters={"style": "F. Last"}),
        PaperAction(action_type=ActionType.SUBMIT, parameters={}),
    ]

    final_score = 0.0
    for action in actions:
        result = env.step(action)
        final_score = result.reward
        if result.done:
            break

    env.close()
    threshold = get_task("task_easy").success_threshold
    assert final_score >= threshold, (
        f"Optimal easy solution should score >= {threshold}, got {final_score:.3f}"
    )
    print(f"✓ test_full_easy_episode (score={final_score:.3f} >= threshold={threshold})")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────

def run_all():
    tests = [
        test_reset_returns_valid_observation,
        test_step_returns_valid_result,
        test_state_returns_episode_state,
        test_all_three_tasks_have_graders,
        test_grader_deterministic,
        test_grader_score_in_range,
        test_partial_progress_reward_increases,
        test_episode_terminates_on_submit,
        test_episode_terminates_on_max_steps,
        test_invalid_action_handled_gracefully,
        test_hard_task_is_harder_than_easy,
        test_full_easy_episode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"FAILED: {failed} tests")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")


if __name__ == "__main__":
    run_all()
