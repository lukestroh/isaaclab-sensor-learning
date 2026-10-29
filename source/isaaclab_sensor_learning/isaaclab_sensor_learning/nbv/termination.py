#!/usr/bin/env python3
"""
Per-env termination for the NBV loop.

Each env stops on its own. Isaac Lab steps every env together, so a finished env keeps being simulated but is
frozen by the caller: it holds its pose and its observations are no longer binned. The run ends once every env is
done. Each env's stop reason, time, and counters are recorded for the evaluation log.
"""
from dataclasses import asdict, dataclass, field
from enum import IntEnum


class TerminationReason(IntEnum):
    RUNNING = 0
    COVERAGE_TARGET = 1  # coverage reached TerminationConfig.coverage_target
    TIME_BUDGET = 2  # NBV sim time reached max_sim_time_s
    GOAL_BUDGET = 3  # max_goals goals reached
    PREDICTED_GAIN_FLOOR = 4  # best candidate's predicted score fell below min_rel_predicted_gain x the first goal's
    PLATEAU = 5  # coverage gained < plateau_min_gain over the last plateau_window_goals goals
    CANDIDATES_EXHAUSTED = 6  # no unvisited candidate pose within max_search_radius
    UNREACHABLE_GOALS = 7  # too many goals timed out (env failed)


@dataclass
class TerminationConfig:
    # 1. coverage target: fraction of all instances with >= coverage_min_hits hits (instance_coverage "total").
    #    None disables it (NSGA evaluations run to the time budget so final coverage separates good rigs).
    coverage_target: float | None = 0.5
    coverage_min_hits: int = 5
    # 2. NBV sim-time budget (s)
    max_sim_time_s: float = 180.0
    # 3. goal budget
    max_goals: int = 300
    # 4a. predicted-gain floor, relative to the best score when the first goal was chosen
    min_rel_predicted_gain: float = 0.01
    # 4b. plateau: coverage (as in 1) must rise by at least plateau_min_gain over the last plateau_window_goals goals
    plateau_window_goals: int = 20
    plateau_min_gain: float = 0.005
    # 5. candidate search stops expanding past this radius (m)
    max_search_radius: float = 1.0
    # 6. a goal not reached within goal_timeout_s is skipped; too many timeouts end the env as failed
    goal_timeout_s: float = 5.0
    max_consecutive_timeouts: int = 3
    max_total_timeouts: int = 10


@dataclass
class EnvTerminationState:
    reason: TerminationReason = TerminationReason.RUNNING
    t_end: float = float("nan")  # NBV sim time (s) at which the env stopped
    step_end: int = -1  # NBV step at which the env stopped
    n_goals: int = 0  # goals assigned
    n_goals_reached: int = 0
    n_timeouts: int = 0
    consecutive_timeouts: int = 0
    goal_start_t: float = 0.0
    first_best_score: float | None = None
    coverage: float = 0.0  # latest coverage (as in TerminationConfig 1)
    coverage_at_goals: list[float] = field(default_factory=list)  # coverage each time a goal was reached

    @property
    def done(self) -> bool:
        return self.reason != TerminationReason.RUNNING


class TerminationTracker:
    """
    Tracks per-env termination. Times are NBV sim time in seconds (0 at the start of the NBV loop) and steps are NBV
    steps. The caller reports events (goal assigned / reached / coverage) and freezes envs once they are done.
    """

    def __init__(self, n_envs: int, config: TerminationConfig | None = None):
        self.config = config if config is not None else TerminationConfig()
        self.states = [EnvTerminationState() for _ in range(n_envs)]

    @property
    def active_env_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.states) if not s.done]

    @property
    def all_done(self) -> bool:
        return all(s.done for s in self.states)

    def is_done(self, env_id: int) -> bool:
        return self.states[env_id].done

    def finish(self, env_id: int, reason: TerminationReason, t: float, step: int) -> bool:
        """Stop an env; no-op if it already stopped. Returns True if this call stopped it."""
        state = self.states[env_id]
        if state.done:
            return False
        state.reason, state.t_end, state.step_end = reason, t, step
        return True

    def update(self, t: float, step: int, coverage: dict[int, float]) -> list[int]:
        """Per-step checks (coverage target, time budget). Returns the envs stopped by this call."""
        stopped = []
        for env_id in self.active_env_ids:
            state = self.states[env_id]
            state.coverage = coverage.get(env_id, state.coverage)
            if self.config.coverage_target is not None and state.coverage >= self.config.coverage_target:
                reason = TerminationReason.COVERAGE_TARGET
            elif t >= self.config.max_sim_time_s:
                reason = TerminationReason.TIME_BUDGET
            else:
                continue
            self.finish(env_id, reason, t, step)
            stopped.append(env_id)
        return stopped

    def on_goal_assigned(self, env_id: int, t: float, step: int, best_score: float) -> bool:
        """
        Record a newly chosen goal and its predicted score. Returns True if the env stopped instead (the predicted
        gain fell below the floor, or the goal budget is used up), in which case the goal should not be pursued.
        """
        state, cfg = self.states[env_id], self.config
        if state.done:
            return True
        if state.n_goals >= cfg.max_goals:
            self.finish(env_id, TerminationReason.GOAL_BUDGET, t, step)
            return True
        if state.first_best_score is None:
            state.first_best_score = best_score
        elif state.first_best_score > 0 and best_score < cfg.min_rel_predicted_gain * state.first_best_score:
            self.finish(env_id, TerminationReason.PREDICTED_GAIN_FLOOR, t, step)
            return True
        state.n_goals += 1
        state.goal_start_t = t
        return False

    def on_goal_reached(self, env_id: int, t: float, step: int) -> bool:
        """Record a reached goal; checks the plateau. Returns True if the env stopped."""
        state, cfg = self.states[env_id], self.config
        if state.done:
            return True
        state.n_goals_reached += 1
        state.consecutive_timeouts = 0
        state.coverage_at_goals.append(state.coverage)
        history = state.coverage_at_goals
        if len(history) > cfg.plateau_window_goals:
            if history[-1] - history[-1 - cfg.plateau_window_goals] < cfg.plateau_min_gain:
                return self.finish(env_id, TerminationReason.PLATEAU, t, step)
        return False

    def goal_timed_out(self, env_id: int, t: float) -> bool:
        state = self.states[env_id]
        return not state.done and t - state.goal_start_t > self.config.goal_timeout_s

    def on_goal_timeout(self, env_id: int, t: float, step: int) -> bool:
        """Record a goal that was not reached in time. Returns True if the env stopped (too many timeouts)."""
        state, cfg = self.states[env_id], self.config
        if state.done:
            return True
        state.n_timeouts += 1
        state.consecutive_timeouts += 1
        if state.consecutive_timeouts >= cfg.max_consecutive_timeouts or state.n_timeouts >= cfg.max_total_timeouts:
            return self.finish(env_id, TerminationReason.UNREACHABLE_GOALS, t, step)
        return False

    def summary(self) -> list[dict]:
        """One row per env for logging."""
        return [
            {
                "env_id": env_id,
                "reason": int(s.reason),
                "reason_name": s.reason.name,
                "t_end": s.t_end,
                "step_end": s.step_end,
                "n_goals": s.n_goals,
                "n_goals_reached": s.n_goals_reached,
                "n_timeouts": s.n_timeouts,
                "final_coverage": s.coverage,
            }
            for env_id, s in enumerate(self.states)
        ]

    def config_dict(self) -> dict:
        return asdict(self.config)
