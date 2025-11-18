"""Autonomous chase demo with upgraded pursuit and evasion heuristics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import mujoco.viewer as viewer
import numpy as np

from mujoco_env import ChaseMujocoEnv, SPAWN_HEIGHT

GROUND_EPS = 0.06


@dataclass
class AdaptiveChaser:
    """Predictive chaser that respects air visibility and jump timing."""

    max_ground_speed: float = 1.55
    air_speed: float = 0.95
    intercept_horizon: float = 0.45
    jump_distance: float = 1.65
    sight_loss_height: float = SPAWN_HEIGHT + 0.22
    intercept_probability: float = 0.28
    intercept_commit_steps: int = 12
    intercept_velocity_threshold: float = 0.45
    intercept_boost: float = 0.35
    seed: Optional[int] = None
    _last_sighted: np.ndarray | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _intercept_timer: int = field(default=0, init=False, repr=False)
    _intercept_dir: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64), init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, env: ChaseMujocoEnv) -> None:
        self._last_sighted = env.data.xpos[env.orange_body, :2].copy()
        self._intercept_timer = 0
        self._intercept_dir[:] = 0.0

    def step(self, env: ChaseMujocoEnv) -> np.ndarray:
        blue_body = env.blue_body
        orange_body = env.orange_body

        blue_pos = env.data.xpos[blue_body, :2]
        orange_pos = env.data.xpos[orange_body, :2]
        orange_height = env.data.xpos[orange_body, 2]
        orange_vel = env.data.cvel[orange_body, 3:5]
        orange_speed = np.linalg.norm(orange_vel)

        rel_now = orange_pos - blue_pos
        dist_now = np.linalg.norm(rel_now) + 1e-6

        if self._intercept_timer > 0:
            self._intercept_timer -= 1
            if self._intercept_timer == 0:
                self._intercept_dir[:] = 0.0

        on_ground_target = orange_height <= SPAWN_HEIGHT + GROUND_EPS
        # Occasionally commit to an advanced intercept path when the evader sprints in a clear direction.
        if (
            self._intercept_timer == 0
            and on_ground_target
            and orange_speed > self.intercept_velocity_threshold
            and self._rng.random() < self.intercept_probability
        ):
            horizon = np.clip(dist_now * 0.35, 0.18, self.intercept_horizon)
            predicted_path = orange_pos + orange_vel * horizon
            intercept_vec = predicted_path - blue_pos
            intercept_norm = np.linalg.norm(intercept_vec)
            if intercept_norm > 1e-6:
                self._intercept_dir[:] = intercept_vec / intercept_norm
                self._intercept_timer = self.intercept_commit_steps

        if orange_height <= self.sight_loss_height:
            self._last_sighted = orange_pos.copy()

        focus = self._last_sighted if self._last_sighted is not None else orange_pos

        rel = focus - blue_pos
        dist = np.linalg.norm(rel) + 1e-6
        direction = rel / dist

        ground_speed = self.max_ground_speed * np.clip(dist / 3.0, 0.45, 1.0)
        speed = ground_speed if orange_height <= SPAWN_HEIGHT + GROUND_EPS else self.air_speed

        prediction_time = np.clip(dist * 0.25, 0.1, self.intercept_horizon)
        predicted = focus + orange_vel * prediction_time
        intercept_vec = predicted - blue_pos
        intercept_dist = np.linalg.norm(intercept_vec) + 1e-6
        intercept_dir = intercept_vec / intercept_dist

        blend = np.clip(dist / 2.5, 0.2, 0.85)
        pursuit_dir = (direction * (1.0 - blend)) + (intercept_dir * blend)
        pursuit_dir /= np.linalg.norm(pursuit_dir) + 1e-6

        if self._intercept_timer > 0 and np.linalg.norm(self._intercept_dir) > 1e-6:
            progress = 1.0 - (self._intercept_timer / (self.intercept_commit_steps + 1e-6))
            commit_blend = np.clip(progress, 0.25, 0.85)
            mixed_dir = pursuit_dir * (1.0 - commit_blend) + self._intercept_dir * commit_blend
            pursuit_dir = mixed_dir / (np.linalg.norm(mixed_dir) + 1e-6)
            max_boosted = self.max_ground_speed * (1.0 + self.intercept_boost)
            speed = min(speed * (1.0 + self.intercept_boost), max_boosted)

        planar = pursuit_dir * speed

        blue_height = env.data.xpos[blue_body, 2]
        target_lower = orange_height <= SPAWN_HEIGHT + 0.08
        should_jump = dist < self.jump_distance and target_lower and blue_height <= SPAWN_HEIGHT + 0.05

        return np.array([planar[0], planar[1], 1.0 if should_jump else 0.0], dtype=np.float32)


@dataclass
class AdaptiveEvader:
    """Evader that manages jump cadence and keeps lateral escape momentum."""

    base_run: float = 1.25
    panic_run: float = 1.6
    orbit_gain: float = 0.55
    boundary_push: float = 0.9
    jump_distance: float = 1.75
    jump_cooldown: int = 24
    approach_threshold: float = 0.25
    noise_scale: float = 0.08
    boundary_slide_gain: float = 1.35
    center_recovery_gain: float = 1.1
    stuck_speed_threshold: float = 0.35
    stuck_time_limit: int = 18
    predictive_horizon: float = 0.36
    drift_decay: float = 0.42
    drift_gain: float = 0.9
    _cooldown: int = 0
    _orbit_sign: float = 1.0
    _momentum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    _stuck_steps: int = 0
    _drift_target: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def reset(self, env: ChaseMujocoEnv) -> None:  # pragma: no cover - viewer only
        self._cooldown = 0
        self._orbit_sign = 1.0 if np.random.random() < 0.5 else -1.0
        self._momentum = np.zeros(2, dtype=np.float64)
        self._stuck_steps = 0
        self._drift_target = np.zeros(2, dtype=np.float64)

    def step(self, env: ChaseMujocoEnv) -> np.ndarray:
        blue_body = env.blue_body
        orange_body = env.orange_body

        blue_pos = env.data.xpos[blue_body, :2]
        blue_vel = env.data.cvel[blue_body, 3:5]
        orange_pos = env.data.xpos[orange_body, :2]
        orange_height = env.data.xpos[orange_body, 2]

        rel = orange_pos - blue_pos
        dist = np.linalg.norm(rel) + 1e-6
        away = rel / dist

        lateral = np.array([-away[1], away[0]], dtype=np.float64) * self._orbit_sign
        run_speed = self.base_run
        if dist < 2.2:
            run_speed = np.interp(dist, [0.6, 2.2], [self.panic_run, self.base_run])

        planar = away * run_speed + lateral * self.orbit_gain

        radius = np.linalg.norm(orange_pos)
        inward = -orange_pos / (radius + 1e-6)
        orange_speed = np.linalg.norm(env.data.cvel[orange_body, 3:5])
        near_boundary = radius > env.boundary_radius - 0.75
        if near_boundary:
            pressure = np.clip((radius - (env.boundary_radius - 0.75)) / 0.75, 0.2, 1.2)
            planar += inward * (self.boundary_push * pressure)

            tangent_ccw = np.array([-inward[1], inward[0]], dtype=np.float64)
            tangent_dir = tangent_ccw
            if np.dot(rel, tangent_dir) < 0.0:
                tangent_dir = -tangent_dir
            slide = tangent_dir * (self.boundary_slide_gain * pressure)

            if self._cooldown <= 0 and np.dot(rel, tangent_dir) > 0.18:
                self._orbit_sign = np.sign(np.dot(tangent_dir, np.array([-away[1], away[0]], dtype=np.float64))) or 1.0

            planar += slide

            drift_target = orange_pos + tangent_dir * np.clip(self.predictive_horizon * pressure, 0.1, 0.5)
            self._drift_target = (1.0 - self.drift_decay) * drift_target + self.drift_decay * self._drift_target
            drift_vec = self._drift_target - orange_pos
            drift_norm = np.linalg.norm(drift_vec)
            if drift_norm > 1e-6:
                planar += (drift_vec / drift_norm) * (self.drift_gain * pressure)

        if radius > env.boundary_radius - 1.4:
            planar += inward * self.center_recovery_gain

        approach_speed = np.dot(blue_vel, -away)
        target_airborne = orange_height > SPAWN_HEIGHT + 0.08
        threat = approach_speed > self.approach_threshold and dist < self.jump_distance

        if self._cooldown > 0:
            self._cooldown -= 1

        on_ground = orange_height <= SPAWN_HEIGHT + GROUND_EPS
        should_jump = on_ground and threat and self._cooldown == 0
        if should_jump:
            self._cooldown = self.jump_cooldown

        jitter = np.random.normal(scale=self.noise_scale, size=2)
        planar += jitter

        if near_boundary and orange_speed < self.stuck_speed_threshold:
            self._stuck_steps += 1
        else:
            self._stuck_steps = max(0, self._stuck_steps - 1)

        if self._stuck_steps >= self.stuck_time_limit:
            planar += inward * (self.boundary_push * 2.4)
            planar += np.random.normal(scale=0.35, size=2)
            planar += lateral * (self.orbit_gain * 1.6)
            self._orbit_sign *= -1.0
            self._stuck_steps = int(self.stuck_time_limit * 0.4)
            self._drift_target = orange_pos + lateral * 0.6

        new_planar = 0.65 * planar + 0.35 * self._momentum
        self._momentum = new_planar
        new_planar = np.clip(new_planar, -1.0, 1.0)

        if target_airborne and dist > 2.3:
            self._cooldown = max(self._cooldown - 2, 0)

        return np.array([new_planar[0], new_planar[1], 1.0 if should_jump else 0.0], dtype=np.float32)


def main() -> None:
    env = ChaseMujocoEnv()
    env.reset()

    controller = AdaptiveChaser()
    controller.reset(env)
    orange_policy = AdaptiveEvader()
    orange_policy.reset(env)

    action = np.zeros(6, dtype=np.float32)

    print(
        "Launching jump demo: blue pursues, orange evades and jumps when pressured.\n"
        "Close the viewer window to stop the simulation."
    )

    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            handle.cam.azimuth = 140
            handle.cam.elevation = -30
            handle.cam.distance = 12.5
            handle.cam.lookat[:] = (0.0, 0.0, 0.35)

            while handle.is_running():
                blue_action = controller.step(env)
                orange_action = orange_policy.step(env)
                action[:3] = blue_action
                action[3:] = orange_action

                _, _, done, info = env.step(action)
                handle.sync()

                sleep_time = env.model.opt.timestep * env.frame_skip
                time.sleep(sleep_time)

                if done:
                    print(
                        f"Episode finished after {env.step_count} steps. "
                        f"Caught={info['caught']} timeout={info['timeout']}"
                    )
                    env.reset()
                    controller.reset(env)
                    orange_policy.reset(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
