"""Gymnasium-compatible wrappers around the MuJoCo chase environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "gymnasium is required for headless training. Install it with 'pip install gymnasium'."
    ) from exc

from mujoco_env import ChaseMujocoEnv, SPAWN_HEIGHT

VISIBILITY_EPS = 0.06


@dataclass
class HeuristicEvader:
    """Rich evasion controller that mixes patrol, dodges, and boundary awareness."""

    jump_distance: float = 1.85
    orbit_scale: float = 0.35
    escape_scale: float = 1.05
    tangential_bias: float = 0.45
    boundary_threshold: float = 0.7
    boundary_push: float = 1.1
    visibility_eps: float = VISIBILITY_EPS
    orbit_flip_prob: float = 0.05
    base_noise: float = 0.02
    noise_close_scale: float = 0.18
    noise_far_scale: float = 0.35
    home_radius: float = 3.4
    home_strength: float = 0.7
    danger_distance: float = 1.9
    danger_dodge_gain: float = 1.15
    patrol_radius: float = 2.8
    patrol_strength: float = 0.55
    patrol_interval: int = 120
    center_preference: float = 0.32
    momentum_weight: float = 0.35
    boundary_linger_limit: int = 110
    air_intercept_distance: float = 2.4
    air_intercept_gain: float = 1.2
    air_predict_horizon: float = 0.32
    bait_window: float = 1.6
    bait_relax_threshold: float = 1.05
    bait_push: float = 0.65
    seed: Optional[int] = None
    _last_seen_blue: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _orbit_sign: float = field(default=1.0, init=False, repr=False)
    _patrol_target: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64), init=False, repr=False)
    _steps_since_patrol: int = field(default=0, init=False, repr=False)
    _boundary_linger: int = field(default=0, init=False, repr=False)
    _last_planar: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64), init=False, repr=False)
    _base_profile: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _current_profile: str = field(default="default", init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:  # pragma: no cover - placeholder
        self._last_seen_blue = None
        if not self._base_profile:
            self._base_profile = {
                "escape_scale": self.escape_scale,
                "orbit_scale": self.orbit_scale,
                "tangential_bias": self.tangential_bias,
                "patrol_strength": self.patrol_strength,
                "patrol_radius": self.patrol_radius,
                "patrol_interval": float(self.patrol_interval),
                "center_preference": self.center_preference,
                "home_strength": self.home_strength,
                "air_intercept_gain": self.air_intercept_gain,
                "air_intercept_distance": self.air_intercept_distance,
                "air_predict_horizon": self.air_predict_horizon,
                "bait_push": self.bait_push,
                "bait_window": self.bait_window,
                "noise_close_scale": self.noise_close_scale,
                "noise_far_scale": self.noise_far_scale,
                "base_noise": self.base_noise,
                "momentum_weight": self.momentum_weight,
                "boundary_linger_limit": float(self.boundary_linger_limit),
            }
        self._randomize_profile()
        self._orbit_sign = 1.0 if self._rng.random() < 0.5 else -1.0
        self._patrol_target = self._sample_patrol_target()
        self._steps_since_patrol = 0
        self._boundary_linger = 0
        self._last_planar = np.zeros(2, dtype=np.float64)

    def _sample_patrol_target(self, radius_span: Optional[Tuple[float, float]] = None) -> np.ndarray:
        if radius_span is None:
            r_min = 0.55 * self.patrol_radius
            r_max = self.patrol_radius
        else:
            r_min, r_max = radius_span
        r_min = max(0.35, min(r_min, r_max - 1e-6))
        r_max = max(r_min + 1e-6, r_max)
        radius = self._rng.uniform(r_min, r_max)
        angle = self._rng.uniform(0.0, 2.0 * np.pi)
        return np.array([
            np.cos(angle) * radius,
            np.sin(angle) * radius,
        ], dtype=np.float64)

    def _randomize_profile(self) -> None:
        if not self._base_profile:
            return

        for key, value in self._base_profile.items():
            if key == "boundary_linger_limit":
                setattr(self, key, int(value))
            elif key == "patrol_interval":
                setattr(self, key, int(value))
            else:
                setattr(self, key, value)

        styles = ("balanced", "evasive", "bait", "weave", "burst")
        probs = np.array([0.25, 0.23, 0.18, 0.19, 0.15], dtype=np.float64)
        probs /= probs.sum()
        style = styles[int(self._rng.choice(len(styles), p=probs))]
        self._current_profile = style

        def jitter(scale: float) -> float:
            return float(self._rng.uniform(1.0 - scale, 1.0 + scale))

        # Baseline small randomization so no profile is identical between resets
        self.escape_scale *= jitter(0.05)
        self.orbit_scale *= jitter(0.08)
        self.tangential_bias *= jitter(0.08)
        self.center_preference *= jitter(0.07)
        self.patrol_strength *= jitter(0.08)
        self.base_noise *= jitter(0.12)
        self.noise_close_scale *= jitter(0.18)
        self.noise_far_scale *= jitter(0.18)

        if style == "evasive":
            self.escape_scale *= self._rng.uniform(1.12, 1.28)
            self.tangential_bias *= self._rng.uniform(1.18, 1.42)
            self.center_preference *= self._rng.uniform(0.72, 0.92)
            self.patrol_radius *= self._rng.uniform(0.95, 1.08)
            self.patrol_strength *= self._rng.uniform(1.05, 1.22)
            self.boundary_linger_limit = int(self.boundary_linger_limit * self._rng.uniform(0.9, 1.1))
        elif style == "bait":
            self.escape_scale *= self._rng.uniform(0.68, 0.88)
            self.center_preference *= self._rng.uniform(1.45, 1.75)
            self.bait_push *= self._rng.uniform(1.2, 1.55)
            self.bait_window *= self._rng.uniform(1.15, 1.45)
            self.air_intercept_gain *= self._rng.uniform(1.25, 1.55)
            self.air_intercept_distance *= self._rng.uniform(1.05, 1.3)
        elif style == "weave":
            self.orbit_scale *= self._rng.uniform(1.35, 1.6)
            self.tangential_bias *= self._rng.uniform(1.4, 1.75)
            self.momentum_weight *= self._rng.uniform(0.52, 0.7)
            self.base_noise *= self._rng.uniform(1.05, 1.25)
            self.noise_close_scale *= self._rng.uniform(1.2, 1.45)
            self.patrol_interval = int(max(60, self.patrol_interval * self._rng.uniform(0.9, 1.1)))
        elif style == "burst":
            self.escape_scale *= self._rng.uniform(1.02, 1.18)
            self.center_preference *= self._rng.uniform(1.1, 1.35)
            self.air_intercept_gain *= self._rng.uniform(1.4, 1.75)
            self.air_predict_horizon *= self._rng.uniform(1.05, 1.35)
            self.patrol_interval = int(max(70, self.patrol_interval * self._rng.uniform(0.6, 0.9)))
            self.boundary_linger_limit = int(self.boundary_linger_limit * self._rng.uniform(0.7, 0.95))
        else:  # balanced
            self.escape_scale *= jitter(0.03)
            self.center_preference *= jitter(0.05)
            self.patrol_radius *= jitter(0.04)

        self.boundary_linger_limit = max(60, int(self.boundary_linger_limit))

    def _boundary_adjustment(self, env: ChaseMujocoEnv, orange_pos: np.ndarray) -> np.ndarray:
        center_vec = orange_pos.copy()
        radius = np.linalg.norm(center_vec) + 1e-6
        margin = radius - (env.boundary_radius - self.boundary_threshold)
        if margin <= 0.0:
            return np.zeros(2, dtype=np.float64)
        inward = -center_vec / radius
        strength = min(1.0, margin / self.boundary_threshold) * self.boundary_push
        tangential = np.array([-inward[1], inward[0]], dtype=np.float64) * self.tangential_bias * self._orbit_sign
        return inward * strength + tangential

    def _home_adjustment(self, env: ChaseMujocoEnv, orange_pos: np.ndarray) -> np.ndarray:
        radius = np.linalg.norm(orange_pos) + 1e-6
        center_dir = -orange_pos / radius
        if radius > self.home_radius:
            excess = min(1.0, (radius - self.home_radius) / max(1e-6, env.boundary_radius - self.home_radius))
            return center_dir * self.home_strength * excess
        deficit = (self.home_radius - radius) / max(1e-6, self.home_radius)
        return -center_dir * 0.25 * deficit

    def _update_patrol_target(self, orange_pos: np.ndarray) -> np.ndarray:
        self._steps_since_patrol += 1
        to_target = self._patrol_target - orange_pos
        if np.linalg.norm(to_target) < 0.4 or self._steps_since_patrol >= self.patrol_interval:
            self._patrol_target = self._sample_patrol_target()
            self._steps_since_patrol = 0
            to_target = self._patrol_target - orange_pos
        return to_target

    def action(self, env: ChaseMujocoEnv) -> np.ndarray:
        blue_pos_full = env.data.xpos[env.blue_body]
        blue_pos = blue_pos_full[:2]
        blue_vel = env.data.cvel[env.blue_body, 3:].copy()
        orange_pos = env.data.xpos[env.orange_body, :2]
        orange_height = env.data.xpos[env.orange_body, 2]

        blue_airborne = blue_pos_full[2] > SPAWN_HEIGHT + 0.18
        if not blue_airborne and blue_pos_full[2] <= SPAWN_HEIGHT + self.visibility_eps:
            self._last_seen_blue = blue_pos.copy()

        if self._rng.random() < self.orbit_flip_prob:
            self._orbit_sign *= -1.0

        target_blue = self._last_seen_blue if self._last_seen_blue is not None else blue_pos
        rel = orange_pos - target_blue
        dist = np.linalg.norm(rel)
        dist = max(dist, 1e-6)
        away = rel / dist
        danger = dist < self.danger_distance

        tangential = np.array([-away[1], away[0]], dtype=np.float64) * self.orbit_scale * self._orbit_sign
        boundary = self._boundary_adjustment(env, orange_pos)
        home = self._home_adjustment(env, orange_pos)

        radius = np.linalg.norm(orange_pos) + 1e-6
        near_boundary = radius > env.boundary_radius - 0.65
        if near_boundary:
            self._boundary_linger += 1
        else:
            self._boundary_linger = max(0, self._boundary_linger - 1)

        if self._boundary_linger >= self.boundary_linger_limit:
            self._patrol_target = self._sample_patrol_target((1.1, max(1.5, self.home_radius - 0.2)))
            self._steps_since_patrol = 0
            self._boundary_linger = int(self.boundary_linger_limit * 0.45)
            if self._rng.random() < 0.6:
                self._orbit_sign *= -1.0

        escape = away * self.escape_scale
        dodge = np.zeros(2, dtype=np.float64)
        if danger:
            dodge_dir = np.array([-away[1], away[0]], dtype=np.float64)
            dodge = dodge_dir * self.danger_dodge_gain * self._orbit_sign
            escape = away * (self.escape_scale * 1.12)

        to_patrol = self._update_patrol_target(orange_pos)
        patrol = np.zeros(2, dtype=np.float64)
        patrol_norm = np.linalg.norm(to_patrol)
        if patrol_norm > 1e-6:
            patrol = (to_patrol / patrol_norm) * self.patrol_strength

        center_bias = np.zeros(2, dtype=np.float64)
        if radius > self.home_radius:
            scale = np.clip((radius - self.home_radius) / max(env.boundary_radius - self.home_radius, 1e-6), 0.0, 1.0)
            center_bias = (-orange_pos / radius) * self.center_preference * scale
        elif radius < self.home_radius * 0.6:
            scale = np.clip((self.home_radius * 0.6 - radius) / max(self.home_radius * 0.6, 1e-6), 0.0, 1.0)
            center_bias = (orange_pos / radius) * self.center_preference * 0.35 * scale

        air_intercept = np.zeros(2, dtype=np.float64)
        bait = np.zeros(2, dtype=np.float64)
        if blue_airborne:
            horizon = self.air_predict_horizon * self._rng.uniform(0.85, 1.15)
            predicted = blue_pos + blue_vel[:2] * horizon
            intercept_vec = predicted - orange_pos
            intercept_norm = np.linalg.norm(intercept_vec)
            if intercept_norm > 1e-6:
                gain = self.air_intercept_gain
                if dist < self.air_intercept_distance:
                    gain *= 1.35
                    escape *= 0.45
                    tangential *= 0.55
                    if danger:
                        dodge *= 0.45
                else:
                    escape *= 0.7
                air_intercept = (intercept_vec / intercept_norm) * gain
        else:
            if dist < self.bait_window:
                if dist <= self.bait_relax_threshold:
                    lateral = np.array([-away[1], away[0]], dtype=np.float64) * self._orbit_sign
                    bait = lateral * (self.bait_push * 0.7)
                    escape *= 0.6
                else:
                    bait = (-away) * self.bait_push
                    tangential *= 0.85

        noise_sigma = self.base_noise
        noise_sigma += self.noise_close_scale * (1.0 - np.clip(dist / self.danger_distance, 0.0, 1.0))
        noise_sigma += self.noise_far_scale * np.clip(radius - self.home_radius, 0.0, env.boundary_radius) / max(env.boundary_radius, 1.0)
        if np.linalg.norm(self._last_planar) < 0.15:
            noise_sigma += 0.08
        noise = self._rng.normal(scale=noise_sigma, size=2)

        new_planar = escape + tangential + boundary + home + dodge + patrol + center_bias + air_intercept + bait + noise
        planar = new_planar + self._last_planar * self.momentum_weight
        planar = np.clip(planar, -1.0, 1.0)
        self._last_planar = planar * 0.6 + new_planar * 0.4

        on_ground = orange_height <= SPAWN_HEIGHT + 0.05
        should_jump = on_ground and dist < self.jump_distance

        return np.array([planar[0], planar[1], 1.0 if should_jump else 0.0], dtype=np.float32)


class ChaseGymEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-agent Gymnasium wrapper that trains the blue chaser."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        frame_skip: int = 2,
        catch_radius: float = 0.5,
        max_steps: int = 1500,
        evader_policy: Optional[HeuristicEvader] = None,
        jump_penalty: float = 0.03,
        action_l2_penalty: float = 0.002,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._env = ChaseMujocoEnv(
            frame_skip=frame_skip,
            catch_radius=catch_radius,
            max_steps=max_steps,
            seed=seed,
        )
        self._evader = evader_policy or HeuristicEvader(seed=seed)
        self._last_info: dict[str, Any] = {}
        self._rng = np.random.default_rng(seed)
        self.jump_penalty = float(jump_penalty)
        self.action_l2_penalty = float(action_l2_penalty)

        obs_shape = self._env._get_obs().shape  # pylint: disable=protected-access
        if len(obs_shape) != 1:
            raise ValueError("Observation must be a flat vector.")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> None:  # gym API compatibility
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.seed(seed)
        obs = self._env.reset(seed=seed)
        obs = self._apply_visibility_mask(obs)
        self._evader.reset()
        self._last_info = {
            "distance": float(self._env._distance_xy()),  # pylint: disable=protected-access
            "caught": False,
            "timeout": False,
            "step_count": 0,
        }
        return obs, self._last_info.copy()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):  # type: ignore[override]
        blue_action = np.asarray(action, dtype=np.float32)
        if blue_action.shape != (3,):
            blue_action = blue_action.reshape(3)
        blue_action = np.clip(blue_action, -1.0, 1.0)

        orange_action = self._evader.action(self._env)
        stacked = np.concatenate([blue_action, orange_action], dtype=np.float32)

        obs, reward_dict, done, info = self._env.step(stacked)
        obs = self._apply_visibility_mask(obs)

        base_reward = float(reward_dict.get("blue", 0.0))
        jump_cost = self.jump_penalty if float(blue_action[2]) > 0.5 else 0.0
        move_cost = self.action_l2_penalty * float(np.dot(blue_action[:2], blue_action[:2]))
        reward = base_reward - jump_cost - move_cost
        terminated = bool(info.get("caught", False))
        truncated = bool(info.get("timeout", False))

        self._last_info = info
        info = info.copy()
        info["blue_reward"] = reward_dict.get("blue", 0.0)
        info["orange_reward"] = reward_dict.get("orange", 0.0)
        info["jump_cost"] = jump_cost
        info["action_cost"] = move_cost
        info["shaped_reward"] = reward

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._env.close()
        super().close()

    # ------------------------------------------------------------------
    def render(self):  # pragma: no cover - no headless rendering required
        raise NotImplementedError("Headless environment has no render().")

    # ------------------------------------------------------------------
    def _apply_visibility_mask(self, obs: np.ndarray) -> np.ndarray:
        masked = obs.copy()
        orange_height = self._env.data.xpos[self._env.orange_body, 2]
        blue_height = self._env.data.xpos[self._env.blue_body, 2]
        orange_air = orange_height > SPAWN_HEIGHT + VISIBILITY_EPS
        blue_air = blue_height > SPAWN_HEIGHT + VISIBILITY_EPS

        if orange_air or blue_air:
            masked[6:9] = 0.0
            masked[9:12] = 0.0
            masked[12:14] = 0.0
            masked[14] = 0.0
        return masked


__all__ = ["ChaseGymEnv", "HeuristicEvader"]
