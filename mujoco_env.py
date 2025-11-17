"""Custom MuJoCo environment for a two-agent chase game."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np


ARENA_RADIUS = 5.5
SPAWN_HEIGHT = 0.35
BODY_RADIUS = 0.35
BLUE_FORCE_SCALE = 220.0
ORANGE_FORCE_SCALE = 260.0
JUMP_VELOCITY = 10.0
JUMP_COOLDOWN_STEPS = 30
VELOCITY_SCALE = 12.0

MjModel = getattr(mujoco, "MjModel")
MjData = getattr(mujoco, "MjData")
Renderer = getattr(mujoco, "Renderer")
mj_resetData = getattr(mujoco, "mj_resetData")
mj_forward = getattr(mujoco, "mj_forward")
mj_step = getattr(mujoco, "mj_step")


class ChaseMujocoEnv:
    """Two-agent chase environment backed by MuJoCo."""

    def __init__(
        self,
        xml: Union[str, Path] = "assets/chase_env.xml",
        frame_skip: int = 2,
        catch_radius: float = 0.5,
        max_steps: int = 1500,
        blue_force_scale: float = BLUE_FORCE_SCALE,
        orange_force_scale: float = ORANGE_FORCE_SCALE,
        catch_height_tolerance: float = 1.1,
        contact_margin: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        xml_path = Path(xml)
        if not xml_path.is_absolute():
            xml_path = base_dir / xml_path
        xml_path = xml_path.resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = MjModel.from_xml_path(str(xml_path))
        self.data = MjData(self.model)

        self.frame_skip = max(1, frame_skip)
        self.catch_radius = catch_radius
        self.max_steps = max_steps
        self.catch_height_tolerance = max(0.0, catch_height_tolerance)
        self.contact_margin = max(0.0, contact_margin)
        self.boundary_radius = ARENA_RADIUS - BODY_RADIUS

        self.blue_body = self.model.body("blue").id
        self.orange_body = self.model.body("orange").id
        self.blue_joint = self.model.joint("blue_free").id
        self.orange_joint = self.model.joint("orange_free").id

        self.blue_qpos_adr = self.model.jnt_qposadr[self.blue_joint]
        self.orange_qpos_adr = self.model.jnt_qposadr[self.orange_joint]
        self.blue_qvel_adr = self.model.jnt_dofadr[self.blue_joint]
        self.orange_qvel_adr = self.model.jnt_dofadr[self.orange_joint]

        self.random = np.random.default_rng(seed)
        self.step_count = 0

        self._renderer: Optional[mujoco.Renderer] = None
        self._render_size: Optional[Tuple[int, int]] = None
        self._body_qvel_index = {
            self.blue_body: self.blue_qvel_adr,
            self.orange_body: self.orange_qvel_adr,
        }
        self._jump_cooldowns = {self.blue_body: 0, self.orange_body: 0}
        self._force_scales = {
            self.blue_body: float(blue_force_scale),
            self.orange_body: float(orange_force_scale),
        }
        self._blue_geom_ids = tuple(
            idx for idx in range(self.model.ngeom) if self.model.geom_bodyid[idx] == self.blue_body
        )
        self._orange_geom_ids = tuple(
            idx for idx in range(self.model.ngeom) if self.model.geom_bodyid[idx] == self.orange_body
        )
        zero_vector = np.zeros(2, dtype=np.float64)
        self._last_planar_input = {
            self.blue_body: zero_vector.copy(),
            self.orange_body: zero_vector.copy(),
        }

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.random = np.random.default_rng(seed)

        mj_resetData(self.model, self.data)

        self._randomize_body(self.blue_qpos_adr, self.blue_qvel_adr)
        self._randomize_body(self.orange_qpos_adr, self.orange_qvel_adr)

        mj_forward(self.model, self.data)

        self.step_count = 0
        for body_id in (self.blue_body, self.orange_body):
            self._last_planar_input[body_id][:] = 0.0
        return self._get_obs()

    # ------------------------------------------------------------------
    def step(
        self,
        action: Union[Sequence[float], np.ndarray, Dict[str, Sequence[float]]],
        orange_action: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], bool, Dict[str, float]]:
        blue_cmd, orange_cmd = self._resolve_actions(action, orange_action)

        for _ in range(self.frame_skip):
            self._apply_action(self.blue_body, blue_cmd)
            self._apply_action(self.orange_body, orange_cmd)
            mj_step(self.model, self.data)
            self._enforce_bounds()

        self.step_count += 1

        obs = self._get_obs()
        reward, caught = self._compute_reward()

        done = bool(caught or self.step_count >= self.max_steps)
        info = {
            "distance": float(self._distance_xy()),
            "caught": bool(caught),
            "timeout": self.step_count >= self.max_steps,
            "step_count": self.step_count,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera: Optional[Any] = None,
    ) -> np.ndarray:
        if self._renderer is None or self._render_size != (width, height):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = Renderer(self.model, width, height)
            self._render_size = (width, height)

        renderer = self._renderer
        if renderer is None:
            raise RuntimeError("Renderer failed to initialize.")

        renderer.update_scene(self.data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8)

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._render_size = None

    # ------------------------------------------------------------------
    def _resolve_actions(
        self,
        action: Union[Sequence[float], np.ndarray, Dict[str, Sequence[float]]],
        orange_action: Optional[Sequence[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(action, dict):
            blue_vec = np.asarray(action.get("blue", (0.0, 0.0, 0.0)), dtype=np.float64)
            orange_vec = np.asarray(action.get("orange", (0.0, 0.0, 0.0)), dtype=np.float64)
            return blue_vec, orange_vec

        if orange_action is not None:
            blue_vec = np.asarray(action, dtype=np.float64)
            orange_vec = np.asarray(orange_action, dtype=np.float64)
            return blue_vec, orange_vec

        arr = np.asarray(action, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] == 2:
            blue_vec = arr[0]
            orange_vec = arr[1]
        else:
            flat = arr.reshape(-1)
            if flat.size == 4:
                blue_vec = flat[:2]
                orange_vec = flat[2:]
            elif flat.size >= 6:
                blue_vec = flat[:3]
                orange_vec = flat[3:6]
            else:
                raise ValueError("Action must provide controls for both agents.")
        return blue_vec, orange_vec

    # ------------------------------------------------------------------
    def _apply_action(self, body_id: int, action: np.ndarray) -> None:
        if action.size < 2:
            raise ValueError("Each action must supply at least two components (x, y).")

        force_scale = self._force_scales.get(body_id, BLUE_FORCE_SCALE)
        height = self.data.xpos[body_id, 2]
        on_ground = height <= SPAWN_HEIGHT + 0.045

        planar_input = np.clip(np.asarray(action[:2], dtype=np.float64), -1.0, 1.0)
        stored_input = self._last_planar_input[body_id]
        jump = float(action[2]) if action.size >= 3 else 0.0

        if on_ground:
            planar_norm = np.linalg.norm(planar_input)
            if planar_norm < 1e-3 and jump > 0.5:
                command = stored_input
            else:
                command = planar_input
                self._last_planar_input[body_id] = planar_input.copy()
        else:
            command = stored_input

        planar_force = command * force_scale

        self.data.xfrc_applied[body_id] = 0.0
        if on_ground:
            self.data.xfrc_applied[body_id, 0] = planar_force[0]
            self.data.xfrc_applied[body_id, 1] = planar_force[1]

        if self._jump_cooldowns[body_id] > 0:
            self._jump_cooldowns[body_id] -= 1

        ready = self._jump_cooldowns[body_id] == 0
        if jump > 0.5 and on_ground and ready:
            qvel_adr = self._body_qvel_index[body_id]
            self.data.qvel[qvel_adr + 2] = JUMP_VELOCITY
            self._jump_cooldowns[body_id] = JUMP_COOLDOWN_STEPS

    # ------------------------------------------------------------------
    def _randomize_body(self, qpos_adr: int, qvel_adr: int) -> None:
        radius = self.random.uniform(0.0, self.boundary_radius)
        angle = self.random.uniform(0.0, 2.0 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        self.data.qpos[qpos_adr: qpos_adr + 3] = (x, y, SPAWN_HEIGHT)
        self.data.qpos[qpos_adr + 3: qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[qvel_adr: qvel_adr + 3] = 0.0
        self.data.qvel[qvel_adr + 3: qvel_adr + 6] = 0.0

    # ------------------------------------------------------------------
    def _enforce_bounds(self) -> None:
        for qpos_adr, qvel_adr in (
            (self.blue_qpos_adr, self.blue_qvel_adr),
            (self.orange_qpos_adr, self.orange_qvel_adr),
        ):
            pos = self.data.qpos[qpos_adr: qpos_adr + 3]
            vel = self.data.qvel[qvel_adr: qvel_adr + 3]
            dist = np.linalg.norm(pos[:2])
            if dist > self.boundary_radius:
                pos[:2] = pos[:2] / dist * self.boundary_radius
                vel[:2] = 0.0
            if pos[2] < SPAWN_HEIGHT:
                pos[2] = SPAWN_HEIGHT
                vel[2] = max(0.0, vel[2])
            self.data.qpos[qpos_adr + 3: qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        bx = self.data.xpos[self.blue_body].copy()
        ox = self.data.xpos[self.orange_body].copy()
        bv = self.data.cvel[self.blue_body, 3:].copy()
        ov = self.data.cvel[self.orange_body, 3:].copy()

        rel = ox - bx
        rel_xy = rel[:2]
        dist = np.linalg.norm(rel_xy)
        direction = rel_xy / (dist + 1e-6)

        obs = np.concatenate([
            bx / ARENA_RADIUS,
            bv / VELOCITY_SCALE,
            ox / ARENA_RADIUS,
            ov / VELOCITY_SCALE,
            direction,
            np.array([dist / ARENA_RADIUS], dtype=np.float64),
        ])

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def _distance_xy(self) -> float:
        bx = self.data.xpos[self.blue_body, :2]
        ox = self.data.xpos[self.orange_body, :2]
        return float(np.linalg.norm(bx - ox))

    # ------------------------------------------------------------------
    def _bodies_in_contact(self) -> bool:
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 < 0 or geom2 < 0:
                continue
            if (
                (geom1 in self._blue_geom_ids and geom2 in self._orange_geom_ids)
                or (geom2 in self._blue_geom_ids and geom1 in self._orange_geom_ids)
            ):
                if contact.dist <= self.contact_margin:
                    return True
        return False

    # ------------------------------------------------------------------
    def _compute_reward(self) -> Tuple[Dict[str, float], bool]:
        blue_pos = self.data.xpos[self.blue_body]
        orange_pos = self.data.xpos[self.orange_body]
        rel = blue_pos - orange_pos
        dist_xy = np.linalg.norm(rel[:2])
        dist = float(dist_xy)

        reward_blue = -0.05 * dist
        reward_orange = 0.05 * dist

        vertical_gap = abs(rel[2])
        dist_3d = np.linalg.norm(rel)
        caught = dist < self.catch_radius and (
            dist_3d < self.catch_radius * 1.3 or vertical_gap < self.catch_height_tolerance
        )
        caught = caught or self._bodies_in_contact()

        if caught:
            reward_blue += 5.0
            reward_orange -= 5.0

        reward = {"blue": float(reward_blue), "orange": float(reward_orange)}
        return reward, bool(caught)
