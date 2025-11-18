"""Interactive viewer showcasing the training-time chase dynamics."""

from __future__ import annotations

import time

import numpy as np

from chase_gym_env import HeuristicEvader
from mujoco_env import ChaseMujocoEnv, SPAWN_HEIGHT
    
try:
    import mujoco.viewer as viewer
except ImportError as exc:  # pragma: no cover - viewer dependency
    raise ImportError(
        "mujoco.viewer is required. Install MuJoCo with viewer support before running this demo."
    ) from exc

GROUND_EPS = 0.06


class DemoChaser:
    """Mimics a sight-limited chaser similar to the RL observation constraints."""

    def __init__(self, initial_target: np.ndarray) -> None:
        self._last_sighted = initial_target.copy()

    def action(self, env: ChaseMujocoEnv) -> np.ndarray:
        blue_pos = env.data.xpos[env.blue_body, :2]
        orange_pos_full = env.data.xpos[env.orange_body]
        orange_pos = orange_pos_full[:2]

        orange_on_ground = orange_pos_full[2] <= SPAWN_HEIGHT + GROUND_EPS
        if orange_on_ground:
            self._last_sighted = orange_pos.copy()

        target = self._last_sighted
        rel = target - blue_pos
        dist = np.linalg.norm(rel) + 1e-6
        direction = rel / dist

        chase_speed = 1.45 if orange_on_ground else 0.95
        planar = direction * chase_speed

        should_jump = orange_on_ground and dist < 1.8
        return np.array([planar[0], planar[1], 1.0 if should_jump else 0.0], dtype=np.float32)


def main() -> None:
    env = ChaseMujocoEnv()
    env.reset()
    chaser = DemoChaser(env.data.xpos[env.orange_body, :2])
    evader = HeuristicEvader()

    print(
        "Launching training preview — blue uses a heuristic chaser with limited vision, "
        "orange follows the same heuristic evasion logic used during PPO training."
    )
    print("Close the window to end the demo.")

    try:
        with viewer.launch_passive(env.model, env.data) as handle:
            handle.cam.azimuth = 140
            handle.cam.elevation = -30
            handle.cam.distance = 12.0
            handle.cam.lookat[:] = (0.0, 0.0, 0.35)

            while handle.is_running():
                blue_action = chaser.action(env)
                orange_action = evader.action(env)
                action = np.concatenate([blue_action, orange_action], dtype=np.float32)

                _, _, done, info = env.step(action)
                handle.sync()

                time.sleep(env.model.opt.timestep * env.frame_skip)

                if done:
                    print(
                        f"Episode finished after {env.step_count} steps | "
                        f"caught={info['caught']} timeout={info['timeout']}"
                    )
                    env.reset()
                    chaser = DemoChaser(env.data.xpos[env.orange_body, :2])
                    evader.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()