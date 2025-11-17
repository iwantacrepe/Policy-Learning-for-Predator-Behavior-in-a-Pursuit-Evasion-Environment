"""Autonomous chase demo to stress-test jump mechanics."""

from __future__ import annotations

import time
import mujoco.viewer as viewer
import numpy as np

from chase_gym_env import HeuristicEvader
from mujoco_env import ChaseMujocoEnv, SPAWN_HEIGHT

GROUND_EPS = 0.06


class HeuristicChase:
    """Simple controller that loses track of orange when it is airborne."""

    def __init__(self, initial_target: np.ndarray) -> None:
        self.last_sighted = initial_target.copy()

    def reset(self, env: ChaseMujocoEnv) -> None:
        self.last_sighted = env.data.xpos[env.orange_body, :2].copy()

    def step(self, env: ChaseMujocoEnv) -> np.ndarray:
        blue_pos = env.data.xpos[env.blue_body, :2]
        orange_pos = env.data.xpos[env.orange_body, :2]
        orange_height = env.data.xpos[env.orange_body, 2]

        on_ground = orange_height <= SPAWN_HEIGHT + GROUND_EPS
        if on_ground:
            self.last_sighted = orange_pos.copy()

        rel = self.last_sighted - blue_pos
        dist = np.linalg.norm(rel) + 1e-6
        direction = rel / dist

        chase_speed = 1.35 if on_ground else 0.8
        return np.array([direction[0] * chase_speed, direction[1] * chase_speed, 0.0], dtype=np.float32)


def main() -> None:
    env = ChaseMujocoEnv()
    env.reset()

    controller = HeuristicChase(env.data.xpos[env.orange_body, :2])
    orange_policy = HeuristicEvader()
    orange_policy.reset()

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
                orange_action = orange_policy.action(env)
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
                    orange_policy.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
