"""Visualise a trained PPO policy chasing the heuristic evader."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from chase_gym_env import HeuristicEvader
from mujoco_env import ChaseMujocoEnv

try:
    from stable_baselines3 import PPO  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "stable-baselines3 is required. Install it with 'pip install stable-baselines3'."
    ) from exc

try:
    import mujoco.viewer as viewer
except ImportError as exc:  # pragma: no cover - viewer is optional but recommended
    raise ImportError(
        "mujoco.viewer is required to visualise the trained policy. "
        "Install MuJoCo bindings with viewer support."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play back a trained chaser policy")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the trained PPO .zip file (e.g. checkpoints/ppo_chase_final.zip)",
    )
    parser.add_argument("--device", default="auto", help="Torch device to load the model on")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--sleep-scale", type=float, default=1.0, help="Time dilation factor (1.0=real time)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
    return parser.parse_args()


def run_episode(
    model: PPO,
    env: ChaseMujocoEnv,
    evader: HeuristicEvader,
    deterministic: bool,
    sleep_scale: float,
) -> None:
    obs = env.reset()
    step_count = 0
    caught = False

    with viewer.launch_passive(env.model, env.data) as handle:
        handle.cam.azimuth = 140
        handle.cam.elevation = -30
        handle.cam.distance = 12.0
        handle.cam.lookat[:] = (0.0, 0.0, 0.3)

        while handle.is_running():
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.clip(action, -1.0, 1.0)
            orange_action = evader.action(env)
            stacked = np.concatenate([action, orange_action], dtype=np.float32)

            obs, reward_dict, done, info = env.step(stacked)
            handle.sync()

            timestep = env.model.opt.timestep * env.frame_skip
            time.sleep(max(0.0, timestep * sleep_scale))

            step_count += 1
            caught = bool(info.get("caught", False))
            if done:
                print(
                    f"Episode finished after {step_count} steps | "
                    f"caught={caught} | distance={info.get('distance', 0.0):.3f}"
                )
                break


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    env = ChaseMujocoEnv()
    evader = HeuristicEvader()
    model = PPO.load(str(args.model), device=args.device)

    try:
        for episode in range(args.episodes):
            print(f"Starting episode {episode + 1}/{args.episodes}")
            run_episode(model, env, evader, args.deterministic, args.sleep_scale)
    finally:
        env.close()


if __name__ == "__main__":
    main()
