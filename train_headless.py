"""Headless PPO training entry-point for the chase environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch.nn as nn

try:
    from stable_baselines3 import PPO  # type: ignore[import]
    from stable_baselines3.common.callbacks import (  # type: ignore[import]
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
        StopTrainingOnNoModelImprovement,
    )
    from stable_baselines3.common.monitor import Monitor  # type: ignore[import]
    from stable_baselines3.common.vec_env import (  # type: ignore[import]
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
        VecEnv,
    )
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "stable-baselines3 is required. Install it with 'pip install stable-baselines3'."
    ) from exc

from chase_gym_env import ChaseGymEnv


def linear_schedule(initial_lr: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr

    return schedule


def make_env(seed: int, env_kwargs: Dict[str, Any]) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env = ChaseGymEnv(seed=seed, **env_kwargs)
        monitored = Monitor(env)
        monitored.reset(seed=seed)
        return monitored

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless PPO trainer for the chase game")
    parser.add_argument("--total-steps", type=int, default=9_000_000, help="Total PPO environment steps")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Torch device (auto/cpu/cuda)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate")
    parser.add_argument("--anneal-lr", action="store_true", help="Linearly anneal learning rate to zero")
    parser.add_argument("--n-steps", type=int, default=1536, help="Rollout horizon per environment")
    parser.add_argument("--batch-size", type=int, default=3072, help="Minibatch size")
    parser.add_argument("--clip-range", type=float, default=0.18, help="PPO clip range")
    parser.add_argument("--clip-range-vf", type=float, default=0.25, help="Value function clip range")
    parser.add_argument("--ent-coef", type=float, default=0.008, help="Entropy bonus weight")
    parser.add_argument("--vf-coef", type=float, default=0.55, help="Value loss weight")
    parser.add_argument("--gamma", type=float, default=0.997, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.96, help="GAE lambda")
    parser.add_argument("--target-kl", type=float, default=0.015, help="Early stop if KL exceeds target")
    parser.add_argument("--max-grad-norm", type=float, default=0.7, help="Gradient clipping value")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Checkpoint output directory")
    parser.add_argument("--best-checkpoint-dir", type=Path, default=None, help="Directory for best-eval checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval-frequency", type=int, default=200_000, help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=12, help="Episodes per evaluation run")
    parser.add_argument("--tensorboard", type=Path, default=None, help="Optional TensorBoard log dir")
    parser.add_argument("--policy-hidden", type=int, nargs=2, default=(256, 256), help="MLP hidden layer sizes")
    parser.add_argument("--resume", type=Path, default=None, help="Optional PPO model to resume from")
    parser.add_argument("--normalize-obs", action="store_true", help="Enable VecNormalize observation scaling")
    parser.add_argument("--normalize-reward", action="store_true", help="Enable reward normalization")
    parser.add_argument("--jump-penalty", type=float, default=0.03, help="Penalty applied when blue jumps")
    parser.add_argument("--action-penalty", type=float, default=0.002, help="Quadratic penalty on planar acceleration")
    parser.add_argument("--no-improve-evals", type=int, default=15, help="Stop after N evals without improvement (0 disables)")
    return parser.parse_args()


def build_vec_env(num_envs: int, seed: int, env_kwargs: Dict[str, Any]) -> VecEnv:
    seeds = np.arange(seed, seed + num_envs, dtype=np.int64)
    env_fns = [make_env(int(s), env_kwargs) for s in seeds]
    if num_envs > 1:
        return SubprocVecEnv(env_fns)  # type: ignore[arg-type]
    return DummyVecEnv(env_fns)  # type: ignore[arg-type]


def prepare_eval_env(base_env: VecEnv, seed: int, env_kwargs: Dict[str, Any], normalize_obs: bool, normalize_reward: bool) -> VecEnv:
    eval_env = build_vec_env(1, seed, env_kwargs)
    if isinstance(base_env, VecNormalize):
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        eval_env.obs_rms = base_env.obs_rms.copy()
        eval_env.ret_rms = base_env.ret_rms.copy()
    return eval_env


def maybe_load_vecnormalize(vec_env: VecEnv, path: Path) -> VecEnv:
    if isinstance(vec_env, VecNormalize) and path.exists():
        return VecNormalize.load(str(path), vec_env)
    return vec_env


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir = args.best_checkpoint_dir or args.checkpoint_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "jump_penalty": args.jump_penalty,
        "action_l2_penalty": args.action_penalty,
    }

    vec_env: VecEnv = build_vec_env(args.num_envs, args.seed, env_kwargs)
    vec_norm_path = args.checkpoint_dir / "vecnormalize.pkl"
    if args.normalize_obs or args.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=args.normalize_obs,
            norm_reward=args.normalize_reward,
            clip_obs=10.0,
            clip_reward=10.0,
        )
        vec_env = maybe_load_vecnormalize(vec_env, vec_norm_path)

    eval_env = prepare_eval_env(vec_env, args.seed + 10_000, env_kwargs, args.normalize_obs, args.normalize_reward)

    policy_kwargs = {
        "net_arch": list(args.policy_hidden),
        "activation_fn": nn.Tanh,
        "ortho_init": False,
    }

    learning_rate: Any = linear_schedule(args.learning_rate) if args.anneal_lr else args.learning_rate

    if args.resume is not None and args.resume.exists():
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=args.device,
        )
        print(f"Resumed PPO model from {args.resume}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            tensorboard_log=str(args.tensorboard) if args.tensorboard else None,
            device=args.device,
            policy_kwargs=policy_kwargs,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.num_envs),
        save_path=str(args.checkpoint_dir),
        name_prefix="ppo_chase",
    )

    callbacks: list[BaseCallback] = [checkpoint_callback]

    stop_callback: Optional[StopTrainingOnNoModelImprovement] = None
    if args.no_improve_evals > 0:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.no_improve_evals,
            min_evals=5,
            verbose=1,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(1, args.eval_frequency // args.num_envs),
        deterministic=True,
        callback_on_new_best=stop_callback,
        warn=False,
    )
    callbacks.append(eval_callback)

    model.learn(total_timesteps=args.total_steps, callback=callbacks)

    final_path = args.checkpoint_dir / "ppo_chase_final"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}.zip")

    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(vec_norm_path))

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()