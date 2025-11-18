# Policy Learning for Predator Behavior in a Pursuit–Evasion Arena

A MuJoCo-based research sandbox for studying pursuit–evasion dynamics. Two stylised characters — a Tom-inspired chaser (blue) and a Jerry-inspired evader (orange) — interact in a circular arena. This repository contains:

- A custom MuJoCo physics environment with jump mechanics, vision constraints, and boundary enforcement.
- Heuristic controllers for both agents to support evaluation and curriculum design.
- A full PPO training pipeline built on Stable-Baselines3.
- Utilities for visualisation, manual control, and policy playback.

---

## Contents

- [Physics & Environment](#physics--environment)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Training](#training)
- [Running & Visualising Simulations](#running--visualising-simulations)
- [Debugging & Diagnostics](#debugging--diagnostics)
- [Algorithm & Implementation Notes](#algorithm--implementation-notes)
- [Customisation Hooks](#customisation-hooks)

---

## Physics & Environment

All environment logic lives in [`mujoco_env.py`](mujoco_env.py). Key mechanics:

- **Arena:** 11 m diameter grass arena with a soft boundary cylinder that gently reins agents in.
- **Bodies:** Each agent is a multi-geom rigid body. The chaser emulates a cat (Tom) and the evader a mouse (Jerry) with separate tail, ears, whiskers, etc. Cosmetic assets reside in [`assets/chase_env.xml`](assets/chase_env.xml).
- **Control Inputs:** Per agent `[vx, vy, jump]` commands in the local XY plane. Planar commands are clipped to `[-1, 1]` and mapped to forces via `BLUE_FORCE_SCALE` / `ORANGE_FORCE_SCALE`.
- **Jump Model:** When grounded and the jump channel exceeds 0.5, an impulse sets `qvel[z] = JUMP_VELOCITY`. Forward carry is injected via `JUMP_PLANAR_BOOST`, using the last valid takeoff direction.
- **Sensor Model:** The `DemoChaser` and `HeuristicChase` controllers track last seen evader positions, mimicking limited vision when the evader is airborne.
- **Rewards:** Negative distance shaping plus small jump/action penalties (configurable) encourage efficient pursuit.

The XML scene also includes decorative clouds, sun, distant mountains, and foliage outside the arena to provide spatial cues without affecting physics.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| [`assets/chase_env.xml`](assets/chase_env.xml) | Complete MuJoCo scene: arena geometry, decorative elements, and character meshes/materials. |
| [`mujoco_env.py`](mujoco_env.py) | Core MuJoCo environment wrapper with step/reset logic, boundary enforcement, and jump physics. |
| [`chase_gym_env.py`](chase_gym_env.py) | Gymnasium-compatible wrapper plus advanced `HeuristicEvader` controller used for training and evaluation. |
| [`train_headless.py`](train_headless.py) | Headless PPO training script (Stable-Baselines3) with checkpointing, evaluation, and VecNormalize support. |
| [`training_preview.py`](training_preview.py) | Passive viewer showing heuristic chaser vs. heuristic evader — great for quick physics inspection. |
| [`keyboard_control.py`](keyboard_control.py) | Manual control via keyboard in a GLFW window (WASD vs. IJKL). |
| [`play_trained_policy.py`](play_trained_policy.py) | Load a saved PPO policy and watch it pursue the heuristic evader. |
| [`play_ppo.py`](play_ppo.py) | Formerly `jump_demo.py`; now bundles the heuristic jump demo with PPO checkpoint playback. |
| [`models/`](models) | Example trained policy artefacts (e.g. `ppo_chase_final.zip`). |
| `ppo_chase_final.zip`, `vecnormalize.pkl` | Sample trained policy and normalization statistics. |

---

## Installation

1. **Python environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/macOS
   ```

2. **Dependencies**
   ```bash
   pip install --upgrade pip
   pip install mujoco numpy gymnasium stable-baselines3 torch glfw
   ```
   - Install [MuJoCo](https://mujoco.readthedocs.io/) binaries and ensure `MUJOCO_GL` is configured for your GPU/CPU backend.
   - For viewer utilities you need MuJoCo 2.3+ with `mujoco.viewer` support.

3. **Assets**
   - Ensure `MJLIB_PATH`/`MUJOCO_PY_MUJOCO_PATH` are set if MuJoCo is not installed system-wide.
   - The provided XML uses only bundled textures; no extra models are required.

---

## Training

Run PPO training headlessly:

```bash
python train_headless.py \
  --total-steps 9000000 \
  --num-envs 8 \
  --learning-rate 3e-4 \
  --anneal-lr \
  --normalize-obs --normalize-reward \
  --checkpoint-dir checkpoints
```

Key options:

- `--resume path/to/ppo.zip` resume from an existing checkpoint.
- `--tensorboard runs/ppo` enable TensorBoard logging (`tensorboard --logdir runs`).
- `--jump-penalty` & `--action-penalty` tweak auxiliary losses to shape behaviour.
- `--no-improve-evals 0` disables early stopping on plateau.

Checkpoints and normalization stats are written to the specified directory. The final model is saved as `ppo_chase_final.zip`.

---

## Running & Visualising Simulations

| Task | Command | Notes |
| --- | --- | --- |
| Quick heuristic preview | `python training_preview.py` | Heuristic chaser vs. heuristic evader in the MuJoCo viewer. |
| Jump mechanics demo / PPO rollout | `python play_ppo.py` | Combines the old `jump_demo` heuristics with PPO playback options. |
| Manual keyboard play | `python keyboard_control.py` | Requires GLFW; controls printed on startup. |
| Play a trained PPO model | `python play_trained_policy.py checkpoints/ppo_chase_final.zip --episodes 5` | Use `--deterministic` for evaluation. |

Each viewer exposes camera shortcuts (azimuth/elevation, zoom, reset). Close the window to terminate episodes cleanly.

---

## Debugging & Diagnostics

1. **Physics sanity checks**
   - `training_preview.py` is ideal for verifying textures, lighting, and collision geometry.
   - Toggle `JUMP_PLANAR_BOOST` or `BLUE_FORCE_SCALE` in `mujoco_env.py` to inspect the impact of constants.

2. **Training troubleshooting**
   - Enable TensorBoard via `--tensorboard runs/ppo` to monitor policy loss, value loss, entropy, and KL.
   - Use `--eval-frequency` and `--no-improve-evals` for early stopping if the chaser plateaus.
   - Check `checkpoints/` for saved vector-normalization (`vecnormalize.pkl`) if rewards explode or vanish.

3. **Action/Observation debugging**
   - Run `play_trained_policy.py` with `--sleep-scale 0.2` (slow motion) to inspect decisions frame by frame.
   - Modify `DEBUG` prints in `mujoco_env.py` by adding instrumentation inside `_apply_action` to capture forces.

4. **GLFW viewer issues**
   - Ensure graphics drivers support OpenGL 3.3+. On macOS/Linux, export `MUJOCO_GL=glfw`. On WSL, use X-Server.
   - If the viewer fails, try `pip install mujoco==3.1.2` (or the version matching your MuJoCo binaries).

---

## Algorithm & Implementation Notes

- **RL Algorithm:** Proximal Policy Optimization (PPO) from Stable-Baselines3 with Tanh activations and two 256-unit hidden layers.
- **Observation Space:** Concatenated positions, velocities, relative vectors, and distance metrics from both agents (see `_get_obs`).
- **Action Space:** Continuous 3D vector per agent; PPO only predicts the chaser's actions while the evader uses heuristics.
- **Reward Shaping:** Distance-based shaping, catch bonus, and penalties for excessive jumping/acceleration.
- **Environment Vectorisation:** Parallel environments via `SubprocVecEnv`; optional `VecNormalize` for stabilizing gradients.

---

## Customisation Hooks

- **Physics Tweaks:** Adjust constants near the top of `mujoco_env.py` (`JUMP_VELOCITY`, `BLUE_FORCE_SCALE`, etc.) or replace assets in `assets/chase_env.xml`.
- **Heuristic Evader Profiles:** `HeuristicEvader` in `chase_gym_env.py` exposes parameters for orbiting, dodging, and noise; call `reset()` to randomise behaviours between episodes.
- **Chaser/Evader Demos:** `jump_demo.py` includes `AdaptiveChaser`/`AdaptiveEvader` classes; tweak their dataclass fields to prototype new strategies quickly.
- **Training Hyperparameters:** All PPO knobs are exposed via CLI in `train_headless.py`. Edit defaults or supply arguments per experiment.

---

## Citation / Attribution

If you build upon this project, please cite the repository and acknowledge Stable-Baselines3, MuJoCo, and Gymnasium.

---

Happy chasing! 🐱‍👓🏃‍♂️💨
