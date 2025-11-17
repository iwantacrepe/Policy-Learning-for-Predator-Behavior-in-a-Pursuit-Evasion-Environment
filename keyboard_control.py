"""Manual two-agent control using the MuJoCo GLFW viewer."""

import sys

from typing import Any

import glfw
import mujoco
import numpy as np

from mujoco_env import ChaseMujocoEnv

MjvCamera = getattr(mujoco, "MjvCamera")
MjvOption = getattr(mujoco, "MjvOption")
MjvScene = getattr(mujoco, "MjvScene")
MjrContext = getattr(mujoco, "MjrContext")
MjrRect = getattr(mujoco, "MjrRect")
mjtFontScale = getattr(mujoco, "mjtFontScale")
mjtCatBit = getattr(mujoco, "mjtCatBit")
mjv_updateScene = getattr(mujoco, "mjv_updateScene")
mjr_render = getattr(mujoco, "mjr_render")


def resolve_actions(keys: dict[int, bool]) -> np.ndarray:
    blue_x = float(keys.get(glfw.KEY_D, False) - keys.get(glfw.KEY_A, False))
    blue_y = float(keys.get(glfw.KEY_W, False) - keys.get(glfw.KEY_S, False))

    blue_jump = float(
        keys.get(glfw.KEY_LEFT_SHIFT, False)
        or keys.get(glfw.KEY_RIGHT_SHIFT, False)
        or keys.get(glfw.KEY_Q, False)
    )

    orange_x = float(keys.get(glfw.KEY_L, False) - keys.get(glfw.KEY_J, False))
    orange_y = float(keys.get(glfw.KEY_I, False) - keys.get(glfw.KEY_K, False))
    orange_jump = float(keys.get(glfw.KEY_O, False) or keys.get(glfw.KEY_RIGHT_CONTROL, False))

    blue = np.array([blue_x, blue_y, blue_jump], dtype=np.float32)
    orange = np.array([orange_x, orange_y, orange_jump], dtype=np.float32)

    return np.clip(np.concatenate([blue, orange]), -1.0, 1.0).astype(np.float32)


def update_camera(camera: Any, keys: dict[int, bool]) -> None:
    azimuth_step = 1.2
    elevation_step = 1.0
    distance_step = 0.1

    if keys.get(glfw.KEY_LEFT, False):
        camera.azimuth += azimuth_step
    if keys.get(glfw.KEY_RIGHT, False):
        camera.azimuth -= azimuth_step
    if keys.get(glfw.KEY_UP, False):
        camera.elevation = max(-80.0, camera.elevation - elevation_step)
    if keys.get(glfw.KEY_DOWN, False):
        camera.elevation = min(-5.0, camera.elevation + elevation_step)

    if keys.get(glfw.KEY_EQUAL, False) or keys.get(glfw.KEY_KP_ADD, False):
        camera.distance = max(2.0, camera.distance - distance_step)
    if keys.get(glfw.KEY_MINUS, False) or keys.get(glfw.KEY_KP_SUBTRACT, False):
        camera.distance = min(14.0, camera.distance + distance_step)

    if keys.get(glfw.KEY_HOME, False):
        camera.azimuth = 135
        camera.elevation = -28
        camera.distance = 11.0
        camera.lookat[:] = (0.0, 0.0, 0.35)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("Failed to initialise GLFW.")

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(960, 720, "MuJoCo Chase Keyboard", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Unable to create GLFW window.")

    glfw.make_context_current(window)

    env = ChaseMujocoEnv()
    env.reset()

    print(
        "Controls: Blue=WASD + (Shift/Q) jump, Orange=IJKL + (O/RightCtrl) jump, "
        "Space=reset, Esc=quit, Arrow keys rotate camera, +/- zoom, Home reset view"
    )

    keys: dict[int, bool] = {}

    def on_key(window_ref, key: int, scancode: int, action: int, mods: int) -> None:
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window_ref, True)
            return
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            env.reset()
            return
        if action == glfw.PRESS:
            keys[key] = True
        elif action == glfw.RELEASE:
            keys[key] = False

    glfw.set_key_callback(window, on_key)

    camera = MjvCamera()
    option = MjvOption()
    scene = MjvScene(env.model, maxgeom=1500)
    context = MjrContext(env.model, mjtFontScale.mjFONTSCALE_150)

    camera.azimuth = 135
    camera.elevation = -28
    camera.distance = 11.0
    camera.lookat[:] = (0.0, 0.0, 0.35)

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()

            action = resolve_actions(keys)
            _, _, done, _ = env.step(action)

            update_camera(camera, keys)

            width, height = glfw.get_framebuffer_size(window)
            viewport = MjrRect(0, 0, width, height)

            mjv_updateScene(env.model, env.data, option, None, camera, mjtCatBit.mjCAT_ALL, scene)
            mjr_render(viewport, scene, context)

            glfw.swap_buffers(window)

            if done:
                env.reset()
    finally:
        env.close()
        glfw.terminate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - manual utility
        print(f"[keyboard_control] {exc}", file=sys.stderr)
        raise
