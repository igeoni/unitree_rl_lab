# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--scene",
    type=str,
    default=None,
    choices=["warehouse", "warehouse_multiple_shelves", "full_warehouse"],
    help="Replace terrain with a Nucleus warehouse USD scene.",
)
parser.add_argument(
    "--spawn-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override robot spawn position (e.g. --spawn-pos 2.0 0.0 1.05).",
)
parser.add_argument(
    "--fixed_vel",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Override velocity command with fixed values every step (e.g. --fixed_vel 0.0 0.0 1.0).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    def get_published_pretrained_checkpoint(*_):
        return None
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _setup_keyboard_teleop(device: str, num_envs: int):
    """Set up carb keyboard teleop (polling-based). Returns teleop state dict or None if unavailable."""
    try:
        import carb
        import omni.appwindow

        _input = carb.input.acquire_input_interface()
        _keyboard = omni.appwindow.get_default_app_window().get_keyboard()

        VX, VY, WZ = 1.0, 0.5, 0.6
        key_map = [
            (carb.input.KeyboardInput.W, torch.tensor([VX, 0.0, 0.0])),
            (carb.input.KeyboardInput.S, torch.tensor([-VX, 0.0, 0.0])),
            (carb.input.KeyboardInput.A, torch.tensor([0.0, VY, 0.0])),
            (carb.input.KeyboardInput.D, torch.tensor([0.0, -VY, 0.0])),
            (carb.input.KeyboardInput.Z, torch.tensor([0.0, 0.0, WZ])),
            (carb.input.KeyboardInput.C, torch.tensor([0.0, 0.0, -WZ])),
        ]
        vel = torch.zeros(num_envs, 3, device=device)

        print("[INFO] Keyboard teleop active: W/S=forward/back, A/D=left/right, Q/E=rotate. Focus the Isaac Sim window.")
        return {"vel": vel, "_input": _input, "_keyboard": _keyboard, "key_map": key_map, "device": device}
    except Exception as e:
        print(f"[WARNING] Keyboard teleop unavailable: {e}")
        return None


def _apply_teleop_command(env, teleop: dict):
    """Poll keyboard state, update velocity command, and override env command."""
    try:
        _input = teleop["_input"]
        _keyboard = teleop["_keyboard"]
        vel = teleop["vel"]
        device = teleop["device"]

        new_vel = torch.zeros(3)
        for key, delta in teleop["key_map"]:
            if _input.get_keyboard_value(_keyboard, key):
                new_vel += delta
        vel[:] = new_vel.to(device)

        cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
        cmd_term.vel_command_b[:] = vel
    except Exception:
        pass


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    if args_cli.scene is not None:
        from isaaclab.terrains import TerrainImporterCfg
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
        _scene_map = {
            "warehouse": "warehouse.usd",
            "warehouse_multiple_shelves": "warehouse_multiple_shelves.usd",
            "full_warehouse": "full_warehouse.usd",
        }
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/{_scene_map[args_cli.scene]}",
            collision_group=-1,
        )
        if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
            if hasattr(env_cfg.curriculum, "terrain_levels"):
                env_cfg.curriculum.terrain_levels = None

    if args_cli.spawn_pos is not None:
        x, y, z = args_cli.spawn_pos
        env_cfg.scene.robot.init_state.pos = (x, y, z)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # foot contact publisher (requires ROS2/rclpy)
    try:
        from contact_publisher import ContactPublisher
        contact_pub = ContactPublisher(env)
    except Exception as e:
        print(f"[WARNING]: ContactPublisher unavailable (ROS2 not loaded): {e}")
        contact_pub = None

    # joint state publisher (requires ROS2/rclpy)
    try:
        from joint_state_publisher import JointStatePublisher
        joint_pub = JointStatePublisher(env)
    except Exception as e:
        print(f"[WARNING]: JointStatePublisher unavailable (ROS2 not loaded): {e}")
        joint_pub = None

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        _deprecated_model_fields = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std",
                                    "actor_obs_normalization", "critic_obs_normalization",
                                    "actor_hidden_dims", "critic_hidden_dims"}
        _agent_dict = agent_cfg.to_dict()
        for _key in ("actor", "critic", "policy"):
            if _key in _agent_dict and isinstance(_agent_dict[_key], dict):
                for _field in _deprecated_model_fields:
                    _agent_dict[_key].pop(_field, None)
        runner = OnPolicyRunner(env, _agent_dict, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    if hasattr(runner.alg, "get_policy"):
        # rsl_rl >= 5.0
        policy_nn = runner.alg.get_policy()
    elif hasattr(runner.alg, "policy"):
        # version 2.3
        policy_nn = runner.alg.policy
    else:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    except Exception as e:
        print(f"[WARNING]: Policy export skipped (incompatible with new rsl_rl API): {e}")

    dt = env.unwrapped.step_dt

    # keyboard teleop setup
    teleop = _setup_keyboard_teleop(env.unwrapped.device, env.num_envs)

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # fixed velocity command override
    fixed_vel = None
    if args_cli.fixed_vel is not None:
        fixed_vel = torch.tensor(args_cli.fixed_vel, device=env.unwrapped.device).unsqueeze(0).expand(env.num_envs, -1)
        print(f"[INFO] Fixed velocity command: vx={args_cli.fixed_vel[0]}, vy={args_cli.fixed_vel[1]}, wz={args_cli.fixed_vel[2]}")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        # override velocity commands with fixed value or keyboard input
        if fixed_vel is not None:
            cmd_term = env.unwrapped.command_manager._terms["base_velocity"]
            cmd_term.vel_command_b[:] = fixed_vel
        elif teleop is not None:
            _apply_teleop_command(env, teleop)

        if contact_pub is not None:
            contact_pub.update()
        if joint_pub is not None:
            joint_pub.update()

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if contact_pub is not None:
        contact_pub.close()
    if joint_pub is not None:
        joint_pub.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
