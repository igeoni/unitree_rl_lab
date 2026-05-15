"""ROS2 joint state publisher for IsaacLab environments.

Usage in play.py:
    from joint_state_publisher import JointStatePublisher

    pub = JointStatePublisher(env)       # after gym.make()
    while simulation_app.is_running():
        obs, _, dones, _ = env.step(actions)
        pub.update()                     # in the simulation loop
    pub.close()
"""

import json
import os
import subprocess

_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_bridge.py")
_ROS2_PYTHON = "/usr/bin/python3"


class JointStatePublisher:
    """Reads joint state from IsaacLab articulation and publishes via ROS2.

    Spawns a Python 3.10 subprocess (joint_bridge.py) that handles rclpy,
    and communicates with it via stdin JSON lines.

    Topic: /joint_states  (sensor_msgs/JointState)
    Frame: env_0 only (play mode uses num_envs=1)
    """

    def __init__(self, env, topic: str = "/joint_states"):
        robot = env.unwrapped.scene["robot"]
        self._robot = robot
        self._sim = env.unwrapped.sim

        pairs: list[tuple[int, str]] = []
        for actuator in robot.actuators.values():
            indices = actuator.joint_indices
            if isinstance(indices, slice):
                indices = list(range(robot.num_joints))
            else:
                indices = indices.tolist()
            pairs.extend(zip(indices, actuator.joint_names))
        pairs.sort()

        self._joint_ids = [p[0] for p in pairs]
        self._joint_names: list[str] = [p[1] for p in pairs]

        proc = subprocess.Popen(
            [_ROS2_PYTHON, _BRIDGE_SCRIPT, topic],
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        self._proc = proc
        self._stdin = proc.stdin

        print(f"[JointStatePublisher] Joints ({len(self._joint_names)}): {self._joint_names}")

    def update(self):
        """Read current joint state and publish. Call once per env.step()."""
        sim_time = self._sim.current_time
        data = {
            "stamp": float(sim_time),
            "names": self._joint_names,
            "pos": self._robot.data.joint_pos[0, self._joint_ids].tolist(),
            "vel": self._robot.data.joint_vel[0, self._joint_ids].tolist(),
            "eff": self._robot.data.applied_torque[0, self._joint_ids].tolist(),
        }
        self._stdin.write(json.dumps(data) + "\n")
        self._stdin.flush()

    def close(self):
        self._stdin.close()
        self._proc.wait()
