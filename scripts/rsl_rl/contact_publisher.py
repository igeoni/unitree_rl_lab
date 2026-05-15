"""ROS2 foot contact publisher for IsaacLab environments.

Usage in play.py:
    from contact_publisher import ContactPublisher

    pub = ContactPublisher(env)          # after gym.make()
    while simulation_app.is_running():
        obs, _, dones, _ = env.step(actions)
        pub.update()                     # in the simulation loop
    pub.close()
"""

import json
import os
import subprocess

# Names must match the body names tracked by scene["contact_forces"]
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contact_bridge.py")
_ROS2_PYTHON = "/usr/bin/python3"


class ContactPublisher:
    """Reads foot contact forces from IsaacLab and publishes via ROS2.

    Spawns a Python 3.10 subprocess (contact_bridge.py) that handles rclpy,
    and communicates with it via stdin JSON lines.

    Publishes one WrenchStamped per foot (env_0 only):
        /contact/left_foot   (geometry_msgs/WrenchStamped)  — world frame, N
        /contact/right_foot  (geometry_msgs/WrenchStamped)  — world frame, N
    """

    def __init__(self, env):
        self._sensor = env.unwrapped.scene["contact_forces"]
        self._sim = env.unwrapped.sim
        self._foot_ids = self._resolve_foot_ids()

        proc = subprocess.Popen(
            [_ROS2_PYTHON, _BRIDGE_SCRIPT],
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        self._proc = proc
        self._stdin = proc.stdin

        print(f"[ContactPublisher] Tracking feet: {FOOT_BODY_NAMES}")
        print(f"[ContactPublisher] Body IDs: {self._foot_ids}")

    def _resolve_foot_ids(self) -> list:
        body_names = self._sensor.body_names
        ids = []
        for name in FOOT_BODY_NAMES:
            if name in body_names:
                ids.append(body_names.index(name))
            else:
                raise RuntimeError(
                    f"[ContactPublisher] '{name}' not found in contact sensor bodies: {body_names}"
                )
        return ids

    def update(self):
        """Read current contact forces and publish. Call once per env.step()."""
        forces = self._sensor.data.net_forces_w[0, self._foot_ids, :]  # (2, 3) env_0
        data = {
            "stamp": float(self._sim.current_time),
            "lf": forces[0].tolist(),
            "rf": forces[1].tolist(),
        }
        self._stdin.write(json.dumps(data) + "\n")
        self._stdin.flush()

    def close(self):
        self._stdin.close()
        self._proc.wait()
