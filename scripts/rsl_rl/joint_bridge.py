#!/usr/bin/env python3
"""ROS2 joint state publisher bridge — runs under system Python 3.10.

Reads JSON lines from stdin and publishes JointState messages.
Launched as a subprocess by JointStatePublisher in Python 3.11.
"""
import json
import sys

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/joint_states"
    rclpy.init()
    node = rclpy.create_node("isaac_joint_state_publisher")
    pub = node.create_publisher(JointState, topic, 10)

    msg = JointState()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            sec = int(data["stamp"])
            nsec = int((data["stamp"] % 1.0) * 1e9)

            msg.header = Header()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
            msg.header.frame_id = "world"
            msg.name = data["names"]
            msg.position = data["pos"]
            msg.velocity = data["vel"]
            msg.effort = data["eff"]

            pub.publish(msg)
        except Exception as e:
            print(f"[joint_bridge] error: {e}", file=sys.stderr, flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
