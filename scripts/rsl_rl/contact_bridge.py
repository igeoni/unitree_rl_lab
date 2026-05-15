#!/usr/bin/env python3
"""ROS2 contact publisher bridge — runs under system Python 3.10.

Reads JSON lines from stdin and publishes WrenchStamped messages.
Launched as a subprocess by ContactPublisher in Python 3.11.
"""
import json
import sys

import rclpy
from geometry_msgs.msg import WrenchStamped


def main():
    rclpy.init()
    node = rclpy.create_node("isaac_contact_publisher")
    pub_left = node.create_publisher(WrenchStamped, "/contact/left_foot", 10)
    pub_right = node.create_publisher(WrenchStamped, "/contact/right_foot", 10)

    msg_left = WrenchStamped()
    msg_right = WrenchStamped()
    msg_left.header.frame_id = "left_ankle_roll_link"
    msg_right.header.frame_id = "right_ankle_roll_link"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            sec = int(data["stamp"])
            nsec = int((data["stamp"] % 1.0) * 1e9)

            for msg, key in [(msg_left, "lf"), (msg_right, "rf")]:
                msg.header.stamp.sec = sec
                msg.header.stamp.nanosec = nsec
                f = data[key]
                msg.wrench.force.x = f[0]
                msg.wrench.force.y = f[1]
                msg.wrench.force.z = f[2]

            pub_left.publish(msg_left)
            pub_right.publish(msg_right)
        except Exception as e:
            print(f"[contact_bridge] error: {e}", file=sys.stderr, flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
