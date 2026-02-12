#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


DEFAULT_JSON_PATH = Path(
    "/home/iki/ws_ros2/src/anygrasp_ros2/anygrasp_sdk/grasp_detection/example_data/best_grasp.json"
)


class GraspJsonPublisher(Node):
    def __init__(self, json_path: Path, pose_topic: str, width_topic: str,
                 publish_hz: float, watch: bool):
        super().__init__("grasp_json_publisher")

        self.json_path = Path(json_path).expanduser().resolve()
        self.pose_topic = pose_topic
        self.width_topic = width_topic
        self.publish_hz = publish_hz
        self.watch = watch
        self.last_mtime = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,          # ✅ fixed
            durability=DurabilityPolicy.TRANSIENT_LOCAL,     # late subscribers still get last message
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, qos)
        self.width_pub = self.create_publisher(Float32, self.width_topic, qos)

        period = 1.0 / max(0.1, self.publish_hz)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(f"JSON: {self.json_path}")
        self.get_logger().info(f"Publishing pose:  {self.pose_topic}")
        self.get_logger().info(f"Publishing width: {self.width_topic}")
        self.get_logger().info(f"watch={self.watch} publish_hz={self.publish_hz}")

    def tick(self):
        if not self.json_path.exists():
            self.get_logger().warn(f"JSON not found: {self.json_path}")
            return

        mtime = self.json_path.stat().st_mtime
        if self.watch:
            if self.last_mtime is not None and mtime == self.last_mtime:
                return
            self.last_mtime = mtime

        try:
            data = json.loads(self.json_path.read_text())
        except Exception as e:
            self.get_logger().error(f"Failed to read JSON: {e}")
            return

        if not data.get("ok", False):
            self.get_logger().warn(f"JSON says no grasp: {data.get('reason', 'unknown')}")
            return

        frame_id = data.get("frame_id", "")
        pos = data["position"]
        ori = data["orientation"]
        width = float(data.get("width", 0.0))

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(pos["x"])
        msg.pose.position.y = float(pos["y"])
        msg.pose.position.z = float(pos["z"])
        msg.pose.orientation.x = float(ori["x"])
        msg.pose.orientation.y = float(ori["y"])
        msg.pose.orientation.z = float(ori["z"])
        msg.pose.orientation.w = float(ori["w"])

        self.pose_pub.publish(msg)

        w = Float32()
        w.data = width
        self.width_pub.publish(w)

        self.get_logger().info(
            f"Published grasp pose frame={frame_id} "
            f"pos=({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},{msg.pose.position.z:.3f}) "
            f"width={width:.3f}"
        )


def main():
    p = argparse.ArgumentParser()

    # Optional override, otherwise use the hardcoded path
    p.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help=f"Path to best_grasp.json (default: {DEFAULT_JSON_PATH})"
    )

    p.add_argument("--pose_topic", default="/anygrasp/grasp_pose")
    p.add_argument("--width_topic", default="/anygrasp/gripper_width")
    p.add_argument("--publish_hz", type=float, default=2.0, help="How often to check/publish")
    p.add_argument("--watch", action="store_true", help="Only publish when JSON file changes")

    args = p.parse_args()

    rclpy.init()
    node = GraspJsonPublisher(
        json_path=Path(args.json),
        pose_topic=args.pose_topic,
        width_topic=args.width_topic,
        publish_hz=args.publish_hz,
        watch=args.watch,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
