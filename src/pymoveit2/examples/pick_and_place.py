#!/usr/bin/env python3
"""
Pick and place node - handles all three colours (R, G, B) sequentially.
Picks each coloured cube and drops it into its matching bin once.

ros2 run pymoveit2 pick_and_place.py
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

import math


class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")

        self.declare_parameter("approach_offset", 0.31)
        self.approach_offset = float(self.get_parameter("approach_offset").value)

        self.colors_done = set()
        self.currently_picking = False

        self.callback_group = ReentrantCallbackGroup()

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=panda.joint_names(),
            base_link_name=panda.base_link_name(),
            end_effector_name=panda.end_effector_name(),
            group_name=panda.MOVE_GROUP_ARM,
            callback_group=self.callback_group,
        )
        self.moveit2.max_velocity = 0.1
        self.moveit2.max_acceleration = 0.1

        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=panda.gripper_joint_names(),
            open_gripper_joint_positions=panda.OPEN_GRIPPER_JOINT_POSITIONS,
            closed_gripper_joint_positions=panda.CLOSED_GRIPPER_JOINT_POSITIONS,
            gripper_group_name=panda.MOVE_GROUP_GRIPPER,
            callback_group=self.callback_group,
            gripper_command_action_name="gripper_action_controller/gripper_cmd",
        )

        self.sub = self.create_subscription(
            String, "/color_coordinates", self.coords_callback, 10
        )

        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        self.home_joints  = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0, math.radians(92.0), math.radians(50.0)]

        self.drop_joints = {
            "R": [1.645, 1.496, 1.143, -1.222, -1.457, 2.285, 0.861],
            "G": [2.897, 0.943, 0.047, -1.790, -0.298, 2.428, -0.110],
            "B": [2.897, 1.210, 1.081, -1.546, -1.268, 2.143, 0.016],
        }

        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()
        self.get_logger().info("Ready. Waiting for colour coordinates...")

    def coords_callback(self, msg):
        if self.currently_picking:
            return

        try:
            color_id, x, y, z = msg.data.split(",")
            color_id = color_id.strip().upper()

            if color_id in self.colors_done:
                return

            self.currently_picking = True
            target = [float(x), float(y), float(z)]
            self.get_logger().info(
                f"[{color_id}] Picking — target [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
            )

            pick_position = [target[0], target[1], target[2] - 0.60]
            quat_xyzw = [0.0, 1.0, 0.0, 0.0]

            # 1. Move to home
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            # 2. Move above target
            self.moveit2.move_to_pose(position=pick_position, quat_xyzw=quat_xyzw)
            self.moveit2.wait_until_executed()

            # 3. Open gripper
            self.gripper.open()
            self.gripper.wait_until_executed()

            # 4. Descend to object
            approach_position = [
                pick_position[0],
                pick_position[1],
                pick_position[2] - self.approach_offset,
            ]
            self.moveit2.move_to_pose(
                position=approach_position, quat_xyzw=quat_xyzw, cartesian=True
            )
            self.moveit2.wait_until_executed()

            # 5. Close gripper
            self.gripper.close()
            self.gripper.wait_until_executed()

            # 6. Return to home
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            # 7. Move to this colour's bin
            self.moveit2.move_to_configuration(self.drop_joints[color_id])
            self.moveit2.wait_until_executed()

            # 8. Open gripper to release
            self.gripper.open()
            self.gripper.wait_until_executed()

            # 9. Close gripper
            self.gripper.close()
            self.gripper.wait_until_executed()

            # 10. Return to start
            self.moveit2.move_to_configuration(self.start_joints)
            self.moveit2.wait_until_executed()

            self.colors_done.add(color_id)
            self.get_logger().info(f"[{color_id}] Done.")

            if len(self.colors_done) == 3:
                self.get_logger().info("All colours placed. Shutting down.")
                rclpy.shutdown()
            else:
                self.currently_picking = False

        except Exception as e:
            self.get_logger().error(f"Error in coords_callback: {e}")
            self.currently_picking = False


def main():
    rclpy.init()
    node = PickAndPlace()

    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        executor_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
