#!/usr/bin/env python3
"""
Pick and place node - handles all three colours (R, G, B) sequentially.
Picks each coloured cube and drops it into its matching bin.
Runs NUM_TRIALS pick-and-place cycles per colour to test success rate.

ros2 run pymoveit2 pick_and_place.py
"""

from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from pymoveit2 import MoveIt2, GripperInterface
from pymoveit2.robots import panda

import math


NUM_TRIALS = 10
# If both gripper fingers average above this (metres), the object is held
GRIPPER_GRASP_THRESHOLD = 0.005


class PickAndPlace(Node):
    def __init__(self):
        super().__init__("pick_and_place")

        self.declare_parameter("approach_offset", 0.31)
        self.approach_offset = float(self.get_parameter("approach_offset").value)

        # Trial tracking
        self.trial_counts   = {"R": 0, "G": 0, "B": 0}
        self.success_counts = {"R": 0, "G": 0, "B": 0}
        self.colors_done    = set()
        self.currently_picking = False

        # Original cube positions in panda_link0 / world frame
        self.cube_origins = {
            "R": (0.6, 0.6, 0.70),
            "G": (0.8, 0.6, 0.70),
            "B": (0.4, 0.6, 0.70),
        }
        self.cube_model_names = {
            "R": "red_box",
            "G": "green_box",
            "B": "blue_box",
        }

        # Latest average gripper finger position (updated from /joint_states)
        self.gripper_position = 0.0

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

        # Read gripper finger positions to detect successful grasps
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Gazebo set_pose service to reset cubes between trials
        self.set_pose_client = self.create_client(
            SetEntityPose, '/world/empty_world/set_pose'
        )

        self.sub = self.create_subscription(
            String, "/color_coordinates", self.coords_callback, 10
        )

        self.start_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.radians(-125.0)]
        self.home_joints  = [0.0, 0.0, 0.0, math.radians(-90.0), 0.0, math.radians(92.0), math.radians(50.0)]

        # Per-colour drop joint configurations
        self.drop_joints = {
            "R": [1.645, 1.496, 1.143, -1.222, -1.457, 2.285, 0.861],
            "G": [2.897, 0.943, 0.047, -1.790, -0.298, 2.428, -0.110],
            "B": [2.897, 1.210, 1.081, -1.546, -1.268, 2.143, 0.016],
        }

        self.moveit2.move_to_configuration(self.start_joints)
        self.moveit2.wait_until_executed()
        self.get_logger().info(
            f"Ready. Running {NUM_TRIALS} trials per colour on /color_coordinates..."
        )

    def joint_state_callback(self, msg):
        """Store the average of the two finger joint positions."""
        try:
            idx1 = msg.name.index('panda_finger_joint1')
            idx2 = msg.name.index('panda_finger_joint2')
            self.gripper_position = (msg.position[idx1] + msg.position[idx2]) / 2.0
        except ValueError:
            pass

    def reset_cube(self, color_id):
        """Teleport a cube back to its spawn position via Gazebo set_pose service."""
        if not self.set_pose_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "set_pose service not available — cube will not be reset"
            )
            return

        origin = self.cube_origins[color_id]
        req = SetEntityPose.Request()
        req.entity.name = self.cube_model_names[color_id]
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(origin[0])
        req.pose.position.y = float(origin[1])
        req.pose.position.z = float(origin[2])
        req.pose.orientation.w = 1.0

        future = self.set_pose_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().info(f"[{color_id}] Cube reset to {origin}")

    def print_results(self):
        self.get_logger().info("========== PICK AND PLACE RESULTS ==========")
        for c in ["R", "G", "B"]:
            rate = self.success_counts[c] / NUM_TRIALS * 100
            self.get_logger().info(
                f"  {c}: {self.success_counts[c]}/{NUM_TRIALS} successful  ({rate:.0f}%)"
            )
        self.get_logger().info("============================================")

    def coords_callback(self, msg):
        if self.currently_picking:
            return

        try:
            color_id, x, y, z = msg.data.split(",")
            color_id = color_id.strip().upper()

            if color_id in self.colors_done:
                return
            if self.trial_counts.get(color_id, 0) >= NUM_TRIALS:
                return

            self.currently_picking = True
            self.trial_counts[color_id] += 1
            trial_num = self.trial_counts[color_id]

            target = [float(x), float(y), float(z)]
            self.get_logger().info(
                f"[{color_id}] Trial {trial_num}/{NUM_TRIALS} — "
                f"target [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
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

            # 6. Detect success: fingers stopped before fully closing means object is held
            pick_success = self.gripper_position > GRIPPER_GRASP_THRESHOLD
            if pick_success:
                self.success_counts[color_id] += 1
                self.get_logger().info(
                    f"[{color_id}] Trial {trial_num}: SUCCESS "
                    f"(finger avg = {self.gripper_position:.4f} m)"
                )
            else:
                self.get_logger().info(
                    f"[{color_id}] Trial {trial_num}: FAILED "
                    f"(finger avg = {self.gripper_position:.4f} m)"
                )

            # 7. Return to home (with or without object)
            self.moveit2.move_to_configuration(self.home_joints)
            self.moveit2.wait_until_executed()

            # 8. Move to this colour's bin
            self.moveit2.move_to_configuration(self.drop_joints[color_id])
            self.moveit2.wait_until_executed()

            # 9. Open gripper to release
            self.gripper.open()
            self.gripper.wait_until_executed()

            # 10. Close gripper
            self.gripper.close()
            self.gripper.wait_until_executed()

            # 11. Return to start
            self.moveit2.move_to_configuration(self.start_joints)
            self.moveit2.wait_until_executed()

            # 12. Reset cube to its original position for the next trial
            self.reset_cube(color_id)

            self.get_logger().info(
                f"[{color_id}] Trial {trial_num} complete. "
                f"Running total: {self.success_counts[color_id]}/{trial_num}"
            )

            # Check if this colour has finished all trials
            if self.trial_counts[color_id] >= NUM_TRIALS:
                self.colors_done.add(color_id)
                self.get_logger().info(
                    f"[{color_id}] All {NUM_TRIALS} trials done. "
                    f"Final success rate: "
                    f"{self.success_counts[color_id]/NUM_TRIALS*100:.0f}%"
                )

            if len(self.colors_done) == 3:
                self.print_results()
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
