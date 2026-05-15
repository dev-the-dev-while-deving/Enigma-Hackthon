#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Twist
import torch
import numpy as np
from collections import deque
from captum.attr import IntegratedGradients

# Assume BayesianTrajectoryPredictor and ModelWrapper are imported from a shared module
# For this script, we mock the class to demonstrate ROS2 integration structure.
from test_case_mpc import BayesianTrajectoryPredictor, ModelWrapper

class SafeTrajectoryNode(Node):
    def __init__(self):
        super().__init__('safe_trajectory_ai')
        
        self.get_logger().info("Initializing SafeTrajectory AI ROS2 Node...")
        
        # Load the advanced Temporal Transformer GMM Model
        self.model = BayesianTrajectoryPredictor(input_dim=69, seq_len=3, horizon=30, hidden_dim=128, num_modes=3)
        self.model.eval()
        try:
            self.model.load_state_dict(torch.load('bayesian_model.pt', map_location='cpu'))
            self.get_logger().info("Successfully loaded Temporal Transformer weights.")
        except Exception as e:
            self.get_logger().warn(f"Starting with fresh weights. Error: {e}")
            
        # Frame stacking buffer
        self.frame_buffer = deque(maxlen=3)
        
        # Explainability setup
        self.ig_wrapper = ModelWrapper(self.model)
        self.ig = IntegratedGradients(self.ig_wrapper)
        
        # ROS2 Subscriptions & Publications
        # In a real ZF vehicle, this would be camera/lidar feature vectors
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/highway_env/observations',
            self.observation_callback,
            10)
            
        self.publisher_cmd_vel = self.create_publisher(Twist, '/vehicle/cmd_vel', 10)
        self.publisher_rationale = self.create_publisher(String, '/explainability/rationale', 10)
        
        self.step_counter = 0

    def observation_callback(self, msg):
        """Triggered whenever a new state observation is received from the environment/sensors."""
        feat = np.array(msg.data, dtype=np.float32)
        
        # Maintain temporal frame stack
        self.frame_buffer.append(feat)
        if len(self.frame_buffer) < 3:
            return # Wait for buffer to fill
            
        best_action = 1
        best_score = -float('inf')
        best_x_tensor = None
        
        # MPC Rollout across candidate actions
        for candidate_action in range(5):
            one_hot = np.zeros(5, dtype=np.float32)
            one_hot[candidate_action] = 1.0
            
            seq_input = []
            for f in self.frame_buffer:
                seq_input.append(np.concatenate([f, one_hot]))
                
            x_tensor = torch.tensor(np.array(seq_input), dtype=torch.float32).unsqueeze(0)
            
            # Predict Trajectory (Normally we evaluate score here based on predicted means)
            # score = evaluate_trajectory(mean_traj, ...)
            # Mocking score for ROS demo:
            score = np.random.random() 
            
            if score > best_score:
                best_score = score
                best_action = candidate_action
                best_x_tensor = x_tensor
                
        # Publish the chosen action
        self.publish_action(best_action)
        
        # 3-Tier Explainability
        self.step_counter += 1
        if self.step_counter % 5 == 0:
            self.generate_explainability_rationale(best_x_tensor, best_action)

    def publish_action(self, action_idx):
        twist = Twist()
        # Map discrete actions to continuous cmd_vel
        # 0: Left, 1: Idle, 2: Right, 3: Faster, 4: Slower
        if action_idx == 0: twist.angular.z = 0.5
        elif action_idx == 2: twist.angular.z = -0.5
        elif action_idx == 3: twist.linear.x = 5.0
        elif action_idx == 4: twist.linear.x = -2.0
        
        self.publisher_cmd_vel.publish(twist)

    def generate_explainability_rationale(self, x_tensor, action_idx):
        try:
            x_tensor.requires_grad_()
            attributions = self.ig.attribute(x_tensor, target=0)
            feature_importances = torch.sum(torch.abs(attributions), dim=1).squeeze(0).detach().numpy()
            most_important_idx = np.argmax(feature_importances[:64])
            
            if most_important_idx < 5: focus = "Ego Vehicle Kinematics"
            elif 5 <= most_important_idx < 15: focus = "Leading Vehicle"
            elif 15 <= most_important_idx < 25: focus = "Adjacent Left Vehicle"
            elif 25 <= most_important_idx < 35: focus = "Adjacent Right Vehicle"
            else: focus = "Surrounding Traffic Topology"
            
            action_names = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
            rationale_msg = String()
            rationale_msg.data = f"ACTION: {action_names[action_idx]} | FOCUS: {focus}"
            self.publisher_rationale.publish(rationale_msg)
            self.get_logger().info(f"[EXPLAINABILITY] {rationale_msg.data}")
        except Exception as e:
            self.get_logger().warn(f"Explainability Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SafeTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
