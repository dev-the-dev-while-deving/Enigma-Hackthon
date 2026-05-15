import gymnasium as gym
import torch
import numpy as np
from test_cases.test_case_mpc import BayesianTrajectoryPredictor, extract_64dim_features, evaluate_trajectory, mc_dropout_predict
from collections import deque
import logging

logging.getLogger().setLevel(logging.ERROR)

model = BayesianTrajectoryPredictor()
try:
    model.load_state_dict(torch.load('bayesian_model.pt', map_location='cpu'))
except Exception as e:
    print("Error loading model:", e)
model.eval()

def run_eval(env_name, config, name, max_steps=100, is_intersection=False):
    env = gym.make(env_name, render_mode="rgb_array", config=config)
    obs, info = env.reset()
    done = truncated = False
    step = 0
    frame_buffer = deque(maxlen=3)
    
    while not (done or truncated) and step < max_steps:
        feat = extract_64dim_features(obs, env)
        frame_buffer.append(feat)
        while len(frame_buffer) < 3: frame_buffer.append(feat)
        
        best_action = 1
        best_score = -float('inf')
        
        for a in range(5):
            one_hot = np.zeros(5, dtype=np.float32); one_hot[a] = 1.0
            seq_input = [np.concatenate([f, one_hot]) for f in frame_buffer]
            x_tensor = torch.tensor(np.array(seq_input), dtype=torch.float32).unsqueeze(0)
            
            mean_traj, epi_std, alea_std = mc_dropout_predict(model, x_tensor, num_samples=3)
            
            # Use appropriate evaluation metric based on environment type
            if is_intersection:
                xs, ys = mean_traj.flatten()[:5], mean_traj.flatten()[5:]
                score = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) if len(xs) > 0 else 0
            else:
                score = evaluate_trajectory(mean_traj.flatten(), epi_std.flatten(), obs)
                
            if score > best_score:
                best_score = score
                best_action = a
                
        # Basic hard shield for eval
        ego = env.unwrapped.vehicle
        crash_ahead = False
        blocked_left = False
        blocked_right = False
        imminent_crash = False
        
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                if is_intersection:
                    dist = np.sqrt((v.position[0] - ego.position[0])**2 + (v.position[1] - ego.position[1])**2)
                    if dist < 5.5: imminent_crash = True
                else:
                    dx, dy = v.position[0] - ego.position[0], v.position[1] - ego.position[1]
                    if abs(dy) < 2.0 and 0 < dx < 15.0: crash_ahead = True
                    if -6.0 < dy <= -2.0 and abs(dx) < 10.0: blocked_left = True
                    if 2.0 <= dy < 6.0 and abs(dx) < 10.0: blocked_right = True

        action = best_action
        
        if is_intersection:
            if imminent_crash: action = 4
            gym_action_map = {0: 1, 1: 1, 2: 1, 3: 2, 4: 0}
            env_action = gym_action_map[action]
        else:
            if crash_ahead and action in [1, 3]:
                if not blocked_left: action = 0
                elif not blocked_right: action = 2
                else: action = 4
            if action == 0 and blocked_left: action = 1 if not crash_ahead else 4
            if action == 2 and blocked_right: action = 1 if not crash_ahead else 4
            env_action = action
            
        obs, _, done, truncated, _ = env.step(env_action)
        step += 1
        
    print(f"[{name}] Survived {step} steps.")

# 1. Baseline Highway Cruising
config_baseline = {
    "manual_control": False, "lanes_count": 2, "duration": 10000000, "simulation_frequency": 15,
    "collision_reward": -50, "high_speed_reward": 1, "right_lane_reward": 0.1, "reward_speed_range": [20, 30]
}
run_eval("highway-v0", config_baseline, "Baseline MPC (Cruising)", max_steps=200)

# 2. Oncoming Traffic Dodge
config_oncoming = {
    "manual_control": False, "duration": 10000000, "simulation_frequency": 15,
    "collision_reward": -50, "high_speed_reward": 1.5,
    "observation": {"type": "Kinematics", "noise": 0.05, "features": ["presence", "x", "y", "vx", "vy"]}
}
run_eval("two-way-v0", config_oncoming, "Two-Way Oncoming (Collision Avoidance)", max_steps=200)

# 3. Dense Urban Intersection
config_intersection = {
    "manual_control": False, "duration": 10000000, "simulation_frequency": 15,
    "collision_reward": -50, "high_speed_reward": 1, "arrived_reward": 10, "spawn_probability": 0.6,
    "observation": {"type": "Kinematics", "vehicles_count": 12, "noise": 0.05, "features": ["presence", "x", "y", "vx", "vy"]}
}
run_eval("intersection-v0", config_intersection, "Urban Intersection (Yielding)", max_steps=200, is_intersection=True)

