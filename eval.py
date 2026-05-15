import gymnasium as gym
import torch
import numpy as np
from test_cases.test_case_mpc import BayesianTrajectoryPredictor, extract_64dim_features, evaluate_trajectory, mc_dropout_predict
from collections import deque
import os
import logging
logging.getLogger().setLevel(logging.ERROR)

env = gym.make("highway-v0", render_mode="rgb_array", config={
    "manual_control": False,
    "lanes_count": 2,            
    "duration": 10000000,        
    "simulation_frequency": 15,
    "collision_reward": -50,   
    "high_speed_reward": 1,    
    "right_lane_reward": 0.1,  
    "reward_speed_range": [20, 30]
})

model = BayesianTrajectoryPredictor()
try:
    model.load_state_dict(torch.load('bayesian_model.pt'))
except:
    pass
model.eval()

survived_steps = []
for ep in range(3): # Run 3 quick episodes
    obs, info = env.reset()
    done = truncated = False
    step = 0
    frame_buffer = deque(maxlen=3)
    
    while not (done or truncated) and step < 500: # cap at 500
        feat = extract_64dim_features(obs, env)
        frame_buffer.append(feat)
        while len(frame_buffer) < 3: frame_buffer.append(feat)
        
        best_action = 1
        best_score = -float('inf')
        
        for a in range(5):
            one_hot = np.zeros(5, dtype=np.float32); one_hot[a] = 1.0
            seq_input = [np.concatenate([f, one_hot]) for f in frame_buffer]
            x_tensor = torch.tensor(np.array(seq_input), dtype=torch.float32).unsqueeze(0)
            
            mean_traj, epi_std, alea_std = mc_dropout_predict(model, x_tensor, num_samples=3) # lower samples for speed
            score = evaluate_trajectory(mean_traj.flatten(), epi_std.flatten(), obs)
            if score > best_score:
                best_score = score
                best_action = a
                
        ego = env.unwrapped.vehicle
        crash_ahead, blocked_left, blocked_right = False, False, False
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                dx, dy = v.position[0] - ego.position[0], v.position[1] - ego.position[1]
                if abs(dy) < 2.0 and 0 < dx < 15.0: crash_ahead = True
                if -6.0 < dy <= -2.0 and abs(dx) < 10.0: blocked_left = True
                if 2.0 <= dy < 6.0 and abs(dx) < 10.0: blocked_right = True

        action = best_action
        if crash_ahead and action in [1, 3]:
            if not blocked_left: action = 0
            elif not blocked_right: action = 2
            else: action = 4
        if action == 0 and blocked_left: action = 1 if not crash_ahead else 4
        if action == 2 and blocked_right: action = 1 if not crash_ahead else 4
            
        obs, _, done, truncated, _ = env.step(action)
        step += 1
    
    survived_steps.append(step)
    print(f"Eval Episode {ep+1} Survived for: {step} steps")

print(f"AVERAGE_STEPS: {np.mean(survived_steps):.2f}")
