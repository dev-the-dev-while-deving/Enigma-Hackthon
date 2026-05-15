import re

file_path = 'test_cases/test_case_intersection.py'
with open(file_path, 'r') as f:
    code = f.read()

# 1. Update evaluate_trajectory
eval_old = """    # We want to maximize forward progress (X axis)
    progress = xs[-1] - xs[0] if len(xs) > 0 else 0
    
    # We want to penalize moving off-road or swerving too far laterally
    off_road_penalty = np.sum((ys < -2) | (ys > 14)) * 10"""

eval_new = """    # We want to maximize progress in any direction (for intersection turning/crossing)
    progress = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) if len(xs) > 0 else 0
    
    # Off-road penalty removed since intersection map has different boundaries
    off_road_penalty = 0.0"""
code = code.replace(eval_old, eval_new)

# 2. Update Env setup
env_old = """env = gym.make("highway-v0", render_mode="human", config={
    "manual_control": False,
    "lanes_count": 2,            
    "duration": 10000000,        
    "simulation_frequency": 15,
    "collision_reward": -50,   # Massive negative reward instead of a lesser reward
    "high_speed_reward": 1,    # Reward for making progress
    "right_lane_reward": 0.1,  # Reward for staying right
    "reward_speed_range": [20, 30]
})"""

env_new = """env = gym.make("intersection-v0", render_mode="human", config={
    "manual_control": False,
    "duration": 10000000,        
    "simulation_frequency": 15,
    "collision_reward": -50,
    "high_speed_reward": 1,
    "arrived_reward": 10,
    "spawn_probability": 0.6,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 12,
        "noise": 0.05,
        "features": ["presence", "x", "y", "vx", "vy"]
    }
})"""
code = code.replace(env_old, env_new)

# 3. Update Hard Shield and Env Step Action mapping
shield_old = """        # --- Absolute Hardcoded Safety Bound (AEB) ---
        # Instead of relying on the neural network's predictions, we use the true 
        # physical coordinates of the cars to mathematically guarantee we never crash.
        ego = env.unwrapped.vehicle
        crash_ahead = False
        blocked_left = False
        blocked_right = False
        
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                dx = v.position[0] - ego.position[0]
                dy = v.position[1] - ego.position[1]
                
                # Check Same Lane
                if abs(dy) < 2.0:
                    if 0 < dx < 15.0:  # Car is directly in front and very close (15 meters)
                        crash_ahead = True
                
                # Check Left Lane
                if -6.0 < dy <= -2.0:
                    if abs(dx) < 10.0: # Car is blocking the left blind spot / gap
                        blocked_left = True
                        
                # Check Right Lane
                if 2.0 <= dy < 6.0:
                    if abs(dx) < 10.0: # Car is blocking the right blind spot / gap
                        blocked_right = True

        action = best_action

        # Hardcoded Overtake / Brake logic based on physical bounds
        if crash_ahead and action in [1, 3]:
            # We are about to rear-end someone. Evade or brake!
            if not blocked_left:
                action = 0 # Evade Left
                print("[HARD SHIELD] Crash ahead! Evading Left.")
            elif not blocked_right:
                action = 2 # Evade Right
                print("[HARD SHIELD] Crash ahead! Evading Right.")
            else:
                action = 4 # Boxed in! Slam Brakes!
                print("[HARD SHIELD] Boxed in! Slamming Brakes.")
                
        # Prevent unsafe lane changes
        if action == 0 and blocked_left:
            action = 1
            print("[HARD SHIELD] Left lane blocked! Canceling lane change.")
            if crash_ahead: action = 4
            
        if action == 2 and blocked_right:
            action = 1
            print("[HARD SHIELD] Right lane blocked! Canceling lane change.")
            if crash_ahead: action = 4
            
        # --------------------------------------------"""

shield_new = """        # --- Absolute Hardcoded Safety Bound (AEB) ---
        ego = env.unwrapped.vehicle
        imminent_crash = False
        
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                dist = np.sqrt((v.position[0] - ego.position[0])**2 + (v.position[1] - ego.position[1])**2)
                # If any vehicle is closer than 8 meters in the intersection, brake immediately
                if dist < 8.0:
                    imminent_crash = True

        action = best_action

        if imminent_crash:
            action = 4 # Slam Brakes
            print("[HARD SHIELD] Imminent intersection collision! Slamming Brakes.")
            
        # Map our 5-mode Neural Network action to intersection-v0's Discrete(3) action
        # intersection-v0 actions: 0: SLOWER, 1: IDLE, 2: FASTER
        # NN actions: 0: LEFT, 1: IDLE, 2: RIGHT, 3: FASTER, 4: SLOWER
        gym_action_map = {0: 1, 1: 1, 2: 1, 3: 2, 4: 0}
        env_action = gym_action_map[action]
        # --------------------------------------------"""
code = code.replace(shield_old, shield_new)

# 4. Update env.step call
step_old = """        # 3. Environment Step
        obs, current_reward, done, truncated, info = env.step(action)"""

step_new = """        # 3. Environment Step
        obs, current_reward, done, truncated, info = env.step(env_action)"""
code = code.replace(step_old, step_new)

with open(file_path, 'w') as f:
    f.write(code)

