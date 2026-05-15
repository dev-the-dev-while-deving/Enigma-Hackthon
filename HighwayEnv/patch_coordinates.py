import glob
import os

for file_path in glob.glob('test_cases/test_case_*.py'):
    with open(file_path, 'r') as f:
        code = f.read()

    # 1. Fix ego_pos
    old_ego = """        try:
            ego_pos = (float(obs[0][1]), float(obs[0][2]))
        except:
            ego_pos = (0.0, 0.0)"""
    new_ego = """        try:
            ego_pos = (float(env.unwrapped.vehicle.position[0]), float(env.unwrapped.vehicle.position[1]))
        except:
            ego_pos = (0.0, 0.0)"""
    code = code.replace(old_ego, new_ego)

    # 2. Fix finish_episode
    old_fin = """            for i in range(N):
                future = positions[i+1 : i+31]
                while len(future) < 30:
                    future.append(positions[-1] if len(positions) > 0 else (0.0, 0.0))
                true_futures.append(np.array(future, dtype=np.float32).flatten())"""
    new_fin = """            for i in range(N):
                future = positions[i+1 : i+31]
                while len(future) < 30:
                    future.append(positions[-1] if len(positions) > 0 else (0.0, 0.0))
                
                curr_x, curr_y = positions[i] if N > 0 else (0.0, 0.0)
                rel_future = []
                for (fx, fy) in future:
                    rel_future.append(fx - curr_x)
                    rel_future.append(fy - curr_y)
                true_futures.append(np.array(rel_future, dtype=np.float32).flatten())"""
    code = code.replace(old_fin, new_fin)

    with open(file_path, 'w') as f:
        f.write(code)

print("Coordinates successfully patched across all test cases!")
