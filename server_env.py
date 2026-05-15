import gymnasium as gym
import highway_env
import socket
import json
import time

env = gym.make("highway-v0", render_mode="human", config={
    "lanes_count": 2,            # 2 lanes limit
    "duration": 1000000,         # Unlimited extending
    "simulation_frequency": 15,  # smooth gameplay
})

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 5005))
server.listen(1)
server.settimeout(0.1)

print("\n" + "="*50)
print("SERVER RUNNING - WAITING FOR INPUT FROM ANOTHER SCRIPT")
print("="*50)

while True:
    try:
        conn = None
        while conn is None:
            try:
                conn, addr = server.accept()
                print(f"\n[+] Script connected from {addr}")
            except socket.timeout:
                # keep rendering to prevent pygame freezing
                env.render()

        obs, info = env.reset()
        done = truncated = False
        conn.settimeout(None) # Now wait for client commands
        
        # Initial ready message
        conn.sendall(json.dumps({"status": "ready"}).encode() + b'\n')

        while not (done or truncated):
            try:
                data = conn.recv(1024)
                if not data:
                    break
                    
                # Action mapping: 0=Left, 1=Idle, 2=Right, 3=Faster, 4=Slower
                action = int(data.decode().strip())
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                
                # Send result back
                response = json.dumps({
                    "reward": float(reward),
                    "done": bool(done or truncated)
                })
                conn.sendall(response.encode() + b'\n')
                
            except Exception as e:
                print("[-] Disconnected:", e)
                break
                
        print("[-] Game Over or Disconnected. Waiting for new connection...")
        conn.close()
    except KeyboardInterrupt:
        break

server.close()
env.close()
print("Shutdown server.")
