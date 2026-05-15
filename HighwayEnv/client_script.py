import socket
import json
import time
import random

# Connect to the game server
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 5005))
print("Connected to HighwayEnv Game Server!")

# Wait for ready status
data = client.recv(1024).decode()
print("Server says:", data.strip())

print("\nStarting automated random script loop...")
print("The script controls the car by sending random actions.")
print("0: Left, 1: Idle, 2: Right, 3: Faster, 4: Slower\n")

try:
    step = 0
    while True:
        # Test all possible moves sequentially (0: Left, 1: Idle, 2: Right, 3: Faster, 4: Slower)
        action = step % 5
        
        # Send action to server
        client.sendall(str(action).encode() + b'\n')
        
        # Get response back
        data = client.recv(1024)
        if not data:
            print("Server disconnected.")
            break
            
        result = json.loads(data.decode().strip())
        print(f"[{step}] Sent Action: {action} | Reward: {result['reward']:.2f} | Done: {result['done']}")
        
        if result['done']:
            print("\nGame Over! The car crashed. Closing the game window.")
            client.sendall(b"close\n")
            break
            
        step += 1
        time.sleep(0.01) # Very fast input rate

except KeyboardInterrupt:
    print("Agent stopped manually.")
finally:
    client.close()
