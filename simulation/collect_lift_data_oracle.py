import numpy as np
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T
import json

# Config
num_episodes = 2500
dataset_path = "/home/alberto/0_master/Thesis/robomimic/datasets/custom/lift_oracle_improved"
target_lift_height = 0.10 

controller_config = load_composite_controller_config(controller="BASIC")

env_kwargs = {
    "env_name": "LiftImproved",
    "robots": "Panda",
    "controller_configs": controller_config,
    "has_renderer": False,      
    "has_offscreen_renderer": False,
    "use_camera_obs": False,
    "reward_shaping": True,
    "control_freq": 20,
    "horizon": 400,
}

env = suite.make(**env_kwargs)

env = DataCollectionWrapper(env, dataset_path)

# Print env config
print("\n" + "="*40)
print("ENVIRONMENT CONFIGURATION")
print("="*40)
print(json.dumps(env_kwargs, indent=4, default=str))

print("\n--- Extracted Environment Specs ---")
print(f"Action Dimension: {env.action_dim}")
print(f"Observation Keys: {list(env.observation_spec().keys())}")
print("="*40 + "\n")

print(f"Generating {num_episodes} smooth Oracle demonstrations...")


for ep in range(num_episodes):
    obs = env.reset()
    state = "REACH"
    done = False
    
    # Get cube starting height
    cube_start_z = obs['cube_pos'][2]

    for _ in range(env.horizon):
        # Get current positions
        eef_pos = obs['robot0_eef_pos']
        cube_pos = obs['cube_pos']

        cube_yaw = T.mat2euler(T.quat2mat(obs['cube_quat']))[2]
        eef_yaw = T.mat2euler(T.quat2mat(obs['robot0_eef_quat']))[2]
        
        # Calculate how much to twist
        yaw_error = cube_yaw - eef_yaw
        yaw_error = (yaw_error + np.pi/4) % (np.pi/2) - np.pi/4
        
        # Initialize action: [dx, dy, dz, ax, ay, az, gripper]
        action = np.zeros(7)

        # Apply the rotation to the Z-axis
        action[5] = yaw_error * 2.0
        
        if state == "REACH":
            # Move horizontally to hover above cube
            target = cube_pos + [0, 0, 0.05]
            dist = np.linalg.norm(target - eef_pos)
            action[:3] = (target - eef_pos) * 5.0
            # Keep gripper open
            action[6] = -1.0 
            if dist < 0.01 and abs(yaw_error) < 0.05: 
                state = "GRAB"

        elif state == "GRAB":
            # Descend and close
            target = cube_pos + np.array([0, 0, -0.025])
            dist = np.linalg.norm(target - eef_pos)
            action[:3] = (target - eef_pos) * 5.0
            action[6] = -1.0
            if dist < 0.02: 
                action[6] = 1.0
                state = "CLOSE"
                close_timer = 0

        elif state == "CLOSE":
            # Stop the arm and close
            action[:3] = 0.0
            action[5] = 0.0
            action[6] = 1.0
            
            close_timer += 1
            if close_timer >= 10:
                state = "LIFT"

        elif state == "LIFT":
            action[2] = 0.5
            action[6] = 1.0
        
        # Step
        obs, reward, env_done, info = env.step(action)
        if env_kwargs["has_renderer"]: 
            env.render()

        # Check success condition
        current_cube_z = obs['cube_pos'][2]
        if current_cube_z > (cube_start_z + target_lift_height):
            print(f"Episode {ep} Success!")
            break

print("Data collection complete. Metadata is saved in the .hdf5 file.")