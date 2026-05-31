import numpy as np
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
from robosuite.controllers import load_composite_controller_config
import robosuite.utils.transform_utils as T

# Config
num_episodes = 2500
dataset_path = "/home/alberto/0_master/Thesis/robomimic/datasets/custom/pick_place_oracle"

gripper_interpolation_step = 0.15 

controller_config = load_composite_controller_config(controller="BASIC")

env = suite.make(
    env_name="PickPlaceCan", 
    robots="Panda",    
    controller_configs=controller_config,   
    use_object_obs=True,      
    has_renderer=False,       
    has_offscreen_renderer=False,
    use_camera_obs=False,       
    control_freq=20,
    horizon=500,                
    render_camera="frontview",  
)

env = DataCollectionWrapper(env, dataset_path)

print(f"Generating {num_episodes} Oracle demonstrations...")

for ep in range(num_episodes):
    obs = env.reset()
    state = "REACH" 
    
    # Initialize state variables
    current_gripper_val = -1.0
    target_gripper_state = -1.0 
    drop_timer = 0
    success_timer = 0

    for step in range(env.horizon):
        eef_pos = obs['robot0_eef_pos']
        can_pos = obs['Can_pos']
        
        # Gets the center of the specific area assigned to the can
        bin_pos = env.target_bin_placements[env.object_id]
        
        action = np.zeros(7) 

        if state == "REACH":
            # Hover above the can
            target = can_pos + np.array([0, 0, 0.15])
            dist = np.linalg.norm(target - eef_pos)
            action[:3] = (target - eef_pos) * 5.0
            target_gripper_state = -1.0 
            if dist < 0.02: 
                state = "GRAB"

        elif state == "GRAB":
            target = can_pos + np.array([0, 0, 0.02]) 
            dist = np.linalg.norm(target - eef_pos)
            action[:3] = (target - eef_pos) * 5.0
            target_gripper_state = -1.0
            if dist < 0.02: 
                state = "CLOSE_AND_WAIT"

        elif state == "CLOSE_AND_WAIT":
            target = can_pos + np.array([0, 0, 0.0])
            action[:3] = (target - eef_pos) * 5.0
            target_gripper_state = 1.0 
            
            # Wait until fingers are at least 50% closed before lifting
            if current_gripper_val > 0.5: 
                state = "LIFT"

        elif state == "LIFT":
            target_gripper_state = 1.0 
            # Move up
            target = np.array([eef_pos[0], eef_pos[1], 1.15])
            action[:3] = (target - eef_pos) * 5.0

            if eef_pos[2] > 1.10: 
                state = "HOVER_BIN"

        elif state == "HOVER_BIN":
            target_gripper_state = 1.0
            # Move to the box center
            target = np.array([bin_pos[0], bin_pos[1], 1.15]) 
            dist_xy = np.linalg.norm(target[:2] - eef_pos[:2])
            action[:3] = (target - eef_pos) * 5.0
            
            if dist_xy < 0.03: 
                state = "PLACE_DOWN"

        elif state == "PLACE_DOWN":
            target_gripper_state = 1.0
            # Lower into the bin
            target = np.array([bin_pos[0], bin_pos[1], bin_pos[2] + 0.08]) 
            dist_z = abs(eef_pos[2] - target[2])
            action[:3] = (target - eef_pos) * 5.0
            
            if dist_z < 0.02: 
                state = "RELEASE"

        elif state == "RELEASE":
            target_gripper_state = -1.0 
            action[:3] = np.zeros(3)
            
            # Wait for fingers to fully open
            if current_gripper_val <= -0.9: 
                drop_timer += 1
                # Wait ~0.5 seconds for can to settle
                if drop_timer > 10: 
                    state = "VERIFY"

        elif state == "VERIFY":
            target = np.array([eef_pos[0], eef_pos[1], 1.15])
            action[:3] = (target - eef_pos) * 5.0
            target_gripper_state = -1.0 
            
            # Check simulation success
            if env._check_success():
                success_timer += 1
                # Wait for 1 sec
                if success_timer > 20:
                    break 
            else:
                success_timer = 0 

        if current_gripper_val < target_gripper_state:
            current_gripper_val = min(target_gripper_state, current_gripper_val + gripper_interpolation_step)
        elif current_gripper_val > target_gripper_state:
            current_gripper_val = max(target_gripper_state, current_gripper_val - gripper_interpolation_step)

        # Output the action to the environment
        action[6] = current_gripper_val
        
        obs, reward, env_done, info = env.step(action)
        if env.has_renderer: 
            env.render()
            
    print(f"Finished episode {ep + 1}/{num_episodes}")

print("Data collection complete. .hdf5 file saved successfully!")