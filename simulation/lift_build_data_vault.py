import torch
import numpy as np
import random
import copy

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.env_utils as EnvUtils

# Configuration
MODEL_PATH = "/media/alberto/ExtDrive/Thesis/models/lift_vision_bc_edl_v7_frame_stacking/20260425131914/models/model_epoch_2125_best_validation_-6.364668560028076.pth"
SEQ_LENGTH = 3
MAX_STEPS = 250
FIXED_SEED = 42
OUTPUT_FILE = "/home/alberto/0_master/Thesis/MasterThesis/simulation/thesis_dataset_locked_seed_ALL_ENVS_BASE_edl_v7.npz"

# Seed locking
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Model setup
print("Loading the Evidential Model...")
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=MODEL_PATH, device=device)
env_meta = ckpt_dict["env_metadata"]

base_model = policy.policy.nets["policy"]
base_model.eval() 
obs_encoder = base_model.nets["encoder"]

NUM_EPISODES = 50

# Data recording
def record_multiple_episodes(env_name, num_distractors, spatial_ood=False, num_episodes=NUM_EPISODES):
    print(f"\n---> Recording [{env_name}] | Distractors: {num_distractors} | Spatial: {spatial_ood} <---")
    
    master_ep_data = None
    ep_successes = [] 
    
    for ep in range(num_episodes):
        print(f"  -> Running Episode {ep+1}/{num_episodes}")
        
        current_seed = FIXED_SEED + ep
        set_seeds(current_seed)
        
        curr_meta = copy.deepcopy(env_meta)
        if env_name == "LiftImproved":
            curr_meta["env_kwargs"]["num_distractors"] = num_distractors
            curr_meta["env_kwargs"]["spatial_ood_mode"] = spatial_ood

        env = EnvUtils.create_env_from_metadata(
            env_meta=curr_meta, env_name=env_name, render=False, render_offscreen=True, use_image_obs=True
        )

        ep_data = {
            "agentview_64d": [], "wrist_64d": [], "kinematics_27d": [],
            "raw_eef_pos": [], "raw_eef_quat": [], "action_taken": [], 
            "edl_gamma": [], "edl_nu": [], "edl_alpha": [], "edl_beta": []
        }

        obs = env.reset()
        policy.start_episode()
        obs_history = [obs for _ in range(SEQ_LENGTH)]
        
        ep_is_successful = False

        for step in range(MAX_STEPS):
            batch_obs = {}
            for mod in ["agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
                stacked = np.stack([h[mod] for h in obs_history], axis=0)
                
                if "image" in mod:
                    stacked = stacked.transpose(0, 3, 1, 2)
                    T, C, H, W = stacked.shape
                    stacked = stacked.reshape(T * C, H, W)

                    batch_obs[mod] = (torch.tensor(stacked).unsqueeze(0).float() / 255.0).to(device)
                else:
                    batch_obs[mod] = torch.tensor(stacked).unsqueeze(0).float().to(device)

            with torch.no_grad():
                nig = base_model.forward_train(obs_dict=batch_obs)
                
                gamma = nig["gamma"].cpu().numpy().flatten()
                nu = nig["v"].cpu().numpy().flatten()
                alpha = nig["alpha"].cpu().numpy().flatten()
                beta = nig["beta"].cpu().numpy().flatten()
                
                ep_data["edl_gamma"].append(gamma)
                ep_data["edl_nu"].append(nu)
                ep_data["edl_alpha"].append(alpha)
                ep_data["edl_beta"].append(beta)
                
                action = gamma[-7:] if len(gamma) > 7 else gamma

                v_cores = [m for m in obs_encoder.modules() if m.__class__.__name__ == 'VisualCore']
                vis_agent = v_cores[0](batch_obs["agentview_image"]).cpu().numpy().flatten()
                vis_wrist = v_cores[1](batch_obs["robot0_eye_in_hand_image"]).cpu().numpy().flatten()
                
                p_vec = torch.cat([batch_obs["robot0_eef_pos"].flatten(), batch_obs["robot0_eef_quat"].flatten(), batch_obs["robot0_gripper_qpos"].flatten()], dim=-1).cpu().numpy().flatten()

                ep_data["agentview_64d"].append(vis_agent)
                ep_data["wrist_64d"].append(vis_wrist)
                ep_data["kinematics_27d"].append(p_vec)

            ep_data["raw_eef_pos"].append(obs["robot0_eef_pos"])
            ep_data["raw_eef_quat"].append(obs["robot0_eef_quat"])
            ep_data["action_taken"].append(action)

            obs, _, _, _ = env.step(action)
            obs_history.pop(0)
            obs_history.append(obs)
            
            if hasattr(env.env, '_check_success') and env.env._check_success():
                ep_is_successful = True
                print(f"     Task succeeded at step {step}! Padding the rest of the array...")
                break

        # Padding
        steps_taken = len(ep_data["action_taken"])
        if steps_taken < MAX_STEPS:
            steps_to_pad = MAX_STEPS - steps_taken
            for _ in range(steps_to_pad):
                for key in ep_data.keys():
                    ep_data[key].append(ep_data[key][-1]) 

        if hasattr(env, 'env') and hasattr(env.env, 'close'): 
            env.env.close()
        else: 
            env.close()
            
        ep_successes.append(ep_is_successful)
            
        if master_ep_data is None:
            master_ep_data = {k: [] for k in ep_data.keys()}
            
        for k, v in ep_data.items():
            master_ep_data[k].append(np.array(v))

    final_dataset = {k: np.stack(v, axis=0) for k, v in master_ep_data.items()}
    final_dataset["success"] = np.array(ep_successes)
    
    return final_dataset

print("\n[Running ID Normal Baseline...]")
data_id = record_multiple_episodes("Lift", num_distractors=0, spatial_ood=False)

print("\n[Running OOD Giraffe...]")
data_gir = record_multiple_episodes("LiftGiraffe", num_distractors=0, spatial_ood=False)

print("\n[Running Spatial OOD...]")
data_spa = record_multiple_episodes("LiftImproved", num_distractors=0, spatial_ood=True)

print("\n[Running OOD 1 Distractor...]")
data_c1 = record_multiple_episodes("LiftImproved", num_distractors=1, spatial_ood=False)

print("\n[Running OOD 2 Distractors...]")
data_c2 = record_multiple_episodes("LiftImproved", num_distractors=2, spatial_ood=False)

print("\n[Running OOD 3 Distractors...]")
data_c3 = record_multiple_episodes("LiftImproved", num_distractors=3, spatial_ood=False)

print("\nCreating the vault...")
master_dataset = {}

# Map keys slightly differently so they don't accidentally stack the 1D success array incorrectly later
for key in data_id.keys():
    master_dataset[f"id_{key}"] = data_id[key]
    master_dataset[f"gir_{key}"] = data_gir[key]
    master_dataset[f"spa_{key}"] = data_spa[key]
    master_dataset[f"c1_{key}"] = data_c1[key]
    master_dataset[f"c2_{key}"] = data_c2[key]
    master_dataset[f"c3_{key}"] = data_c3[key]

# Save to disk
np.savez_compressed(OUTPUT_FILE, **master_dataset)
print(f"SUCCESS! File saved as '{OUTPUT_FILE}'.")