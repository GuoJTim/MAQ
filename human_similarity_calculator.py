import argparse
import numpy as np
import torch
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import os
import sys
import re
sys.path.append("human_similarity")
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
import gym 
import d4rl
import json
import csv
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import pprint
import pandas as pd
import matplotlib.pyplot as plt
import os
import moviepy.video.io.ImageSequenceClip
import pickle
from human_similarity.utils import *
import glob
from tensorboard.backend.event_processing import event_accumulator
sys.path.append('../offline_data')
from human_similarity_utils import (
    run_agent_episode,
    evaluate_agent_performance,
    evaluate_saved_predictions,
    calculate_dtw_distances,
    calculate_wasserstein_distances,
    calculate_min_dtw_distances,
    calculate_state_action_occupancy_mmd,
    calculate_trajectory_quality_diversity
)

DEFAULT_SEQLEN_IQL = 3
DEFAULT_K_IQL = 16
MAQ_IQL_CSV_FILENAME = "experiment_short.csv" 

# --- Environment Horizon Constant ---
# Set this to None to use environment's default horizon, or set to a specific number
# If set, episodes will terminate early if success is achieved before the horizon
ENVIRONMENT_HORIZON = None  # You can change this value (e.g., 1000, 500, etc.) 

def load_trajectories(file):
    """從檔案加載 trajectory 數據"""
    with open(file, 'rb') as f:
        return pickle.load(f)

def get_d4rl_dataset(env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
    dataset = d4rl.qlearning_dataset(gym.make(env))
    dones_float = np.zeros_like(dataset['rewards'])

    for i in range(len(dones_float) - 1):
        if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6 or dataset['terminals'][i] == 1.0:
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1

    return dataset['observations'].astype(np.float32), dataset['actions'].astype(np.float32), dataset['rewards'].astype(np.float32), dones_float.astype(np.float32), dataset['next_observations'].astype(np.float32)


def normalized_score(env,score):
    return env.get_normalized_score(score)*100

def Calc_Distance(A, B):
    # get L1 distance
    assert A.shape == B.shape
    return np.sum(np.abs(A - B))

def compute_dtw_distance(traj1, traj2):
    assert traj1[0].shape == traj2[0].shape
    distance, _ = fastdtw(traj1, traj2, dist=euclidean)
    return distance

def compute_euclidean_distance(traj1, traj2):
    assert len(traj1) == len(traj2), "The trajectories must have the same length."
    
    for t1, t2 in zip(traj1, traj2):
        assert t1.shape == t2.shape, "Each step in the trajectories must have the same shape."

    total_distance = sum(np.linalg.norm(t1 - t2) for t1, t2 in zip(traj1, traj2))
    
    return total_distance


def similarity_metrics(env, env_id, testing_dataset, eval_seqlen, eval_episodes, report_file, 
                        agents = [],
                        render=False,
                        horizon=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics_to_calculate = {
        'dtw': calculate_dtw_distances,
        'wasserstein': calculate_wasserstein_distances,
        'min_dtw': calculate_min_dtw_distances,
        'occupancy': calculate_state_action_occupancy_mmd,
        'traj_qd': calculate_trajectory_quality_diversity
    }


    # --- Evaluate Each Agent --- 
    for agent in agents:
        agent_seed = agent.get_seed()
        agent_type = agent.get_agent_type()

        human_traj_states = []
        human_traj_actions = []
        test_trajectories = load_trajectories(f"offline_data/{testing_dataset}")
        for traj in test_trajectories:
            human_traj_states.append(traj['observations'])
            human_traj_actions.append(traj['actions'])
        print(f"Loaded {len(human_traj_states)} human trajectories for comparison (seed {agent_seed}).")
    
        print(f"Processing Agent: {agent_type}, Seed: {agent_seed}")

        evaluate_agent_performance(
            agent=agent,
            env_id=env_id,
            eval_episodes=eval_episodes,
            human_trajs_states=human_traj_states,
            human_trajs_actions=human_traj_actions,
            metric_fns=metrics_to_calculate,
            base_seed=agent_seed, # Use agent's seed for its eval episodes
            render=render,
            results_prefix="agent_performance", # Specific prefix for performance results
            horizon=horizon # Pass horizon parameter
        )

    print("\n--- similarity_metrics function finished ---")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating methods.')
    parser.add_argument('--env', type=str,help='')
    parser.add_argument('--gpuid', type=str,help='',default='3')
    parser.add_argument('--eval_agent', type=str,help='',default='')
    parser.add_argument('--render', type=bool, default=False, help='Render and save video of agent performance')
    parser.add_argument('--suffix', type=str,help='',default='')
    parser.add_argument('--model_path', type=str,help='',default='')
    parser.add_argument('--seed', type=str,help='',default='')
    parser.add_argument('--training_dataset', type=str,help='',default='')
    parser.add_argument('--testing_dataset', type=str,help='',default='')
    
    

    args = parser.parse_args()
    if args.gpuid != '':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid  
    
    

    # Parameters
    eval_seqlen = 3
    eval_episodes = 100 
    if args.render:
        eval_episodes = 5
    
    seed = args.seed
    env_id = args.env
    method = args.eval_agent
    training_dataset = args.training_dataset
    testing_dataset = args.testing_dataset
    suffix = args.suffix
    if not method in ["MAQ+DSAC","MAQ+RLPD","MAQ+IQL","SAC","RLPD","IQL"]:
        print("Method must be one of MAQ+DSAC, MAQ+RLPD, MAQ+IQL, SAC, RLPD, IQL")
        exit(0)

    generic_agent_type = None
    if method == "MAQ+RLPD":
        base_paths = [f"RLPD_MAQ/log/exp_{suffix}"]
        target_keywords = ["RLPDMAQ"]
        print(f"Base Paths: {base_paths}")

        agents = get_best_MAQ_agent(
            base_paths, # Use the constructed base paths
            args.env,
            env_id,
            agent_type=RLPDMAQAgent,
            target_keywords=target_keywords,
            tag=suffix,
            seed=str(seed)
        )
        generic_agent_type=RLPDMAQAgent

    elif method == "MAQ+DSAC":
        base_paths = [f"RLPD_MAQ/log/exp_{suffix}"]
        target_keywords = ["DSACMAQ"]
        print(f"Base Paths: {base_paths}")

        agents = get_best_MAQ_agent(
            base_paths, # Use the constructed base paths
            args.env,
            env_id,
            agent_type=DSACMAQAgent,
            target_keywords=target_keywords,
            tag=suffix,
            seed=str(seed)
        )
        generic_agent_type=DSACMAQAgent
        
    elif method == "MAQ+IQL":
        generic_agent_type=MAQIQLAgent
        if args.model_path == '':
            base_paths = [
                f"IQL/log/{env_id}/MAQ_iql{suffix}_seed{seed}",
            ]
            print(f"Base Paths: {base_paths}")

            agents = get_best_MAQ_IQL_agent(
                base_paths,
                args.env,
                env_id,
                tag=suffix
            )
        
        
    elif method == "SAC":
        generic_agent_type=SB3Agent
        if args.model_path == '':
            agents = []
            ckpt = find_best_checkpoint_SAC(f"SAC/log/{args.env}/SAC_seed{seed}")
            agents.append(FlexibleAgentInterface(SB3Agent, ckpt, env_id))

        
    elif method == "RLPD":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        import d4rl.gym_mujoco
        import d4rl.locomotion
        import dmcgym
        from rlpd.wrappers import wrap_gym
        from human_similarity.agent_rlpd import *
        agents = []
        #checkpoint_0.orbax-checkpoint-tmp-0
        ckpt = f"/workspace/rlpd/log_door_seed1/s1_0pretrain_LN/checkpoints/checkpoint_1000000"
        agents.append(FlexibleAgentInterface(RLPDAgent, ckpt, env_id))
        generic_agent_type=RLPDAgent
    elif method == "IQL":  
        generic_agent_type=IQLAgent
        if args.model_path == '':
            agents = get_best_IQL_agent(
                [
                    f"IQL/log/{args.env}/iql_seed{seed}"
                ],
                args.env,
            env_id
        )
    
    if args.model_path != '':
        # load specific model path
        agents = []
        agents.append(FlexibleAgentInterface(generic_agent_type, args.model_path, env_id))
    
    if len(agents) == 0:
        print("No agents found!")
        exit(0)
    # print(">>>",agents)
    # tmp_check = []
    # for agent in agents:
    #     s = agent.test()
    #     tmp_check.append(s)
    #     agent.close()
    # print("\n".join(tmp_check))    
    # print("total agents:",len(agents)) # current version do not support multiple agents at same time
    assert len(agents) == 1 # must equals to one
    

    horizon = ENVIRONMENT_HORIZON
    
    similarity_metrics(args.env, env_id, testing_dataset, eval_seqlen, eval_episodes, f"total_human_sim_report_{args.env}.csv", agents, render=args.render, horizon=horizon)
