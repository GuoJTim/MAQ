import os
import json
import numpy as np
from abc import ABC, abstractmethod
from AbstractAgent import AbstractAgent
import sys
sys.path.append("RL")
sys.path.append("VQVAE")
sys.path.append("IQL")
from IQL.iql_MAQ import load_IQL_agent, load_IQL_and_decide_actions, load_IQL_and_decide_actions_time
import time
import yaml

class MAQIQLAgent(AbstractAgent):
    save_time = False
    time_csv = "MAQIQL_time.csv"
    def load_agent(self, model_path, env_id):
        seed = self.get_seed(model_path)
        print(model_path)
        
        # Default values
        k = 16
        seqlen = 9
        
        # Try finding config.yaml in the parent directory of the model_path
        # Assuming model_path is like .../IQLMAQ_seed1/IQLMAQ_seed1.pt
        parent_dir = os.path.dirname(model_path)
        config_path = os.path.join(parent_dir, "config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                if 'k' in config:
                    k = config['k']
                if 'seqlen' in config:
                    seqlen = config['seqlen']
                
                print(f"Loaded config from {config_path}: k={k}, seqlen={seqlen}")
            except Exception as e:
                print(f"Error loading config.yaml: {e}")
        else:
            print(f"No config.yaml found at {config_path}, using defaults or parsing path.")
            # Fallback to path parsing if needed, or just stick to defaults as per user "if no ... use ..."
            if "_k" in model_path:
                 try:
                     k = int(model_path.split("_k")[1].split("_")[0])
                 except: pass
            if "_sq" in model_path:
                 try:
                    seqlen = int(model_path.split("_sq")[1].split("_")[0])
                 except: pass

        print(k, seqlen)
        agent, _, self.state_mean, self.state_std, self.vqvae_model, self.prior_model = load_IQL_agent(model_path=model_path, env_id=env_id, seed=seed, k=k, seqlen=seqlen)
        return agent
        # return None, None, None, None, None, None
    def inference(self, state):
        policy_start = time.time()
        
        if self.save_time:
            actions, policy_time, vqvae_time = load_IQL_and_decide_actions_time(self.agent, state, self.state_mean, self.state_std, self.vqvae_model, self.prior_model)
            inference_time = policy_time + vqvae_time
            print(f"MAQIQL decide_agent_actions time: {policy_time} seconds")
            print(f"MAQIQL decoded_primitive_actions time: {vqvae_time} seconds")
            self._save_detailed_timing_to_csv(policy_time, vqvae_time, inference_time)
        else:
            actions = load_IQL_and_decide_actions(self.agent, state, self.state_mean, self.state_std, self.vqvae_model, self.prior_model)

        # print(actions.shape) 
        return actions[0]

    def get_seed(self,model_path):
        if "seed100" in model_path:
            return 100
        if "seed10" in model_path:
            return 10
        if "seed1" in model_path:
            return 1
        if "VQVAE" in model_path:
            if "seed100" in model_path['VQVAE']:
                return 100
            if "seed10" in model_path['VQVAE']:
                return 10
            if "seed1" in model_path['VQVAE']:
                return 1