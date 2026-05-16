# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from gym.wrappers.frame_stack import LazyFrames

# import d4rl
import gym
from gym.wrappers import AtariPreprocessing, FrameStack, RecordVideo
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append('..')

TensorBatch = List[torch.Tensor]

import pickle
sys.path.append('../offline_data')

# VQVAE
sys.path.append("../VQVAE")
sys.path.append("..")
from VQVAE.modules import VectorQuantizedVAE
from VQVAE.prior_train import PriorNet
from VQVAE.vqvae_train import save, load
from VQVAE.dataset import load_data
from torch.distributions import Categorical
import json
from torch.utils.data import Dataset, DataLoader

class D4RLGameDataset(Dataset):
    def __init__(self, trajectories, sequence_length=4, normalize_reward=False, normalize=False, env=None):
        self.sequence_length = sequence_length

        # 壓縮多個遊戲為單一連續數據
        self.observations = []
        self.actions = []
        self.rewards = []

        cnt = 0
        LIMIT = -1

        for traj in trajectories:
            cnt += 1
            if cnt == LIMIT:
                break
            self.observations.append(traj['observations'])
            self.actions.append(traj['actions'])
            self.rewards.extend(traj['rewards'])

        self.observations = np.concatenate(self.observations, axis=0)
        self.actions = np.concatenate(self.actions, axis=0, dtype=np.float32)
        self.rewards = np.array(self.rewards, dtype=np.float32)

        self.reward_mod_dict = {}
        if normalize_reward:
            self.reward_mod_dict = modify_reward({"rewards": self.rewards}, env)
            self.rewards /= (self.reward_mod_dict["max_ret"] - self.reward_mod_dict["min_ret"])
            self.rewards *= self.reward_mod_dict["max_episode_steps"]

        if normalize:
            state_mean, state_std = compute_mean_std(self.observations, eps=1e-3)
            self.observations = normalize_states(self.observations, state_mean, state_std)
        else:
            state_mean, state_std = 0, 1 
        self.state_mean = state_mean
        self.state_std = state_std

        self.valid_indices = len(self.observations) - self.sequence_length + 1
    def get_state_mean(self):
        return self.state_mean
    def get_reward_mod_dict(self):
        return self.reward_mod_dict
    def get_state_std(self):
        return self.state_std
    
    def __len__(self):
        return self.valid_indices

    def __getitem__(self, idx):
        obs_seq = self.observations[idx]
        acts_seq = self.actions[idx:idx + self.sequence_length]

        reward_sum = np.sum(self.rewards[idx:idx + self.sequence_length])

        return obs_seq, acts_seq, reward_sum

import argparse
    
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENVS_WITH_GOAL = ("pen", "door", "hammer", "relocate")

parser = argparse.ArgumentParser(description='vqvae train.')
parser.add_argument('--env', type=str, help='Environment name')
parser.add_argument('--seed', type=int, help='Random seed')
parser.add_argument('--eval_agent', type=str, help='Evaluation agent')
parser.add_argument('--gpuid', type=str, help='GPU ID')
parser.add_argument('--render', type=bool, default=False, help='Render environment')
parser.add_argument('--seqlen', type=int, default=9, help='Sequence length')
parser.add_argument('--k', type=int, default=16, help='Codebook size')
parser.add_argument('--suffix', type=str, default="", help='Custom suffix to append to paths')
parser.add_argument('--vqvae_path', type=str, default="", help='Custom VQVAE model path (overrides defaults)')
parser.add_argument('--prior_path', type=str, default="", help='Custom prior model path (overrides defaults)')
parser.add_argument('--training_dataset', type=str, default="",
                    help='Training dataset, must put your dataset in root/offline_data/')
parser.add_argument('--testing_dataset', type=str, default="",
                    help='Testing dataset, must put your dataset in root/offline_data/')

args = parser.parse_args()
env = args.env
envname = env
training_dataset = args.training_dataset
testing_dataset = args.testing_dataset

suffix = args.suffix
 
@dataclass
class TrainConfig:
    eval_agent: str = ""
    device: str = "cuda"
    seed: int = args.seed
    render: bool = args.render
    env: str = envname
    eval_freq: int = int(1e4)
    n_episodes: int = 10
    offline_iterations: int = int(1e6)
    online_iterations: int = int(1e6)
    checkpoints_path: str = f"log/{env}/MAQ_iql{suffix}_seed{args.seed}"
    load_model: str = ""

    gpuid: str = "0"
    buffer_size: int = 500000
    batch_size: int = 32
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    lr: float = 3e-4
    normalize: bool = False
    normalize_reward: bool = False
    seqlen: int = args.seqlen
    k: int = args.k
    training_dataset: str = f"{training_dataset}"
    testing_dataset: str = f"{testing_dataset}"
    # MAQ settings
    bm_loss_coefficient: float = 0.5
    suffix: str = f"{suffix}"
    vqvae_model_path: str = args.vqvae_path if args.vqvae_path else f"../VQVAE/log/{args.env}_{suffix}/"
    prior_model_path: str = args.prior_path if args.prior_path else f"../VQVAE/log/prior/{args.env}_{suffix}/"
    


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
        
def hard_update(target: nn.Module, source: nn.Module):
    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(local_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def decoded_primitive_actions(states, K_index, vqvae_model):
    decoded_actions = vqvae_model.forward_decoder(states, K_index)
    return decoded_actions

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        seq_len: int = 9,
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, seq_len , action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._offline_datas = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads from npk file
    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, dataset, file_name, sequence_length=9):
        n_transitions = len(dataset)
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        offline_datas = []
        for i in range(n_transitions - 1):
            obs, acts_seq, reward_sum = dataset[i]
            next_obs, _, _ = dataset[i + 1]
            states_list.append(obs)
            actions_list.append(acts_seq)
            rewards_list.append(reward_sum)
            next_states_list.append(next_obs)
            dones_list.append(0.0)  
            offline_datas.append(1.0)
        obs, acts_seq, reward_sum = dataset[n_transitions-1]
        states_list.append(obs)
        actions_list.append(acts_seq)
        rewards_list.append(reward_sum)
        next_states_list.append(obs)
        dones_list.append(1.0)
        offline_datas.append(1.0)
        
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)
        rewards = np.array(rewards_list, dtype=np.float32)
        next_states = np.array(next_states_list, dtype=np.float32)
        dones = np.array(dones_list, dtype=np.uint8)
        offline_datas = np.array(offline_datas, dtype=np.uint8)
        n = states.shape[0]
        if self._size + n > self._buffer_size:
            raise ValueError("Total transitions exceed replay buffer size!")
        
        self._states[self._size: self._size + n] = self._to_tensor(states)
        self._actions[self._size: self._size + n] = self._to_tensor(actions)
        self._rewards[self._size: self._size + n] = self._to_tensor(rewards[..., None])
        self._next_states[self._size: self._size + n] = self._to_tensor(next_states)
        self._dones[self._size: self._size + n] = self._to_tensor(dones[..., None])
        self._offline_datas[self._size: self._size + n] = self._to_tensor(offline_datas[..., None])
        self._size += n
        self._pointer = self._size - 1

        print(f"File path {file_name} has data size: {n}. Total replay buffer size: {self._size}")


    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        offline_datas = self._offline_datas[indices]
        return [states, actions, rewards, next_states, dones, offline_datas]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        is_offline: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._offline_datas[self._pointer] = self._to_tensor(is_offline)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)
        # raise NotImplementedError

def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward



@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, render: bool = False, vqvae_model = None
) -> Tuple[np.ndarray, np.ndarray]:
    if render:
        env = RecordVideo(env, video_folder="log/result_video", name_prefix="iql_pacman")
    actor.eval()
    episode_rewards = []
    episode_lengths = []
    successes = []
    for i in range(n_episodes):
        state, done = env.reset(), False
        done = False
        episode_reward = 0.0
        episode_length = 0
        goal_achieved = False
        
        while not done:
            index_action = actor.act(state, device) 
            

            action = decoded_primitive_actions(
                torch.tensor(
                    np.expand_dims(state, axis=0), device=device, dtype=torch.float32
                ),
                torch.tensor(
                    np.expand_dims(index_action, axis=0), device=device
                )
                ,
                vqvae_model
            )
            for a in action[0]:
                state, reward, done, env_infos = env.step(a)
                episode_length += 1    
                episode_reward += reward
                if not goal_achieved:
                    goal_achieved = is_goal_reached(reward, env_infos)
            
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes)

def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.conv(x)

class PolicyValueFunction(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.cnn = CNN()
        self.action_dim = act_dim # act dim = codebook size
        self.qf1 = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.qf2 = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.vf = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, state, action=None):
        x = state
        if action is not None:
            a = F.one_hot(action.long(), num_classes=self.action_dim).float()
            x1 = torch.cat([x, a], 1)
            q1 = self.qf1(x1).squeeze(1)
            q2 = self.qf2(x1).squeeze(1)
        v = self.vf(x).squeeze(1)
        logits = self.policy(x)

        if action is not None:
            return logits, v, q1, q2
        return logits, v, None, None
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state, device=device, dtype=torch.float32)
        logits = self.policy(state)
        return torch.argmax(logits, dim=1).cpu().numpy()[0]

class MAQImplicitQLearning:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        config: TrainConfig = None,
    ):
        self.actor = actor
        self.q_target = copy.deepcopy(self.actor).requires_grad_(False).to(device)
        self.actor_optimizer = actor_optimizer
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device
        self.config = config


        
        
    def _update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
        prior_model: PriorNet,
        vqvae_model: VectorQuantizedVAE,
        OFFLINE_STAGE: torch.Tensor
    ):
        terminals = terminals.squeeze(1)
        rewards = rewards.squeeze(1)

        offline_mask = OFFLINE_STAGE.bool()  
        direct_index_action = actions[:, 0, 0].unsqueeze(1) 
        actions = actions.view(actions.size(0), -1) 
        
        
        _, _, _, vqvae_index_action = vqvae_model(observations, actions)
        vqvae_index_action = vqvae_index_action.unsqueeze(1)  
        index_actions = torch.where(offline_mask, vqvae_index_action, direct_index_action).squeeze(1) 
        
        with torch.no_grad():
            _, _, q1, q2 = self.q_target(observations, index_actions)
            target_q = torch.min(q1, q2)
        logits, v, q1_pred, q2_pred = self.actor(observations, index_actions)
        assert target_q.shape == v.shape
        adv = target_q - v
        log_dict["q"] = q1_pred.mean().item()
        log_dict["v"] = v.mean().item()
        log_dict["adv"] = adv.mean().item()
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        with torch.no_grad():
            _, next_v, _, _ = self.actor(next_observations)
            next_q = rewards + (1.0 - terminals.float()) * self.discount * next_v
        assert q1_pred.shape == q2_pred.shape == next_q.shape
        q_loss = (F.mse_loss(q1_pred, next_q) + F.mse_loss(q2_pred, next_q)) / 2
        log_dict["q_loss"] = q_loss.item()
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        log_probs = torch.distributions.Categorical(logits=logits).log_prob(index_actions)
        actor_loss = torch.mean(-log_probs * exp_adv)
        log_dict["actor_loss"] = actor_loss.item()

        target_logits = prior_model(observations)
        
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        kl_div = kl_loss(F.log_softmax(logits, dim=1), F.log_softmax(target_logits.detach(), dim=1))



        loss = v_loss + q_loss + actor_loss + self.config.bm_loss_coefficient * kl_div
        log_dict["loss"] = loss.item()
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.actor, self.tau)

    def train(self, batch: TensorBatch,
              prior_model: PriorNet, 
              vqvae_model: VectorQuantizedVAE,
              ) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            OFFLINE_STAGE,
        ) = batch
        log_dict = {}
        self._update(observations, actions, rewards, next_observations, dones, log_dict, prior_model, vqvae_model, OFFLINE_STAGE)


        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]
    
    def decide_agent_actions(self, state, eval=True):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        logits = self.actor.policy(self.actor.cnn(state))
        return torch.argmax(logits, dim=1).cpu().numpy(), None, None




@pyrallis.wrap()
def load_IQL_agent(config: TrainConfig, env_id: str, model_path: str, seed: int):
    # Use the paths directly from config instead of reconstructing them
    config.env = envname
    observations, actions = load_data(config.env, config.seed)

    
    vqvae_config = json.load(open(os.path.join(config.vqvae_model_path, 'config.json')))
    vqvae_model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=vqvae_config['sequence_length'], K=vqvae_config['k'], dim=vqvae_config['hidden_size'], output_dim=actions.shape[1]).to(config.device)
    ls = [filename for filename in os.listdir(config.vqvae_model_path) if filename.endswith(".pth") ]

    sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[1]))
    
    last_vqvae_cp = sorted_checkpoints[-1]
    load(vqvae_model, os.path.join(config.vqvae_model_path, last_vqvae_cp))
    vqvae_model.eval()
    prior_model = PriorNet(state_dim=observations.shape[1], hidden_dim=vqvae_config['hidden_size'], output_K=vqvae_config['k']).to(config.device)
    

    ls = [filename for filename in os.listdir(config.prior_model_path) if filename.endswith(".pth") ]
    

    sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[2]))
    
    last_prior_cp = sorted_checkpoints[-1]
    load(prior_model, os.path.join(config.prior_model_path,last_prior_cp))
    prior_model.eval()
    N_action = vqvae_config['k']
    sequence_length = vqvae_config['sequence_length']
    if config.normalize: 
        state_mean, state_std = 0, 1
        pass
    else:
        state_mean, state_std = 0, 1

    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] # action

    actor = PolicyValueFunction(state_dim= state_dim,act_dim=N_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)


    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
        "config": config,
    }

    trainer = MAQImplicitQLearning(**kwargs)
    trainer.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))

    return trainer, eval_env, state_mean, state_std, vqvae_model, prior_model


@pyrallis.wrap()
def train(config: TrainConfig):
    global envname
    config.env = envname 

    observations, actions = load_data(config.env, config.seed)
    
    vqvae_config = json.load(open(os.path.join(config.vqvae_model_path, 'config.json')))
    vqvae_model = VectorQuantizedVAE(state_dim=observations.shape[1], seq_len=vqvae_config['sequence_length'], 
                                     K=vqvae_config['k'], dim=vqvae_config['hidden_size'], 
                                     output_dim=actions.shape[1]).to(config.device)

    ls = [filename for filename in os.listdir(config.vqvae_model_path) if filename.endswith(".pth")]
    
    sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[1]))
    
    last_vqvae_cp = sorted_checkpoints[-1]
    load(vqvae_model, os.path.join(config.vqvae_model_path, last_vqvae_cp))
    vqvae_model.eval()
    
    prior_model = PriorNet(state_dim=observations.shape[1], hidden_dim=vqvae_config['hidden_size'], 
                          output_K=vqvae_config['k']).to(config.device)
    
    ls = [filename for filename in os.listdir(config.prior_model_path) if filename.endswith(".pth")]
    
    sorted_checkpoints = sorted(ls, key=lambda x: int(x.split('_')[2]))
    
    last_prior_cp = sorted_checkpoints[-1]
    load(prior_model, os.path.join(config.prior_model_path, last_prior_cp))
    prior_model.eval()
    N_action = vqvae_config['k']
    sequence_length = vqvae_config['sequence_length']

    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  
    reward_mod_dict = {}
    file_name = f"../offline_data/{training_dataset}"
    
    trajectories = load_trajectories(file_name)
    dataset = D4RLGameDataset(trajectories, sequence_length=sequence_length,
                                normalize_reward=config.normalize_reward, 
                                normalize=config.normalize, env=config.env)
    state_mean = dataset.get_state_mean()
    state_std = dataset.get_state_std()
    reward_mod_dict = dataset.get_reward_mod_dict()
    
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
        seq_len=sequence_length
    )
    replay_buffer.load_d4rl_dataset(dataset, file_name=file_name, sequence_length=sequence_length)


    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
    
    actor = PolicyValueFunction(state_dim=state_dim, act_dim=N_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)

    kwargs = {
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
        "config": config,
    }

    print("---------------------------------------")
    print(f"Training Discrete IQL, Env: {config.env}")
    print("---------------------------------------")

    # Initialize trainer
    trainer = MAQImplicitQLearning(**kwargs)

    # Load model if specified
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # Setup logging
    writer = SummaryWriter(config.checkpoints_path)

    evaluations = []

    # Environment variables for online training
    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False
    total_online_episodes = 0
    total_online_timesteps = 0
    eval_successes = []
    train_successes = []

    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
        
        online_log = {}
        # Online fine-tuning phase
        if t >= config.offline_iterations:
            # Get action from policy
            action_logits, _, _, _ = actor(
                torch.tensor(
                    np.expand_dims(state, axis=0), device=config.device, dtype=torch.float32
                )
            )
            
            index_action = torch.distributions.Categorical(logits=action_logits).sample()
            index_action = index_action.cpu().detach().numpy()
            index_action = torch.from_numpy(index_action).to(config.device)
            
            # Convert index action to primitive actions
            action = decoded_primitive_actions(
                torch.tensor(
                    np.expand_dims(state, axis=0), device=config.device, dtype=torch.float32
                ),
                index_action,
                vqvae_model
            )
            
            action_reward = 0
            # Execute primitive actions in sequence
            for a in action[0]:
                next_state, reward, done, env_infos = env.step(a)
                action_reward += reward
                episode_step += 1
                total_online_timesteps += 1
                
                if not goal_achieved:
                    goal_achieved = is_goal_reached(reward, env_infos)
                
            episode_return += action_reward
            
            # Normalize reward if needed
            if config.normalize_reward:
                action_reward = modify_reward_online(action_reward, config.env, **reward_mod_dict)
                
            # Format action for replay buffer
            index_action = index_action.repeat(1, sequence_length, action_dim).cpu().detach().numpy()
            replay_buffer.add_transition(state, index_action, action_reward, next_state, done, False)
                        
            state = next_state
            
            # Handle episode completion
            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = normalized_return * 100.0
                
                total_online_episodes += 1
                print(f"Online episode {total_online_episodes}, episode return {episode_return}, episode length {episode_step}")
                
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        # Sample batch and train
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch, prior_model, vqvae_model)
        
        # Update logging info
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        
        # Log to tensorboard
        for key, value in log_dict.items():
            writer.add_scalar(key, value, trainer.total_it)

        # Evaluate periodically
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                vqvae_model=vqvae_model
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(eval_score)
            
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
                
            normalized_eval_score = normalized * 100.0
            evaluations.append(normalized_eval_score)
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            
            # Save checkpoint
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
                
            # Log evaluation metrics
            for key, value in eval_log.items():
                writer.add_scalar(key, value, trainer.total_it)

def load_trajectories(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    train() 