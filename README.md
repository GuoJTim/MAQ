# Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization

This is the official repository of the NeurIPS 2025 paper [Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization](https://rlg.iis.sinica.edu.tw/papers/MAQ/).

If you use this work for research, please consider citing our paper as follows:
```
@inproceedings{
  guo2025learning,  
  title={Learning Human-Like RL Agents Through Trajectory Optimization With Action Quantization},
  author={Jian-Ting Guo, Yu-Cheng Chen, Ping-Chun Hsieh, Kuo-Hao Ho, Po-Wei Huang, Ti-Rong Wu, I-Chen Wu},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=1A4Nlibwl5}
}
```

<img src="docs/imgs/maq_architecture.svg" width="100%" />

We propose a human-likeness aware framework called Macro Action Quantization (MAQ) that consists of two components: (1) Human behavior distillation and (2) Reinforcement learning with Macro Actions.

The following instructions are prepared for reproducing the main experiments in the paper.


## Human-like Reinforcement Learning 
<!-- // show the door task and hammer task, with and without MAQ in RLPD
// in the door task mention that RLPD using back hand to open the door and MAQ+RLPD (our method) using a human like way to open the door 
// in the hammer task mention that RLPD due to its the reward-drvien RL agent, they maximize the reward by hammering the nail faster leading not human like behaviors -->

Human-like reinforcement learning remains underexplored in the RL community. Most research focuses on designing reward-driven agents; only a few studies investigate human-like RL that seeks both human-like behavior and optimal performance. But most of these methods rely on pre-defined behavior constraints or rule-based penalties, requiring substanital effort for handcrafed design.
### Door Task
| MAQ+RLPD (Ours) | RLPD |
|:---:|:---:|
| <img src="docs/clips/RLPD/MAQRLPD_door.gif" width="100%" /> | <img src="docs/clips/RLPD/RLPDAgent_door.gif" width="100%" /> |
| **Human-like RL** <br> Opens the door naturally. | **Reward-driven RL** <br> Uses an unnatural "backhand" strategy to open the door, maximizing reward but sacrificing naturalness. |
### Hammer Task
| MAQ+RLPD (Ours) | RLPD |
|:---:|:---:|
| <img src="docs/clips/RLPD/MAQRLPD_hammer.gif" width="100%" /> | <img src="docs/clips/RLPD/RLPDAgent_hammer.gif" width="100%" /> |
| **Human-like RL** <br> Performs smoothly, striking the nail multiple times with controlled precision. | **Reward-driven RL** <br> Hammers aggressively fast to maximize reward, resulting in unnatural behavior. |

## Training Macro Action Quantization

### Prerequisites

The program requires a Linux platform with at least one NVIDIA GPU to operate.
For training RLPD, the CUDA version must be newer than 12.0.

### Build Programs

Clone this repository with the required submodules:
```bash
git clone --recursive ..
cd MAQ
```

Enter the container to continue the instructions:
```bash
# start the container
./scripts/start-container.sh
```

> [!NOTE]
> All the instructions must be executed in the container.

### (Optional) Preprocessing Training and Testing Datasets
<!-- // the dataset must store in the offline_data and can use ./offline_data/gen_offline_data.py to generate the dataset provided by d4rl
// tell them the dataset format, and if using the customize dataset must change to that format -->


> [!NOTE]
> This section is optional. If you are using a customized dataset, please review this section. If you are using the default dataset, you can skip this.

The datasets used for training and testing must be saved as `.pkl` files in the `offline_data/` folder, and must have the same format as the following:
```python
[
    { # Trajecotry 1
        'observations': np.array([...]),      # Shape: (T, obs_dim)
        'actions': np.array([...]),           # Shape: (T, action_dim)
        'rewards': np.array([...]),           # Shape: (T, )
        'next_observations': np.array([...]), # Shape: (T, obs_dim)
        'terminals': np.array([...])          # Shape: (T, )
    },
    ... # Trajecotry 2
]
# T is the trajectory length, obs_dim is the observation dimension, action_dim is the action dimension
```
For verification, you can run:
```bash
python3 offline_data/check_dataset_format.py --dataset_path "your_dataset.pkl"
```
If the dataset is legal, it will print "is legal"; otherwise, it will print "illegal: error_message".


### Train Macro Action Quantization Methods
To reproduce the results in the paper, please run:
```bash
# For MAQ based methods
# Section 5.2.2: MAQ+RLPD in door task (with macro action length=9 and codebook size=16)
./scripts/train.sh --method "MAQ+RLPD" --sequence_length 9 --codebook_size 16 --environment "door-human-v1" --seed 1

# Section 5.2.2: MAQ+IQL in door task (with macro action length=9 and codebook size=16)
./scripts/train.sh --method "MAQ+IQL" --sequence_length 9 --codebook_size 16 --environment "door-human-v1" --seed 1

# Section 5.2.2: MAQ+DSAC in door task (with macro action length=8 and codebook size=8)
./scripts/train.sh --method "MAQ+DSAC" --sequence_length 8 --codebook_size 8 --environment "door-human-v1" --seed 1

# For baseline methods
# Section 5.2.2: SAC in door task
./scripts/train.sh --method "SAC" --environment "door-human-v1" --seed 1

# Section 5.2.2: RLPD in door task
./scripts/train.sh --method "RLPD" --environment "door-human-v1" --seed 1

# Section 5.2.2: IQL in door task
./scripts/train.sh --method "IQL" --environment "door-human-v1" --seed 1
```

For detailed parameters, please refer to the following table:

| Parameter | Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| **Method** | `--method` | Training method (MAQ+RLPD, MAQ+IQL, MAQ+DSAC, SAC, RLPD, IQL) | `MAQ+RLPD` |
| **Environment** | `-env`, `--environment` | D4RL environment name (e.g., door-human-v1, hammer-human-v1, pen-human-v1, relocate-human-v1) | `door-human-v1` |
| **Sequence Length** | `-seqlen`, `--sequence_length` | Macro action length | `9`|
| **Codebook Size** | `-cbsz`, `--codebook_size` | VQ-VAE codebook size | `16`|
| **Seed** | `-s`, `--seed` | Random seed | `1`|
| **Training Source** | `-trs`, `--training_source` | Training dataset path (relative to `offline_data/`) | `""` (Defaults to environment dataset provided by D4RL)|
| **Testing Source** | `-tes`, `--testing_source` | Testing dataset path (relative to `offline_data/`) | `""` (Defaults to environment dataset provided by D4RL)|
| **Tag** | `-t`, `--tag` | Tag for the experiment (e.g., date) | `""`|
| **Auto Evaluate** | `--auto_evaluate` | Automatically evaluate the trained model after training | `False`|



#### Training MAQ+RLPD and MAQ+DSAC

#### Training MAQ+IQL and IQL

#### Training SAC

#### Training RLPD


### Train Baselines
// RLPD, IQL, SAC
// scripts/train.sh things here

#### SAC
// need to metion the method using stablebaselines3 and the log file stores as...


#### IQL
// need to mention the method using ... and the log file stores as...


#### RLPD
// need to specify the instructions here, and need to mention that we have make small changes on RLPD about the dataset loading



### Train Macro Action Quantization with Different Macro Action Length and Codebook Size
// lazy_rerun.sh things here
// user can change the combination of macro action length and codebook size and methods in lazy_rerun.sh

## Evaluation
// introduce the evaluation_results
// introduce DTW/WD? 

### Evaluate Macro Action Quantization
// MAQ+RLPD, MAQ+IQL, MAQ+DSAC
// scripts/evaluate.sh things here

### Evaluate Baselines
// RLPD, IQL, SAC
// scripts/evaluate.sh things here

