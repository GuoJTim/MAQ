import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import sys
import pickle
import argparse

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import D4RLGameDataset
from modules import VectorQuantizedVAE

def load_trajectories(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description='VQ-VAE t-SNE visualization')
    parser.add_argument('--dataset', type=str, default='door-human-v1_test_seed1_ratio0.9.pkl',
                        help='Dataset file name in offline_data/')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to trained model .pth file')
    parser.add_argument('--k', type=int, default=16, help='Codebook size')
    parser.add_argument('--dim', type=int, default=32, help='Hidden dimension') 
    parser.add_argument('--seqlen', type=int, default=4, help='Sequence length')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
    
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join('../offline_data', args.dataset)
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    trajectories = load_trajectories(data_path)
    
    # Initialize Model dims from first trajectory
    sample_obs = trajectories[0]['observations'][0]
    sample_acts = trajectories[0]['actions']
    if len(sample_acts.shape) == 1: sample_acts = sample_acts.reshape(-1, 1)
    
    obs_dim = sample_obs.shape[0]
    act_dim = sample_acts.shape[1]
    
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}, Seq len: {args.seqlen}")
    
    model = VectorQuantizedVAE(state_dim=obs_dim, seq_len=args.seqlen, K=args.k, dim=args.dim, output_dim=act_dim).to(device)
    model.double() 
    
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("Warning: No model path provided or file does not exist. Using random initialization.")

    # Use only the first trajectory for detailed analysis
    traj = trajectories[0]
    obs = traj['observations']
    # We apply VQVAE on macro actions, so we use obs as state condition.
    
    n_steps = obs.shape[0] - args.seqlen + 1 # valid starting steps for seqlen
    if n_steps <= 0:
        print("Trajectory too short for sequence length.")
        return

    print(f"Processing first trajectory with {n_steps} valid steps (total steps {obs.shape[0]}). Codebook size K={args.k}")

    generated_actions = []
    code_indices = []
    progress_values = []
    
    model.eval()
    
    # Create batch of all K indices
    all_k_indices = torch.arange(args.k, device=device) # (K,)
    
    print("Generating macro-actions for all codes at each step...")
    with torch.no_grad():
        for i in range(n_steps):
            # Current state
            state = obs[i]
            # Repeat state K times to match all_k_indices
            state_batch = torch.tensor(state, dtype=torch.float64).to(device).unsqueeze(0).repeat(args.k, 1) # (K, obs_dim)
            
            # Decode
            # forward_decoder returns numpy array (K, seq_len, act_dim)
            decoded_acts = model.forward_decoder(state_batch, all_k_indices)
            
            # Flatten to (K, seq_len * act_dim)
            flat_acts = decoded_acts.reshape(args.k, -1)
            
            generated_actions.append(flat_acts)
            
            # Store metadata
            code_indices.append(np.arange(args.k))
            
            # Progress value for this step
            p = i / float(n_steps)
            progress_values.append(np.full(args.k, p))

    if not generated_actions:
        print("No actions generated.")
        return

    all_gen_actions = np.concatenate(generated_actions, axis=0) # (T*K, flat_dim)
    all_codes = np.concatenate(code_indices, axis=0) # (T*K,)
    all_progress = np.concatenate(progress_values, axis=0) # (T*K,)
    
    print(f"Total generated macro-actions: {all_gen_actions.shape}")

    # t-SNE
    print("Running t-SNE...")
    # Subsample if necessary (though T*K might be small enough: 500 steps * 16 codes = 8000)
    if all_gen_actions.shape[0] > 10000:
        indices = np.random.choice(all_gen_actions.shape[0], 10000, replace=False)
        tsne_data = all_gen_actions[indices]
        tsne_codes = all_codes[indices]
        tsne_progress = all_progress[indices]
    else:
        tsne_data = all_gen_actions
        tsne_codes = all_codes
        tsne_progress = all_progress
        
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(tsne_data)
    
    # Plot 1: Color by Progress, Marker by Code Index
    print("Plotting by Code Index (Shape) and Progress (Color)...")
    plt.figure(figsize=(14, 12))
    
    # Define 5 progress bins and corresponding colors
    prog_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    prog_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    # Distinct colors for progress
    prog_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Define markers for Codes (support up to ~20 codes)
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', '8', 'P', '1', '2', '3']
    if args.k > len(available_markers):
        print(f"Warning: K={args.k} exceeds available distinct markers {len(available_markers)}. Some markers will repeat.")
        markers = available_markers * (args.k // len(available_markers) + 1)
    else:
        markers = available_markers
        
    # Legend handles for Progress (Colors)
    from matplotlib.lines import Line2D
    prog_handles = [Line2D([0], [0], marker='o', color='w', label=l, markerfacecolor=c, markersize=10) for c, l in zip(prog_colors, prog_labels)]
    
    # Legend handles for Codes (Markers)
    # Use neutral color for marker legend
    code_handles = [Line2D([0], [0], marker=markers[k], color='w', label=f'Code {k}', markerfacecolor='k', markersize=10) for k in range(args.k)]
    
    for k in range(args.k):
        code_mask = (tsne_codes == k)
        if not np.any(code_mask):
            continue
            
        marker = markers[k]
        
        # For each code, plot points with different colors based on progress
        for bus_idx in range(5):
            prog_mask = (tsne_progress >= prog_bins[bus_idx]) & (tsne_progress < prog_bins[bus_idx+1])
            combined_mask = code_mask & prog_mask
            
            if np.any(combined_mask):
                plt.scatter(embedded[combined_mask, 0], embedded[combined_mask, 1], 
                            color=prog_colors[bus_idx], marker=marker, alpha=0.6, s=20) # Increased size for visibility

    plt.title(f't-SNE VQ-VAE Generated Actions (Color=Progress, Shape=Code) K={args.k}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    # Add first legend (Progress)
    first_legend = plt.legend(handles=prog_handles, title="Progress", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    
    # Add second legend (Codes)
    plt.legend(handles=code_handles, title="Codes", bbox_to_anchor=(1.05, 0.8), loc='upper left', ncol=1 if args.k < 15 else 2)

    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    output_dir = '../log_tmp' if os.path.exists('../log_tmp') else '.'
    output_png = os.path.join(output_dir, 'vqvae_decoded_codes_tsne.png')
    plt.savefig(output_png)
    print(f"Saved plot to {output_png}")
    
    # Plot 2: Color by Progress (Optional but helpful context)
    # plt.figure(figsize=(12, 10))
    # plt.scatter(embedded[:, 0], embedded[:, 1], c=tsne_progress, cmap='viridis', alpha=0.5, s=10)
    # plt.title('t-SNE VQ-VAE Generated Actions (Colored by Progress)')
    # plt.colorbar(label='Trajectory Progress')
    # plt.savefig(os.path.join(output_dir, 'vqvae_decoded_progress_tsne.png'))

if __name__ == "__main__":
    main()
