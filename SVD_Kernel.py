import torch
import pickle
import tensorly as tl
from tensorly import unfold
import matplotlib.pyplot as plt
import os

# Set tensorly backend to pytorch
tl.set_backend('pytorch')

# File paths
input_file_path = '/pscratch/sd/m/mdevaney/Boltzmann'
input_filename = 'norm_A_CP_M41_R400.pkl'
input_full_path = os.path.join(input_file_path, input_filename)

output_folder = 'SVD Results'
os.makedirs(output_folder, exist_ok=True)

# Load CP-decomposed tensor
with open(input_full_path, "rb") as file:
    data = pickle.load(file)

norm_A_CP = data['norm_A_CP']
weights = norm_A_CP[0]
factors = norm_A_CP[1]

# Reconstruct full tensor
full_tensor = tl.cp_to_tensor((weights, factors))

# Loop over modes 0, 1, 2
for mode in range(3):
    # Unfold along mode
    unfolded = unfold(full_tensor, mode=mode)

    # Compute SVD (full_matrices=False)
    U, S, Vh = torch.linalg.svd(unfolded, full_matrices=False)

    # Save singular values as pickle
    svd_data = {
        'mode': mode,
        'singular_values': S.cpu().numpy()
    }
    svd_pickle_filename = f'singular_values_mode{mode}.pkl'
    with open(os.path.join(output_folder, svd_pickle_filename), "wb") as f:
        pickle.dump(svd_data, f)

    # Plot and save figure
    plt.figure(figsize=(8, 5))
    plt.semilogy(S.cpu().numpy(), marker='o')
    plt.title(f'Singular Values of Mode-{mode} Unfolded Tensor')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.grid(True)
    plt.tight_layout()

    plot_filename = f'singular_values_mode{mode}.png'
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()
