import sys
import pickle
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import os

# Load the normalized CP decomposition
save_path = 'norm_A_CP_M41_R400.pkl'
with open(save_path, "rb") as file:
    norm_A_CP = pickle.load(file)

# Extract weights and factor matrices
weights = norm_A_CP[0]  # Shape: (rank,)
factors = norm_A_CP[1]  # List of 6 factor matrices

# Convert factor matrices to NumPy arrays
A_tensor = [np.array(matrix) for matrix in factors]

# Validate shapes
rank = len(weights)
assert all(A.shape[1] == rank for A in A_tensor), "Inconsistent rank dimension in CP factors"

# Compute rank-1 tensors (M, M, M, rank)
rank1_tensors_set1 = (
    A_tensor[0][:, None, None, :] *  # u
    A_tensor[1][None, :, None, :] *  # v
    A_tensor[2][None, None, :, :]    # w
)

rank1_tensors_set2 = (
    A_tensor[3][:, None, None, :] *  # u'
    A_tensor[4][None, :, None, :] *  # v'
    A_tensor[5][None, None, :, :]    # w'
)

# Scale both by weights
rank1_tensors_set1 *= weights[None, None, None, :]
rank1_tensors_set2 *= weights[None, None, None, :]

# Reshape to (rank, M^3)
vectorized_set1 = rank1_tensors_set1.reshape(-1, rank).T
vectorized_set2 = rank1_tensors_set2.reshape(-1, rank).T

# Combine both sets: shape (2*rank, M^3)
vectorized_tensors = np.vstack((vectorized_set1, vectorized_set2))

# Perform SVD
U, S, Vh = np.linalg.svd(vectorized_tensors, full_matrices=False)

# Save SVD components to a compressed .npz file
np.savez('basis_vectors_svd.npz', U=U, S=S, Vh=Vh)

# Plot singular values
plt.figure(figsize=(8, 5))
plt.semilogy(np.arange(1, len(S)+1), S, 'bo-')
plt.title("Singular Value Decay of CP-Decomposed A")
plt.xlabel("Index")
plt.ylabel("Singular Value (log scale)")
plt.grid(True)
plt.tight_layout()
plt.savefig("singular_value_decay.png")
plt.close()

print("SVD saved to 'basis_vectors_svd.npz' and plot saved to 'singular_value_decay.png'")
