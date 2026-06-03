import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import torch

from my_readwrite import my_read_solution, my_get_soltn_file_names

path = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/111 New Training Data/sphomruns training half"


# Load solutions into array

start_time = time.time()
names = my_get_soltn_file_names(path)

sols = []
for i, name in enumerate(names, 1):
    sol, _ = my_read_solution(name)   # shape (1, M^3)
    sols.append(sol)
    if i % 100 == 0:
        print(f"Loaded {i}/{len(names)} solutions")

D = np.vstack(sols)
print("Final dataset shape:", D.shape)
print("Data loading took", time.time() - start_time, "seconds")


# Compute singular values

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sv_start = time.time()
if device == "cuda":
    # Use PyTorch + cuSOLVER
    D_torch = torch.tensor(D, dtype=torch.float32, device=device)
    S_torch = torch.linalg.svdvals(D_torch)
    S = S_torch.cpu().numpy()  # move back to CPU for saving/plotting
else:
    # CPU NumPy fallback
    S = np.linalg.svd(D, full_matrices=False, compute_uv=False)
print("SVD complete in", time.time() - sv_start, "seconds")


# Save singular values

with open("singular_values(1).txt", "w") as f:
    f.write("index singular_value\n")
    for i, s in enumerate(S):
        f.write(f"{i} {s}\n")

print("Singular values saved to singular_values.txt")

### -------------------------
### Compute singular values (cheap method)
### -------------------------
##sv_start = time.time()
##
##N, M = D.shape
##print(f"Matrix shape: {N} x {M}")
##
##if N <= M:
##    # Compute Gram matrix (N x N)
##    G = D @ D.T
##
##    # Compute eigenvalues (symmetric matrix → use eigh)
##    eigvals = np.linalg.eigvalsh(G)
##
##    # Remove tiny negative numerical noise
##    eigvals = np.clip(eigvals, 0, None)
##
##    # Singular values are sqrt of eigenvalues
##    S = np.sqrt(eigvals)
##
##else:
##    # Fallback (rare case)
##    S = np.linalg.svd(D, full_matrices=False, compute_uv=False)
##
### Sort descending
##S = np.sort(S)[::-1]
##
##print("Singular values computed in", time.time() - sv_start, "seconds")
##
##
##np.savetxt(
##    "singular_values.txt",
##    np.column_stack((np.arange(len(S)), S)),
##    header="index singular_value",
##    fmt=["%d", "%.10e"]
##)
##
##print("Singular values saved to singular_values.txt")
