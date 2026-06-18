import h5py
import torch
import pickle
from pathlib import Path
import os

folder = 'C:/Users/Michael/Downloads'
mat_file = os.path.join(folder, "cp_arls_lev_result_R80S2-25E8544.mat")
folder1 = 'F:/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files'
save_path = os.path.join(folder1, "A_CP_M41_R80.pkl")

with h5py.File(mat_file, "r") as f:
    
    relres = f["relres"][:].squeeze().item()
    finalfit = f["finalfit"][:].squeeze().item()
    rank = int(f["R"][:].squeeze().item())

    # CP weights
    weights = torch.tensor(
        f["#refs#/c/lambda"][:].squeeze(),
        dtype=torch.float64
    )

    # CP factor matrices
    factors = []

    u = f["#refs#/c/u"]

    for ref in u[0]:

        factor = torch.tensor(
            f[ref][:].T,          # transpose to (41,80)
            dtype=torch.float64
        )

        factors.append(factor)

norm_A_CP = (weights, factors)

output_data = {
    "relative_error": relres,
    "finalfit": finalfit,
    "norm_A_CP": norm_A_CP
}

with open(save_path, "wb") as file:
    pickle.dump(output_data, file)

print(f"Saved: {save_path}")
print("weights shape:", weights.shape)

for i, F in enumerate(factors):
    print(f"factor {i}: {F.shape}")