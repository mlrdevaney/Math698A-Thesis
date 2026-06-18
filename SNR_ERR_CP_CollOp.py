import numpy as np
import os
import re
import my_readwrite
import Utilities
from my_distributions import maxwellian
from cpQ import cpQ
import tensorly as tl
import pickle
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cuda'
# device = "cpu"

tl.set_backend('pytorch')

home_folder = "F:/AFIT-ARCH/Nguyen_1"

# === Load grid/node data ===
filename_grids = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat')
grids_cap_u, grids_cap_v, grids_cap_w, grids_u_tmp, grids_v_tmp, grids_w_tmp = my_readwrite.read_grids(filename_grids)

filename_nodes = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat')
nodes_u, nodes_v, nodes_w, ngwts = my_readwrite.read_nodes(filename_nodes)

MM = 41
Mtrim = 0

# === Load test data ===
df_pkl = os.path.join(home_folder, f"Data/Non-norm_TOTAL_delta_f_data_MM_41_MT_0_CT_1.0.pkl")
coll_pkl = os.path.join(home_folder, f"Data/Non-norm_TOTAL_coll_data_MM_41_MT_0_CT_1.0.pkl")

df_data_test = Utilities.LoadPickleSolData(df_pkl)   # (B, N)
coll_data_test = Utilities.LoadPickleSolData(coll_pkl) # (B, N)

eps = 1e-12

moments=np.zeros((1,5))
##                moments[0,0] = np.sum(sol * nodes_gwts)
##                moments[0,1] = np.sum(sol * nodes_u * nodes_gwts) / moments[0,0]
##                moments[0,2] = np.sum(sol * nodes_v * nodes_gwts) / moments[0,0]
##                moments[0,3] = np.sum(sol * nodes_w * nodes_gwts) / moments[0,0]
##                moments[0,4] = np.sum(sol * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
##                                                              + (nodes_v - moments[0,2]) ** 2
##                                                              + (nodes_w - moments[0,3]) ** 2))\
##                                   / moments[0,0] / 3.0 * 2.0
moments[0,0]=1
moments[0,1]=0
moments[0,2]=0
moments[0,3]=0
moments[0,4]=0.2

fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1) # verified to be accurate as compared to NN work
ngwts = ngwts.reshape(-1)
file_loc = 'F:/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/norm_A_CP_M41_R400.pkl'
with open(file_loc, "rb") as file:
    norm_A_CP = pickle.load(file)
norm_A_CP = norm_A_CP['norm_A_CP']
# def double_rank(norm_A_CP):
#     omega, factors = norm_A_CP
#     f1, f2, f3, f4, f5, f6 = factors

#     # swapped ordering
#     factors_swap = [f4, f5, f6, f1, f2, f3]

#     # double weights
#     omega_new = 0.5 * torch.cat((omega, omega), dim=0)

#     factors_new = []
#     for f, fs in zip(factors, factors_swap):
#         factors_new.append(
#             torch.cat((f, fs), dim=1)   # (mode_dim, 2R)
#         )

#     return (omega_new, factors_new)
# (omega1, factors1) = double_rank(norm_A_CP)
# output_data = {
#     "relative_error": None,
#     "finalfit": None,
#     "norm_A_CP": (omega1, factors1)
# }

# with open("F:/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/double_A_CP_M41_R800.pkl", "wb") as file:
#     pickle.dump(output_data, file)

# exit()
A_weights = norm_A_CP[0].detach().clone().to(device=device, dtype=torch.float64)
A_factors = [factor.detach() for factor in norm_A_CP[1]]
sigma = len(A_weights) # rank_A
list_A = [factor_matrix.transpose(0, 1).to(torch.float64).to(device=device) for factor_matrix in A_factors]
# print(list_A[0].T)
# plt.plot(list_A[0].T, '--', label=f"--")

# plt.xlabel("")
# plt.ylabel("")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.ioff()
# plt.show(block=True)
# exit()
rank = 10

# for anchor in range(41):
# Output file
out_path = os.path.join('F:/BoltzmannData/Boltzmann_Thesis_MD/SNR', f"cpQ_vs_truth.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
norm = np.linalg.norm
sum_ = np.sum
log10 = np.log10
# lines = []
# lines.append("||Q_true||_2   SNR_dB_Delf      L2_ERR_Delf\n")
with open(out_path, "w") as out_file: 
    out_file.write("||Q_true||_2 SNR_dB_Delf L2_ERR_Delf\n")

    print("Computing collision operator using cpQ...")
    # for i in range(5): #for testing purposes, we can just do this for the first 10 entries of the test data. Change to df_data_test.shape[0] for full test set.
    for i in range(df_data_test.shape[0]):
        df = df_data_test[i:i+1]        # shape (1, N), verified to be accurate as compared to NN work
        Q_true = coll_data_test[i]     # shape (N,), verified to be accurate as compared to NN work
        # f = df + fm
        ffm = df + (2 * fm)
        # print(np.linalg.norm(fm))

        # === Compute approximation ===
        Q_pred_1 = cpQ(ffm, df, A_weights, list_A, MM, device, sigma, rank, g_rank = None).cpu().numpy()
        # Q_pred_2 = cpQ(f, f, A_weights, list_A, MM, device, sigma, rank, g_rank = None).cpu().numpy()

        # === Metrics ===
        q_norm = norm(Q_true)
        
        err1 = Q_true - Q_pred_1
        # err2 = Q_true - Q_pred_2

        signal_energy = sum(Q_true**2 * ngwts)
        noise_energy_1 = sum(err1 * err1 * ngwts)
        # noise_energy_2 = sum(err2 * err2 * ngwts)

        L2_1 = norm(err1) / (q_norm + eps)
        # L2_2 = norm(err2) / (q_norm + eps)

        snr_db_1 = 10 * log10((signal_energy + eps) / (noise_energy_1 + eps))
        # snr_db_2 = 10 * log10((signal_energy + eps) / (noise_energy_2 + eps))
        out_file.write( f"{q_norm:.8e} {snr_db_1:.8f} {L2_1:.8f}\n")
        # lines.append(
        #     f"{q_norm:.8e} {snr_db_1:.8f} {L2_1:.8f} ")
        #     f"{snr_db_2:.8f} {L2_2:.8f}\n"
        # )

        if i % 10 == 0 and i <= 10:
            print(f"Finished writing {i}/{df_data_test.shape[0]} entries of data.")
    # with open(out_path, "w") as out_file:
    #     out_file.writelines(lines)
    print(f"Wrote {out_path}")

