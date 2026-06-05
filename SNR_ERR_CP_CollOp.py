import numpy as np
from numpy import linalg
import os
import re
import my_readwrite
import Utilities
from pathlib import Path
from my_distributions import maxwellian
from BoltzCol2 import boltzcol2   

home_folder = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1"
##datatype = "111TrD"
datatype = "222TsD"

# === Load grid/node data ===
filename_grids = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat')
grids_cap_u, grids_cap_v, grids_cap_w, grids_u_tmp, grids_v_tmp, grids_w_tmp = my_readwrite.read_grids(filename_grids)

filename_nodes = os.path.join(home_folder, 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat')
nodes_u, nodes_v, nodes_w, ngwts = my_readwrite.read_nodes(filename_nodes)

MM = 41
Mtrim = 0

# === Load test data ===
sol_pkl = os.path.join(home_folder, f"Data/Non-norm_{datatype}_delta_f_data_MM_41_MT_0_CT_1.0.pkl")
coll_pkl = os.path.join(home_folder, f"Data/Non-norm_{datatype}_coll_data_MM_41_MT_0_CT_1.0.pkl")

sol_data_test = Utilities.LoadPickleSolData(sol_pkl)   # (B, N)
coll_data_test = Utilities.LoadPickleSolData(coll_pkl) # (B, N)

# Determine file type
filetype = "zeros_test"
match = re.search(r'Non-norm_(\d+)T', sol_pkl)
if match:
    filenum = int(match.group(1))
    filetype = "training" if filenum == 111 else "testing"

# Output file
out_path = f"boltzcol2_vs_truth_{filetype}.txt"
out_file = open(out_path, "w")
out_file.write("||Q_true||_2   SNR_dB_Delf      L2_ERR_Delf   SNR_dB_ff      L2_ERR_ff\n")

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

fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
ngwts = ngwts.reshape(-1)

print("Computing collision operator using boltzcol2...")

for i in range(sol_data_test.shape[0]):

    df = sol_data_test[i:i+1]        # shape (1, N)
    Q_true = coll_data_test[i]     # shape (N,)

    # === Compute approximation ===
    Q_pred_1 = boltzcol2(df+2*fm, df, MM, sigma_p=50)  # adjust rank if needed
    Q_pred_2 = boltzcol2(df+fm, df+fm, MM, sigma_p=50)  # adjust rank if needed

    # === Metrics ===
    q_norm = np.linalg.norm(Q_true)

    signal_energy = np.sum(Q_true**2 * ngwts)
    truth = np.linalg.norm(Q_true)
    noise_energy_1  = np.sum((Q_true - Q_pred_1)**2 * np.tensor(ngwts))
    noise_energy_2  = np.sum((Q_true - Q_pred_2)**2 * np.tensor(ngwts))

    snr_db_1 = 10 * np.log10((signal_energy + eps) / (noise_energy_1 + eps))
    snr_db_2 = 10 * np.log10((signal_energy + eps) / (noise_energy_2 + eps))
    L2_1 = np.linalg.norm(Q_true - Q_pred_1) / (truth + eps)
    L2_2 = np.linalg.norm(Q_true - Q_pred_2) / (truth + eps)

    out_file.write(f"{q_norm:.8e} {snr_db_1:.8f} {L2_1:.8f} {snr_db_2:.8f} {L2_2:.8f}\n")

    if i % 100 == 0:
        print(f"Finished writing {i} entries of data.")

out_file.close()
print(f"Wrote {out_path}")
