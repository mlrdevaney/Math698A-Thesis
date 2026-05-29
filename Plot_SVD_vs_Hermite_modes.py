import my_readwrite
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

##path = "D:/Devaney SSD Backup/AFIT/Nguyen_1/Data/111 New Training Data"
path = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/111 New Training Data"
filename_grids = os.path.join(path, 'take3_M41 good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_grids.dat')
grids_cap_u, grids_cap_v, grids_cap_w, grids_u_tmp, grids_v_tmp, grids_w_tmp = my_readwrite.read_grids(filename_grids)

filename_nodes = os.path.join(path, 'take3_M41 good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat')
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename_nodes)

mu = grids_cap_u[0, 0]
grids_u = grids_u_tmp[:, 0:mu]
grids_v = grids_v_tmp[:, 0:mu]
grids_w = grids_w_tmp[:, 0:mu]

u = nodes_u[0, :]
D3 = nodes_u.shape[1]
Nv = int(np.cbrt(D3))
mid = Nv // 2  # center index







with open('SVD_U_M41.pkl', 'rb') as svd_file:
    U = pickle.load(svd_file)
U = U.double().cpu().numpy()
U = U.reshape((Nv, Nv, Nv, 5))
u1 = u.reshape((Nv, Nv, Nv))[:, mid, mid]

for mode_index in range(5):
    if mode_index == 2 or mode_index ==3:
        continue
    f = U[:, mid, mid, mode_index]

    plt.plot(u1, f, marker='x', label=f"SVD Mode {mode_index}")
plt.title("SVD Mode Projections Along u")
plt.xlabel("u")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("SVD_modes.pdf")
plt.show()




T = 0.2  # set your temperature
# --- Hermite polynomials ---
H = [
    np.ones_like(u),   # H0
    u,                 # H1
    u**2 - 1           # H2
]
# --- Maxwellian weight ---
weight = (T * np.pi)**(-0.5) * np.exp(-u**2 / T)
# --- Compute coefficients ---
c = [
    np.sum(H[0] * weight),
    np.sum(H[1] * weight),
    np.sum(H[2] * weight)
]
# --- Reconstruct Hermite projection ---
for idx in range(3):
    f_H_i = c[idx] * H[idx] * weight
    plt.plot(u, f_H_i, marker='x', label=f"Hermite Polynomial: {idx}")
plt.title("Hermite (Maxwellian-weighted) projections along u")





plt.xlabel("u")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.savefig("Hermite_modes.pdf")
plt.show()
