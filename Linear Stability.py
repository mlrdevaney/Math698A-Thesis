import sys
import pickle
import numpy as np
import tensorly as tl
import time
import csv
import datetime
from tensorly.decomposition import parafac
from tensorly import norm, reshape, cp_to_tensor, shape, zeros
sys.path.append('/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD')
#from BoltzCol1 import boltzcol1
from BoltzCol2 import boltzcol2
sys.path.append('/Volumes/MD Research/BoltzmannData/TTBltzmn')
from my_readwrite import my_read_sol_coll_trim, read_nodes
sys.path.append('/Volumes/MD Research/BoltzmannData/ROM_simulations/cleaned_100_MakePicklefiles_SDV/')
from my_distributions import maxwellian
path = '/Volumes/MD Research/BoltzmannData/nodes_collection/solcollfiles15/'
id_filename = 'TestCP15_2kc1su1sv1sw3NXU15MuUU15MvVU15MwWU_time0.0027000000_SltnColl.dat'
file = path + id_filename
filename = '/Volumes/MD Research/BoltzmannData/nodes_collection/TestCP15_1su1sv1sw15MuUU15MvVU15MwWU_nodes.dat'
MM = 15
MM3 = MM ** 3  # Compute once to avoid recomputation
Mtrim = 0
#sol and soltrm are identical when Mtrim = 0
soltrm1, colltrm, solsze = my_read_sol_coll_trim(file, MM, Mtrim)
nodes_u, nodes_v, nodes_w, nodes_gwts = read_nodes(filename)
# compute moments
moments=np.zeros((1,5))
moments[0,0] = np.sum(soltrm1 * nodes_gwts)
moments[0,1] = np.sum(soltrm1 * nodes_u * nodes_gwts) / moments[0,0]
moments[0,2] = np.sum(soltrm1 * nodes_v * nodes_gwts) / moments[0,0]
moments[0,3] = np.sum(soltrm1 * nodes_w * nodes_gwts) / moments[0,0]
moments[0,4] = np.sum(soltrm1 * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
                                              + (nodes_v - moments[0,2]) ** 2
                                              + (nodes_w - moments[0,3]) ** 2))\
                   / moments[0,0] / 3.0 * 2.0
##ffm = soltrm1 + maxwellian(moments,nodes_u,nodes_v,nodes_w)
##delta_f = soltrm1 - maxwellian(moments,nodes_u,nodes_v,nodes_w)
########################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################
fm = maxwellian(moments,nodes_u,nodes_v,nodes_w)
fm = [fm, 0]
L = np.zeros((MM3, MM3))
E = np.eye(MM3).reshape((MM3, MM, MM, MM))
rank = 25
for j in range(MM3):
    e = [E[j], 0]
    L[j, :] = boltzcol2(e, fm, MM, 1, rank)
eigvals, eigvecs = np.linalg.eig(L)
positive_real_eigvals = eigvals[np.real(eigvals) > 0]
negative_real_eigvals = eigvals[np.real(eigvals) < 0]
zero_real_eigvals = eigvals[np.real(eigvals) == 0]

output_path = "/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/Linear Stability.txt"

with open(output_path, "w") as f:
    f.write("All Eigenvalues:\n")
    for val in eigvals:
        f.write(f"{val}\n")

    f.write("\nAll Eigenvectors (each column is an eigenvector):\n")
    for vec in eigvecs.T:  # columns are eigenvectors
        f.write(f"{vec}\n")

    # Indices for categories
    pos_idx = np.where(np.real(eigvals) > 0)[0]
    neg_idx = np.where(np.real(eigvals) < 0)[0]
    zero_idx = np.where(np.real(eigvals) == 0)[0]

    f.write("\nEigenvalues with Positive Real Parts:\n")
    for i in pos_idx:
        f.write(f"{eigvals[i]}\n")

    f.write("\nCorresponding Positive Eigenvectors:\n")
    for i in pos_idx:
        f.write(f"{eigvecs[:, i]}\n")

    f.write("\nEigenvalues with Negative Real Parts:\n")
    for i in neg_idx:
        f.write(f"{eigvals[i]}\n")

    f.write("\nCorresponding Negative Eigenvectors:\n")
    for i in neg_idx:
        f.write(f"{eigvecs[:, i]}\n")

    f.write("\nEigenvalues with Real Part Zero:\n")
    for i in zero_idx:
        f.write(f"{eigvals[i]}\n")

    f.write("\nCorresponding Zero Eigenvectors:\n")
    for i in zero_idx:
        f.write(f"{eigvecs[:, i]}\n")

# Prepare data to pickle
eigen_data = {
    'eigvals': eigvals,
    'eigvecs': eigvecs,
    'positive_real_eigvals': eigvals[pos_idx],
    'positive_real_eigvecs': eigvecs[:, pos_idx],
    'negative_real_eigvals': eigvals[neg_idx],
    'negative_real_eigvecs': eigvecs[:, neg_idx],
    'zero_real_eigvals': eigvals[zero_idx],
    'zero_real_eigvecs': eigvecs[:, zero_idx],
}

# Pickle output
pkl_output_path = "/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/Linear_Stability.pkl"
with open(pkl_output_path, "wb") as f:
    pickle.dump(eigen_data, f)

#print(f"Pickle file saved to {pkl_output_path}")
print(f"Finished at {datetime.datetime.now()}")

