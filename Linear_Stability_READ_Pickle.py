import pickle
import sys
import numpy as np
import tensorly as tl
from tensorly import shape, reshape
sys.path.append('/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/')
from BoltzCol2 import boltzcol2
sys.path.append('/Volumes/MD Research/BoltzmannData/NN0DBlzm00/')
from my_readwrite import my_read_solution_trim

# Path to your .pkl file
pkl_path = "/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/Linear_Stability.pkl"

# Load the data
with open(pkl_path, "rb") as f:
    eigen_data = pickle.load(f)

rank = 15
MM = 15
Mtrim = 0
path = '/Volumes/MD Research/BoltzmannData/nodes_collection/solcollfiles15/'
id_filename = 'TestCP15_2kc1su1sv1sw3NXU15MuUU15MvVU15MwWU_time0.0004000000_SltnColl.dat'

# let us load initial data
sol, solsize = my_read_solution_trim(path+id_filename,MM, Mtrim)  # this is the first file

H = eigen_data['positive_real_eigvecs']
f = sol.reshape(-1, 1)
HHTf = H @ (H.T @ f)
HHTf = HHTf.reshape(MM, MM, MM)
mu = 1
Q = np.reshape(boltzcol2(sol, sol, MM, rank), (MM, MM, MM))
filter1 = Q - mu * HHTf
print(filter1)

### ALL
##print("All eigenvalues:")
##print(eigen_data['eigvals'])
##
##print("All eigenvectors:")
##print(eigen_data['eigvecs'])
##
### POSITIVE
##print("\nEigenvalues with positive real parts:")
##print(eigen_data['positive_real_eigvals'])
##
##print("\nEigenvectors for pos real:")
##print(eigen_data['positive_real_eigvecs'])
##
### NEGATIVE
##print("\nEigenvalues with negative real parts:")
##print(eigen_data['negative_real_eigvals'])
##
##print("\nEigenvectors for neg real:")
##print(eigen_data['negative_real_eigvecs'])
##
##
### ZERO
##print("\nEigenvalues with zero real parts:")
##print(eigen_data['zero_real_eigvals'])
##
##print("\nEigenvectors for zero real:")
##print(eigen_data['zero_real_eigvecs'])
