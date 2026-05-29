import pickle

# Path to the .pkl file you created
filepath = '/Volumes/Devaney SSD/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/'
filename = 'norm_A_CP_M41_R20.pkl'
pkl_file = filepath + filename

# Open and load the pickle file
with open(pkl_file, 'rb') as file:
    data = pickle.load(file)

# Now 'data' is a dictionary with keys 'relative_error' and 'norm_A_CP'
err_A = data['relative_error']
exec_time = data['execution_time']
norm_A_CP = data['norm_A_CP']

# Print out the relative error
print(f"Relative error: {err_A * 100:.4f} %")
print(f"Execution time: {exec_time} seconds")

##import pickle
##import os
##import numpy as np
##import scipy.io
##import tensorly as tl
##from tensorly import norm, cp_tensor, shape, reshape
##from tensorly import max, min
##from tensorly.decomposition import parafac
##### Path to the saved .pkl file
####file_path = '/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/'
####filename = 'norm_A_CP_M15_R15.pkl'
####file_loc = file_path + filename
####
##### Load the data
####with open(file_loc, "rb") as file:
####    norm_A_CP = pickle.load(file)
##
##file_path = '/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/' 
##filename = 'A15cp_factorsR160Acc95.mat' # ensure this is a matlab ".mat" file
##file_loc = file_path + filename
##mat = scipy.io.loadmat(file_loc) # we load tensor from matlab file
##weights = [mat['Alam']]
##factors = [mat['A1'], mat['A2'], mat['A3'], mat['A4'], mat['A5'], mat['A6']]  # we read the tensor data as an array
##
##
##file_path = '/Volumes/MD Research/BoltzmannData/TTBltzmn/' 
##filename = 'A6DtensorM15_matlab.mat' # ensure this is a matlab ".mat" file
##file_loc = file_path + filename
##tensor_data = scipy.io.loadmat(file_loc) # we load tensor from matlab file
##tensor_array = tensor_data['Atensor'] # we read the tensor data as an array
##A = tl.tensor(tensor_array)
##print(A[1,3,9,7,7,7])
##print(A[7,7,7,1,3,9])
### We compute the error of the CP reconstructed tensor of A against the original A
##CP_A = (reshape(weights[0], (160, )), factors)
##recon_A = tl.cp_to_tensor(CP_A)
##print(recon_A[1,3,9,7,7,7])
##print(recon_A[7,7,7,1,3,9])
### Below we compute the L2 norm for error, increase rank to get closer to 0%
##err_A = norm(recon_A - A, 2) / norm(A, 2)
##print("shape recon A", shape(recon_A))
##print("shape A", shape(A))
##print('max recon A', max(recon_A))
##print('max A', max(A))
##print('min recon A', min(recon_A))
##print('min A', min(A))
##print("Percent error for A: ", err_A * 100, "%")
##save_path = 'norm_A_CP_M15_R160.pkl'
##with open(save_path, "wb") as file:
##    pickle.dump(CP_A, file)
##
### Prints file path
##print(f"Normalized CP decomposition saved to:\n{os.path.abspath(save_path)}")
##
##
