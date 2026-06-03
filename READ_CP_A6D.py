import pickle
import scipy.io
import tensorly as tl
from tensorly import norm, cp_tensor, shape, reshape
from tensorly import max, min
from tensorly.decomposition import parafac

# Path to the .pkl file 
filepath = '/Volumes/Devaney SSD/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/'
filename = 'norm_A_CP_M41_R400.pkl'
pkl_file = filepath + filename

# Open and load the pickle file
with open(pkl_file, 'rb') as file:
    data = pickle.load(file)

# Now 'data' is a dictionary with keys 'relative_error' and 'norm_A_CP'
sav_err_A = data['relative_error']
exec_time = data['execution_time']
norm_A_CP = data['norm_A_CP']

# Load MATLAB of A
file_path = '/Volumes/Devaney SSD/BoltzmannData/Boltzmann_Thesis_MD/'
filename = 'A6DtensorM41_pickle.dat' # ensure this is a matlab ".mat" file
file_loc = file_path + filename
with open(file_loc, 'rb') as file:
    tensor_data = pickle.load(file) # we load tensor from matlab file
tensor_array = tensor_data['Atensor'] # we read the tensor data as an array
A = tl.tensor(tensor_array)

# Convert norm_A_CP into tensor and compute error
recon_A = tl.cp_to_tensor(norm_A_CP)
test_err = norm(recon_A - A, 2) / norm(A, 2)

# Print out the relative error
print(f"Pickle Saved Relative error: {sav_err_A * 100:.4f} %")
print(f"Test Relative error: {test_err * 100:.4f} %")
print(f"Execution time: {exec_time} seconds")
