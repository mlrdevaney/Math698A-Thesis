####################################################################################################################################################################
# This script is used to produce a .pkl file for the collision kernel after
# performing CP tensor decomposition.
# When running this script, some aspects that may need changing are:
# (1) rank for decomposition i.e. sigma
# (2) file_path and filename
# (3) save_path
#
# Otherwise, the way this script operates is given a specified
# rank it will then CP decompose the 6D collision kernel tensor.
# Error will be computed, along with the CP normalized weights and
# factor matrices, and execution time.
# All of this will be dumped into a .pkl file for later use.
####################################################################################################################################################################
print("Python Script Started")
import os                     # for os.path.abspath, used in final print
import re                     # for regex to extract M from filename
import time                   # for measuring execution time
import pickle                 # for saving results as a .pkl file
import numpy as np            # used implicitly via tensorly backends
import scipy.io               # for scipy.io.loadmat
import torch                  # required since tensorly may use PyTorch backend
import tensorly as tl         # core tensor ops and decomposition
from tensorly.decomposition import parafac  # for CP decomposition
from tensorly import norm                  # used for relative error
from datetime import datetime              # for timestamp logging

### === Set backend explicitly if desired ===
##tl.set_backend('numpy')
print("Setting Back-End & Threads")
# Set backend to PyTorch for GPU acceleration if available
tl.set_backend('pytorch')
##device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##print(f"Using device: {device}")
torch.set_num_threads(1) #NOTE unnecessary on PC

# Start time
start = time.time()
current_time = datetime.now()
print(f'Start time: {current_time.strftime("%H:%M:%S")}')

rank = 2 # sigma
file_path = '/pscratch/sd/m/mdevaney/Boltzmann/'
filename = 'A6DtensorM41_pickle.dat'  # ensure this is a matlab ".mat" file
file_loc = file_path + filename

# Extract M_value from filename
match = re.search(r'A6DtensorM(\d+)_pickle.dat', filename)
if not match:
    raise ValueError("Could not extract M value from filename.")
M_value = match.group(1)

save_path = f'/pscratch/sd/m/mdevaney/Boltzmann/norm_A_CP_M{M_value}_R{rank}.pkl'

# Load tensor from MATLAB/Picle.dat file and move to device
##tensor_data = scipy.io.loadmat(file_loc)
##tensor_array = tensor_data['Atensor']
with open(file_loc, 'rb') as file:
    Atensor = pickle.load(file)
A = tl.tensor(Atensor) # device=device)

# Perform CP decomposition
norm_A_CP = parafac(
    A,
    rank=rank,
    normalize_factors=True,
    init='random',
    linesearch=False,
    orthogonalise=False,
)
#norm_A_CP = cp_tensor.cp_normalize(A_CP)

# Compute reconstruction and relative error
recon_A = tl.cp_to_tensor(norm_A_CP)
err_A = norm(recon_A - A, 2) / norm(A, 2)


# Print results
print(f"Normalized CP decomposition saved to:\n{os.path.abspath(save_path)}")
print(f"Relative error: {err_A * 100:.4f} %")
# End time and total duration
end = time.time()
exec_time = end - start
print("Code execution time: ", (exec_time), "seconds")
current_time = datetime.now()
print(f'End time: {current_time.strftime("%H:%M:%S")}')

# Save both norm_A_CP and error to .pkl file
output_data = {
    "relative_error": err_A,
    "execution_time": exec_time,
    "norm_A_CP": norm_A_CP
}
with open(save_path, "wb") as file:
    pickle.dump(output_data, file)
