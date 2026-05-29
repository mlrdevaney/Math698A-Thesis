import numpy as np
import os
import math
import re
import Utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
import pickle
import matplotlib.pyplot as plt
import my_readwrite
from LearnColOpCNN_Class_v7 import LearnColOpCNN
from my_distributions import maxwellian


nodes_path = 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
match = re.search(r'sw(\d+)MuUU', nodes_path)
if match:
    M_val = int(match.group(1))
    SVD_pkl_file = f'SVD_U_M{M_val}.pkl'
    if os.path.exists(SVD_pkl_file) and os.path.getsize(SVD_pkl_file) > 0:
        #print(f"Found precomputed SVD for M={M_val}. Loading from {SVD_pkl_file}...")
        with open(SVD_pkl_file, 'rb') as svd_file:
            U = pickle.load(svd_file)
        U = U.cpu().numpy() if torch.is_tensor(U) else U

MM = 41
Mtrim = 0
path = '/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/BBB good testing half copy/462 copy'
sol_coll_filename = 'CollTrnDta462_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat'
datapath = os.path.join(path, sol_coll_filename)

# let us load initial data
sol, coll, nsize = my_readwrite.my_read_sol_coll_trim(datapath,MM, Mtrim)  # this is the first file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nodes_path = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/take3_M41/good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(nodes_path)

moments=np.zeros((1,5))
moments[0,0] = np.sum(sol * nodes_gwts)
moments[0,1] = np.sum(sol * nodes_u * nodes_gwts) / moments[0,0]
moments[0,2] = np.sum(sol * nodes_v * nodes_gwts) / moments[0,0]
moments[0,3] = np.sum(sol * nodes_w * nodes_gwts) / moments[0,0]
moments[0,4] = np.sum(sol * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
                                              + (nodes_v - moments[0,2]) ** 2
                                              + (nodes_w - moments[0,3]) ** 2))\
                   / moments[0,0] / 3.0 * 2.0
maxwell = maxwellian(moments, nodes_u, nodes_v, nodes_w)
maxwell = np.reshape(maxwell, (1, -1))

#print(U.T @ U)

criterion = nn.L1Loss()

print("norm before projection: ", np.linalg.norm(coll))
collp1 = coll - (coll @ U) @ U.T
print("norm after 1 projection: ", np.linalg.norm(collp1))
print("percentage of collop lost by projecting: ", 100 * np.linalg.norm(coll - collp1)/(np.linalg.norm(coll)))
collp2 = collp1 - (collp1 @ U) @ U.T
print("norm after 2 projection: ", np.linalg.norm(collp2))
print("percentage of collop lost by projecting: ", 100 * np.linalg.norm(collp1 - collp2)/(np.linalg.norm(collp1)))
collp3 = collp2 - (collp2 @ U) @ U.T
print("norm after 3 projection: ", np.linalg.norm(collp3))
print("percentage of collop lost by projecting: ", 100 * np.linalg.norm(collp2 - collp3)/(np.linalg.norm(collp2)))

coll = torch.tensor(coll, dtype=torch.float64)
collp1 = torch.tensor(collp1, dtype=torch.float64)
collp2 = torch.tensor(collp2, dtype=torch.float64)
collp3 = torch.tensor(collp3, dtype=torch.float64)
print(criterion(coll, collp1))
print(criterion(collp1, collp2))
print(criterion(collp2, collp3))

##print("norm before projection: ", np.linalg.norm(sol))
##solp1 = sol - (sol @ U) @ U.T
##print("norm after 1 projection: ", np.linalg.norm(solp1))
##print("percentage of solution lost by projecting: ", 100 * np.linalg.norm(sol - solp1)/(np.linalg.norm(sol)))
##
##delf = sol - maxwell
##print("norm before projection: ", np.linalg.norm(delf))
##delfp1 = delf - (delf @ U) @ U.T
##print("norm after 1 projection: ", np.linalg.norm(delfp1))
##print("percentage of delf lost by projecting: ", 100 * np.linalg.norm(delf - delfp1)/(np.linalg.norm(delf)))
