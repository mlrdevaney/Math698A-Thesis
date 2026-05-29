import my_readwrite
import Hermite_Basis
from my_distributions import maxwellian
import numpy as np
import torch
import math
M = 41
Mtrim = 0
nodes_path = 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
solcoll_path = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/take3_M41_cl/080/CollTrnDta108_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(nodes_path)
f, Q, solsize = my_readwrite.my_read_sol_coll_trim (solcoll_path, M, Mtrim)
moments=np.zeros((1,5))
moments[0,0] = 1
moments[0,1] = 0
moments[0,2] = 0
moments[0,3] = 0
moments[0,4] = 0.2
Temp = moments[0,4]
fM = maxwellian(moments,nodes_u,nodes_v,nodes_w)
maxwell1 = torch.reshape(torch.tensor(fM, dtype=torch.float64), (1, -1))
nodes1_u = torch.tensor(nodes_u, dtype=torch.float64)
nodes1_v = torch.tensor(nodes_v, dtype=torch.float64)
nodes1_w = torch.tensor(nodes_w, dtype=torch.float64)
nodes1_gwts = torch.tensor(nodes_gwts, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
ksis = Hermite_Basis.build_ksis(nodes1_u, nodes1_v, nodes1_w, nodes1_gwts, maxwell1)
nodes2_gwts = (nodes1_gwts * math.sqrt(2/Temp)).clone()
Q_proj1 = Q.clone()
print(torch.norm(Q_proj1))
for j in range(5):
    kj = ksis[:, :, j]
    inner_product = torch.sum(Q * kj * nodes2_gwts, dim=1, keepdim=True)
    #norm = torch.sum(kj * kj * maxwell1 * nodes1_gwts, dim=1, keepdim=True)
    Q_proj1 -= (maxwell1 * kj * inner_product) #/ norm

##def verify_orthogonality(proj, ksis):
##    for j in range(5):
##        print(f"Incorrect? Projected Q orthogonality with ksi_{j+1}:", float(torch.sum(proj * ksis[:, :, j] * maxwell1 * nodes1_gwts)))
##verify_orthogonality(Q_proj, ksis)

def verify_orthogonality_1(proj, ksis):
    for j in range(5):
        print(f"Correct Projected Q orthogonality with ksi_{j+1}:", float(torch.sum(proj * ksis[:, :, j] * nodes1_gwts)))

verify_orthogonality_1(Q_proj1, ksis)

print("Amount of collision operator removed via projection:", 100 * float(torch.norm(Q-Q_proj1)/torch.norm(Q)), "%")
print(torch.norm(Q_proj1))
for j in range(5):
    kj = ksis[:, :, j]
    inner_product = torch.sum(Q_proj1 * kj * nodes2_gwts, dim=1, keepdim=True)
    #norm = torch.sum(kj * kj * maxwell1 * nodes1_gwts, dim=1, keepdim=True)
    Q_proj2 = Q_proj1 -(maxwell1 * kj * inner_product) #/ norm
verify_orthogonality_1(Q_proj2, ksis)

print("Amount of collision operator removed via projection:", 100 * float(torch.norm(Q_proj1-Q_proj2)/torch.norm(Q_proj1)), "%")
print(torch.norm(Q_proj2))
