import my_readwrite   ## import the entire module
import my_utils
import numpy as np
import tensorly
from tensorly import norm
from BoltzCol2 import boltzcol2
from my_distributions import maxwellian

MM = 41
Mtrim = 0
dms=MM-2*Mtrim
rank = 25

path = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/sphomruns/good/385/'
id_filename = 'CollTrnDta385_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat'

# let us load initial data
sol, coll_true, solsize = my_readwrite.my_read_sol_coll_trim(path+id_filename,MM, Mtrim)  # this is the first file
## Retrieves solution as a numpy array in the shape [1,solsize]

# for bookkeeping purposes, we also need meshes from our dicscrete solution.
# specifically, we need arrays of velocity points and gauss weights. these
# points can be obtained from the DG-Boltzmann solution saves:
filename='/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/take3_M41/good/080/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)

# compute moments
moments=np.zeros((1,5)) #NOTE consider loading moments into the while loop so it updates for maxwellian
moments[0,0] = np.sum(sol * nodes_gwts)
moments[0,1] = np.sum(sol * nodes_u * nodes_gwts) / moments[0,0]
moments[0,2] = np.sum(sol * nodes_v * nodes_gwts) / moments[0,0]
moments[0,3] = np.sum(sol * nodes_w * nodes_gwts) / moments[0,0]
moments[0,4] = np.sum(sol * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
                                              + (nodes_v - moments[0,2]) ** 2
                                              + (nodes_w - moments[0,3]) ** 2))\
                   / moments[0,0] / 3.0 * 2.0

ffm = sol + maxwellian(moments,nodes_u,nodes_v,nodes_w)
delta_f = sol - maxwellian(moments,nodes_u,nodes_v,nodes_w)
coll_pred = boltzcol2(ffm, delta_f, MM, rank)
norm_pred = norm(coll_pred, 2)
norm_true = norm(coll_true, 2)
abs_err = norm(coll_true - coll_pred, 2)
rel_err = abs_err / norm_true
print("absolute error:", abs_err)
print("relative error:", rel_err)
print("norm of prediction:", norm_pred)
print("norm of true:", norm_true)
