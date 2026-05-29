import sys
import pickle
import numpy as np
import tensorly as tl
import scipy.io
import time
import statistics as st
from tensorly import shape, max, min
from tensorly.decomposition import parafac
from tensorly import norm, reshape, cp_to_tensor, shape
sys.path.append('/Volumes/Devaney SSD/BoltzmannData/Boltzmann_Thesis_MD')
from BoltzCol1 import boltzcol1
from BoltzCol2 import boltzcol2
#from BoltzCol2_optim import boltzcol2
sys.path.append('/Volumes/Devaney SSD/BoltzmannData/TTBltzmn')
from my_readwrite import my_read_sol_coll_trim, read_nodes
##sys.path.append('/Volumes/MD Research/BoltzmannData/ROM_simulations/cleaned_100_MakePicklefiles_SDV/')
##from my_distributions import maxwellian
path = '/Volumes/Devaney SSD/BoltzmannData/RUNS M=41/sphomruns/good/381/'
id_filename1 = 'CollTrnDta381_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat' # error between 0.00 and 0.000...420 was 1.6%
id_filename2 = 'CollTrnDta381_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.2235000000_SltnColl.dat'
file1 = path + id_filename1
file2 = path + id_filename2
##filename = '/Volumes/MD Research/BoltzmannData/nodes_collection/TestCP15_1su1sv1sw15MuUU15MvVU15MwWU_nodes.dat'
MM = 41
Mtrim = 0
#sol and soltrm are identical when Mtrim = 0
soltrm1, colltrm1, solsze1 = my_read_sol_coll_trim(file1, MM, Mtrim)
soltrm2, colltrm2, solsze2 = my_read_sol_coll_trim(file2, MM, Mtrim)
##nodes_u, nodes_v, nodes_w, nodes_gwts = read_nodes(filename)
### compute moments
##moments=np.zeros((1,5))
##moments[0,0] = np.sum(soltrm1 * nodes_gwts)
##moments[0,1] = np.sum(soltrm1 * nodes_u * nodes_gwts) / moments[0,0]
##moments[0,2] = np.sum(soltrm1 * nodes_v * nodes_gwts) / moments[0,0]
##moments[0,3] = np.sum(soltrm1 * nodes_w * nodes_gwts) / moments[0,0]
##moments[0,4] = np.sum(soltrm1 * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
##                                              + (nodes_v - moments[0,2]) ** 2
##                                              + (nodes_w - moments[0,3]) ** 2))\
##                   / moments[0,0] / 3.0 * 2.0
##ffm = soltrm1 + maxwellian(moments,nodes_u,nodes_v,nodes_w)
##delta_f = soltrm1 - maxwellian(moments,nodes_u,nodes_v,nodes_w)
########################################################################################################################################################################################################################################################################################################
# FOR COMPUTING ERROR IN CP OF A ########################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################
##path = '/Volumes/MD Research/BoltzmannData/Boltzmann_Thesis_MD/norm_A_CP_M15_R160.pkl'
##with open(path, "rb") as file:
##    norm_A_CP = pickle.load(file)
##file_path = '/Volumes/MD Research/BoltzmannData/TTBltzmn/' 
##filename = 'A6DtensorM15_matlab.mat' # ensure this is a matlab ".mat" file
##file_loc = file_path + filename
##tensor_data = scipy.io.loadmat(file_loc) # we load tensor from matlab file
##tensor_array = tensor_data['Atensor'] # we read the tensor data as an array
##A = tl.tensor(tensor_array)
##CP_A = cp_to_tensor(norm_A_CP)
##err_A = norm(A - CP_A, 2) / norm(A, 2)
##print("Error in CP of A", err_A * 100, "%")
########################################################################################################################################################################################################################################################################################################
# FOR COMPUTING ERROR IN CP OF F ########################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################
soltrm1 = reshape(soltrm1, (MM, MM, MM))
soltrm2 = reshape(soltrm2, (MM, MM, MM))
colltrm1 = reshape(colltrm1, (MM, MM, MM))
colltrm2 = reshape(colltrm2, (MM, MM, MM))
dfdt = (soltrm2 - soltrm1) / 0.2235
print("colltrm1", norm(colltrm1 - dfdt, 2) / norm(dfdt, 2) * 100, "%")
print("colltrm2", norm(colltrm2 - dfdt, 2) / norm(dfdt, 2) * 100, "%")
##soltrm = reshape(soltrm1[0], (MM, MM, MM), 'C')
##soltrm = tl.tensor(soltrm)
##CP_soltrm = parafac(soltrm, 40)
##CP_soltrm = cp_to_tensor(CP_soltrm)
##err_f = norm(soltrm - CP_soltrm, 2) / norm(soltrm, 2)
##print("Error in CP of F", err_f * 100, "%")
########################################################################################################################################################################################################################################################################################################
# FOR COMPUTING ERROR IN COLLISION OPERATOR ########################################################################################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################################################################################
##start1 = time.time()
##rank = 50
##Q_Col = boltzcol2(ffm, delta_f, MM, rank)
##end1 = time.time()
##print(f"Collision operator function execution time for rank {rank}: ", (end1 - start1), "seconds")
##x = colltrm[0]
##err = norm(Q_Col - x, 2) / norm(x, 2)
##print(f"Error in collision operator for rank {rank}: ", err * 100, "%")
##print("x", x)
##print("new q", new_q)
##print('x collision', x[7*(15**2) + 7*15 + 6])
##print('x collision', x[7*(15**2) + 7*15 + 7])
##print('x collision', x[7*(15**2) + 7*15 + 8])
##print('new q', new_q[7*(15**2) + 7*15 + 6])
##print('new q', new_q[7*(15**2) + 7*15 + 7])
##print('new q', new_q[7*(15**2) + 7*15 + 8])
##gen_contract = np.einsum('ijk,ijklmn->', CP_soltrm, CP_A)
##print("general", gen_contract)
##summ = 0
##for i in range(MM):
##    for j in range(MM):
##        for k in range(MM):
##            for ii in range(MM):
##                for jj in range(MM):
##                    for kk in range(MM):
##                        summ = summ + CP_soltrm[i,j,k] * CP_soltrm[ii,jj,kk] * CP_A[i,j,k,ii,jj,kk]
##beta_coeff = 1.141376e+2 / 6.4e-2
##print('summ', summ * beta_coeff)

