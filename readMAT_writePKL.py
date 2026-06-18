#############################################################################
### This code is to investigate tensor decompositions of kernels of collision operator
### A. Alekseenko, 10/21/2021
###
### First, we read the collision kernel
### Second, we convert it into a 6 index array
### Third, we try to represent it in a tucker, thensor train, or hierarchical tucker format
##############################################################################

import numpy as np
import my_readwrite


###############################################################################
## Part 1. Reading the array
###############################################################################
# path to the location of the binary file that contains the A array data
pathAarry='F:/BoltzmannData/TTBltzmn/exp0912_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_Aarrs.dat'
# this subrouinte will extract the A arrays fom the binary file
A_capphi, A, A_xi, A_xi1, A_phi = my_readwrite.my_get_Aarrys(pathAarry)
###############################################################################
# A contains values of the collision kernel that is written as a matrix (symmetrix with zero diagonal). a_{ij}
# A_xi has the index i of the entry
# A_xi1 has the index j of the entry
# A_phi has the number of the basis function to which the entry corresponds.
# for s=1 -- piece-wise constant DG  all entries will be for a single function
################################################################################
# we are ready to make a six index array out of A --- most entries are zeros, so it makes sense to
# use sparse matrices -- we may will get to it, but for now, we will use full matrices
M=9  # CHECK THIS MATCHES set the number of velocity points in each dimension
s=1   # order of the DG method
Atensor = np.zeros((M,M,M,M,M,M))    # indices are taken to be (u,v,w,u1,v1,w1)
bfnum = A_phi[0,0]   # this will be the basis function that corresponds to the first record of A
numArec = A_capphi[0,bfnum-1]
nn=A.shape[1]    # number of records
#####################################
# sanity check: is s=1 then nn has to be the same is numArec
if nn != numArec:
    print("Attention: If s=1 then given Value of nn is incosistent with the value of numArec ")
    exit()
#####################################
import my_tools
for i in range(nn):
    i1=A_xi[0,i]
    i2=A_xi1[0,i]
    i11, i12, i13, p11, p12, p13 = my_tools.DGVind(i1, M, s)
    i21, i22, i23, p21, p22, p23 = my_tools.DGVind(i2, M, s)
    Atensor[i11 - 1, i12 - 1, i13 - 1, i21 - 1, i22 - 1, i23 - 1] = A[0, i]
    Atensor[i21 - 1, i22 - 1, i23 - 1, i11 - 1, i12 - 1, i13 - 1] = A[0, i]
####################################
z1=np.amax(A)
z2=np.amax(Atensor)
##print(np.amax(A))
##print(np.amin(A))
#######################################
### Tensor have been recorded
#######################################
import pickle
file=open(f'A6DtensorM{M}_pickle.dat', 'wb')
print("Pickle Dumping Now...")
pickle.dump(Atensor,file)
file.close()
### Saving in matlab format
#import matlab
#Atensor1=matlab.double(Atensor.tolist())
##import scipy.io
##mdict = {"Atensor": Atensor} # .tolist should convert to matlab array
##scipy.io.savemat(f'A6DtensorM{M}_matlab.mat',mdict)
print("done")
