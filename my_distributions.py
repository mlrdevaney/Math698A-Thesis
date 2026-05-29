###################################
#
# module my_distributions.py contains functions to evaluate some classical velocity distributions densities of gas dynamics
#
#####################

#####################################################################
#
# maxwellian(moment_vector_0,nodes_u,nodes_v,nodes_w)
#
# this function implements dimensionless Maxwellian distribution density
#
#   moment_vector_0 -- vecotor of moments density, three components of bulk velocity x density and energy times density (not temperature)
#
# nodes_u,nodes_v,nodes_w - three arrays with components of u,v,w coordimates where the distribution function needs to be evaluated.
#
######################################################################

def maxwellian(moment_vector,nodes_u,nodes_v,nodes_w):
    import numpy as np
    import math
    n=moment_vector[0,0]
    u_0 = moment_vector[0,1]
    v_0 = moment_vector[0,2]
    w_0 = moment_vector[0,3]
    #Temp = 2.0/3.0*(moment_vector[0,4] - (u_0*u_0 + v_0*v_0 +w_0*w_0)*n)
    Temp = moment_vector[0,4]
    ####
    beta = math.sqrt(math.pi * Temp) * (math.pi * Temp)
    y = n * np.exp(-((nodes_u[0,:] - u_0) * (nodes_u[0,:] - u_0) + (nodes_v[0,:]  - v_0) * (nodes_v[0,:]  - v_0) + \
                  (nodes_w[0,:] - w_0) * (nodes_w[0,:] - w_0)) / max(Temp, 0.000001)) / beta
    return y
