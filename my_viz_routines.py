########################
# 07/16/2019 A. Alekseenko
# This module contains functions and
# subroutines that visualize the velocity distribution function
########################

#########################
#  plot2DUV_sol (sol, u_box, w_val, nu, nv  ):
# This subroutine plots a 3D surface graph of a 2D trace of the
# VDF, f(u,v,w) for a constant value of variable w.
#  Variables:
#
#  sol -- an 1D array of solution's galerkin coefficients (k=0) case, size (1,M^3)
#  velbox == a tuple of 6 numbers providing the boundaries of the 3D rectangular region where solution is defined.
#           Format: (u_l, u_r, v_l, v_r, w_l, w_r)
#  velN == a tuple of 3 numbers providing number of cells in velocity variables.
#           Format: (Nu, Nv, Nw)
#  u_val, v_val : 1D arrays of values of variables u and v wehre the solution needs to be evaluated.
#           the total mesh will be the tensor product of u_val and v_val
#  w_val  should be just one number
###########################
def eval0k2DUV_sol (sol, velbox, velN, u_val, v_val, w_val):
    import numpy as np



    ########################################################
    ## Make a 2D array that contains values of the solution
    #########################################################
    ## throw away points that are not on the mesh:
    #u_val = u_val[(u_val >= velbox[0]) & (u_val < velbox[1])]
    #v_val = v_val[(v_val >= velbox[2]) & (v_val < velbox[3])]
    #w_val = w_val[(w_val >= velbox[4]) & (w_val < velbox[5])]
    ## find locations of a point on the mesh:
    u_ival = ((u_val-velbox[0])*velN[0]//(velbox[1]-velbox[0]))
    u_ival = u_ival.astype('int')
    #u_ival = u_ival[(u_ival >= 0 ) & (u_ival <=velN[0]-1)]
    v_ival = ((v_val-velbox[2])*velN[1]//(velbox[3]-velbox[2]))
    v_ival = v_ival.astype('int')
    #v_ival = v_ival[(v_ival >= 0 ) & (v_ival <=velN[1]-1)]
    w_ival = int((w_val-velbox[4])*velN[2]//(velbox[5]-velbox[4]))
    #w_ival = w_ival[(w_ival >= 0 ) & (w_ival <=velN[2]-1)]
    ## check that array of real values and arrays of indices have the same size
    assert len(u_ival) == len(u_val)
    assert len(v_ival) == len(v_val)
    ###############################################################
    ## In this subroutine we are only ploting k=0 Galerkin solutions.
    ## for K=0 the solutions are piece-wise constants in the cells
    ## so we just pull out all values from the sol array and this gives us the value of the function
    #################################################################
    sol_val = np.zeros((len(u_ival), len(v_ival)))  # prepare the space for the solution
    umesh = np.zeros((len(u_ival), len(v_ival)))
    vmesh = np.zeros((len(u_ival), len(v_ival)))
    iw=w_ival
    for iu in range(len(u_ival)):
        for iv in range(len(v_ival)):
         sol_ind = iw + velN[2]*(v_ival[iv])+velN[1]*velN[2]*(u_ival[iu])
         sol_val[iu,iv] = sol[sol_ind]  # this is the value of the solution
         umesh[iu,iv] = u_val[iu]
         vmesh[iu,iv] = v_val[iv]
    # the arryas were computed
    return sol_val, umesh, vmesh
