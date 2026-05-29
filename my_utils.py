####################################
# 08/14/2019 A. Alekseenko
# This module contains functions and
# subroutines that are used by the 0D NN-Boltzmann solver
#
####################################

def get_directional_temperatures_torch(
    sol,
    nodes_u,
    nodes_v,
    nodes_w,
    nodes_gwts,
):
    """
    Compute directional temperatures Tx, Ty, Tz for batched VDFs.

    Parameters
    ----------
    sol : torch.Tensor
        Shape [B, N] or [B, Nx, Ny, Nz]
    nodes_u, nodes_v, nodes_w, nodes_gwts : torch.Tensor
        Shape [N] or [Nx, Ny, Nz]
        Must be on same device/dtype as sol

    Returns
    -------
    Tx, Ty, Tz : torch.Tensor
        Shape [B]
    """

    import torch

    B = sol.shape[0]
    device = sol.device
    dtype = sol.dtype

    # ---- Flatten everything consistently ----
    if sol.ndim == 4:
        sol_f = sol.view(B, -1)
    elif sol.ndim == 2:
        sol_f = sol
    else:
        raise ValueError("sol must have shape [B, N] or [B, Nx, Ny, Nz]")

    nodes_u = nodes_u.view(1, -1).to(device=device, dtype=dtype)
    nodes_v = nodes_v.view(1, -1).to(device=device, dtype=dtype)
    nodes_w = nodes_w.view(1, -1).to(device=device, dtype=dtype)
    weights = nodes_gwts.view(1, -1).to(device=device, dtype=dtype)

    # ---- Mass ----
    mass = torch.sum(sol_f * weights, dim=1)  # [B]

    # ---- Bulk velocity ----
    ux = torch.sum(sol_f * nodes_u * weights, dim=1) / mass
    uy = torch.sum(sol_f * nodes_v * weights, dim=1) / mass
    uz = torch.sum(sol_f * nodes_w * weights, dim=1) / mass

    # ---- Directional temperatures ----
    Tx = torch.sum(
        sol_f * (nodes_u - ux[:, None])**2 * weights,
        dim=1
    ) / mass * (2.0 / 3.0)

    Ty = torch.sum(
        sol_f * (nodes_v - uy[:, None])**2 * weights,
        dim=1
    ) / mass * (2.0 / 3.0)

    Tz = torch.sum(
        sol_f * (nodes_w - uz[:, None])**2 * weights,
        dim=1
    ) / mass * (2.0 / 3.0)

    return Tx, Ty, Tz


####################################
# get_moments(sol,nodes_u, nodes_v, nodes_w, nodes_gwts,time)
#
# This subroutine evaluates moments of the discrete solution
#
###################################
def get_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts,time):
    #####
    import numpy as np
    #### Prepare storage:
    entry_moments=np.zeros((1,21))
    #####
    # get mass
    entry_moments[0,0] = time
    zz = sol*nodes_gwts
    entry_moments[0,1] = np.sum(zz)
    # Bulk velocity:
    entry_moments[0,2] = np.sum(sol * nodes_u * nodes_gwts)/entry_moments[0,1]
    entry_moments[0,3] = np.sum(sol * nodes_v * nodes_gwts)/entry_moments[0,1]
    entry_moments[0,4] = np.sum(sol * nodes_w * nodes_gwts)/entry_moments[0,1]
    # temperatur
    entry_moments[0,5] = np.sum(sol * nodes_gwts * ((nodes_u - entry_moments[0,2]) ** 2
                                                  + (nodes_v - entry_moments[0,3]) ** 2
                                                  + (nodes_w - entry_moments[0,4]) ** 2))\
                       / entry_moments[0,1] / 3.0 * 2.0
    #directional temperatures
    entry_moments[0,6] = np.sum(sol * nodes_gwts * (nodes_u - entry_moments[0,2]) ** 2)\
                       / entry_moments[0,1] / 3.0 * 2.0
    entry_moments[0,7] = np.sum(sol * nodes_gwts * (nodes_v - entry_moments[0,3]) ** 2) \
                       / entry_moments[0,1] / 3.0 * 2.0
    entry_moments[0,8] = np.sum(sol * nodes_gwts * (nodes_w - entry_moments[0,4]) ** 2) \
                       / entry_moments[0,1] / 3.0 * 2.0
    # 3rd Moments
    entry_moments[0,9] = np.sum(sol * nodes_gwts * (nodes_u - entry_moments[0,2]) ** 3) \
                       / entry_moments[0,1]
    entry_moments[0,10] = np.sum(sol * nodes_gwts * (nodes_v - entry_moments[0,3]) ** 3) \
                       / entry_moments[0,1]
    entry_moments[0,11] = np.sum(sol * nodes_gwts * (nodes_w - entry_moments[0,4]) ** 3) \
                       / entry_moments[0,1]
    # 4th Moments
    entry_moments[0,12] = np.sum(sol * nodes_gwts * (nodes_u - entry_moments[0,2]) ** 4) \
                       / entry_moments[0,1]
    entry_moments[0,13] = np.sum(sol * nodes_gwts * (nodes_v - entry_moments[0,3]) ** 4) \
                       / entry_moments[0,1]
    entry_moments[0,14] = np.sum(sol * nodes_gwts * (nodes_w - entry_moments[0,4]) ** 4) \
                       / entry_moments[0,1]
    # 5th Moments
    entry_moments[0,15] = np.sum(sol * nodes_gwts * (nodes_u - entry_moments[0,2]) ** 5) \
                       / entry_moments[0,1]
    entry_moments[0,16] = np.sum(sol * nodes_gwts * (nodes_v - entry_moments[0,3]) ** 5) \
                       / entry_moments[0,1]
    entry_moments[0,17] = np.sum(sol * nodes_gwts * (nodes_w - entry_moments[0,4]) ** 5) \
                       / entry_moments[0,1]
    # 6th Moments
    entry_moments[0,18] = np.sum(sol * nodes_gwts * (nodes_u - entry_moments[0,2]) ** 6) \
                       / entry_moments[0,1]
    entry_moments[0,19] = np.sum(sol * nodes_gwts * (nodes_v - entry_moments[0,3]) ** 6) \
                       / entry_moments[0,1]
    entry_moments[0,20] = np.sum(sol * nodes_gwts * (nodes_w - entry_moments[0,4]) ** 6) \
                       / entry_moments[0,1]

    return entry_moments

########################################################
## enf_conservation(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
## This subroutine enforces conservation laws
##
##
##
##
##################################################
def enf_conservation(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim):
    # note that coll_oper comes in trimmed. So we need to untrim it, first before checking conservation.
    import my_readwrite
    coplong, isizen = my_readwrite.solution_untrim(coll_oper[0,:], MM-2*Mtrim, Mtrim)
    #### Let us compute the violation of the conservation law in the collision integral
    import numpy as np
    cons_err=np.zeros((5))
    cons_err[0] = np.sum(coplong * nodes_gwts)
    cons_err[1] = np.sum(coplong * nodes_u * nodes_gwts)
    cons_err[2] = np.sum(coplong * nodes_v * nodes_gwts)
    cons_err[3] = np.sum(coplong * nodes_w * nodes_gwts)
    cons_err[4] = np.sum(coplong * (nodes_u**2 + nodes_v**2 + nodes_w**2) * nodes_gwts)
    #### Next we need to split the array of collision operator into two parts:
    #### the first part keep values abopve treshhol and the second part will keep values below treshhold
    ############################################################
    trshld = 0.00001
    max_value = np.amax(np.absolute(coll_oper[0,:]))
    index_map = np.zeros((MM**3))
    ###
    coll_b=[]
    coll_s=[]
    nodes_u_b=[]
    nodes_u_s=[]
    nodes_v_b=[]
    nodes_v_s=[]
    nodes_w_b=[]
    nodes_w_s=[]
    nodes_gwts_b=[]
    nodes_gwts_s=[]

    for i in range(MM**3):
        if coplong[0,i]>=trshld*max_value:
            coll_b.extend([coplong[0,i]])
            nodes_u_b.extend([nodes_u[0,i]])
            nodes_v_b.extend([nodes_v[0,i]])
            nodes_w_b.extend([nodes_w[0,i]])
            nodes_gwts_b.extend([nodes_gwts[0,i]])
            index_map[i]=1
        else:
            coll_s.extend([coplong[0,i]])
            nodes_u_s.extend([nodes_u[0,i]])
            nodes_v_s.extend([nodes_v[0,i]])
            nodes_w_s.extend([nodes_w[0,i]])
            nodes_gwts_s.extend([nodes_gwts[0,i]])
            index_map[i]=0
    #the collision operator is now split in two parts.
    #let us compute the addition to the error that comes form the small part:
    add_err=np.zeros((5))
    add_err[0] = np.sum(coll_s * np.array(nodes_gwts_s))
    add_err[1] = np.sum(coll_s * np.array(nodes_u_s) * np.array(nodes_gwts_s))
    add_err[2] = np.sum(coll_s * np.array(nodes_v_s) * np.array(nodes_gwts_s))
    add_err[3] = np.sum(coll_s * np.array(nodes_u_s) * np.array(nodes_gwts_s))
    add_err[4] = np.sum(coll_s * (np.array(nodes_u_s)**2
                                  + np.array(nodes_v_s)**2
                                  + np.array(nodes_w_s)**2) * np.array(nodes_gwts_s))
    ##### all done ###########################################################
    ## next we will make the matrix of the moment operator:
    mat = np.array(nodes_gwts_b).reshape((1,len(nodes_gwts_b)))
    z=np.array(nodes_gwts_b) * np.array(nodes_u_b)
    mat = np.concatenate((mat, z.reshape((1,len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * np.array(nodes_v_b)
    mat = np.concatenate((mat, z.reshape((1,len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * np.array(nodes_w_b)
    mat = np.concatenate((mat, z.reshape((1, len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * (np.array(nodes_w_b)**2+np.array(nodes_u_b)**2+np.array(nodes_v_b)**2)
    mat = np.concatenate((mat, z.reshape((1, len(nodes_gwts_b)))),axis=0)
    ##
    from scipy import linalg
    U, s, Vh = linalg.svd(mat, full_matrices=False)
    ###
    z=cons_err
    Uinv=np.linalg.inv(U)
    y = np.dot(Uinv,z)
    y = y/s
    coll_corr = np.dot(Vh.T,y)
    coll_b = coll_b - coll_corr
    ############ Now we need to place the values of the collision operator back.
    j=0
    for i in range(MM**3):
        if index_map[i]==1:
            coplong[0,i]=coll_b[j]
            j=j+1
    ### Now we need to trim the array.
    coll_cons, newsize = my_readwrite.solution_trim(coplong,MM,Mtrim)

    #################### DEBUG
    cons_err[0] = np.sum(coplong * nodes_gwts)
    cons_err[1] = np.sum(coplong * nodes_u * nodes_gwts)
    cons_err[2] = np.sum(coplong * nodes_v * nodes_gwts)
    cons_err[3] = np.sum(coplong * nodes_w * nodes_gwts)
    cons_err[4] = np.sum(coplong * (nodes_u ** 2 + nodes_v ** 2 + nodes_w ** 2) * nodes_gwts)

    #############################

    return coll_cons, newsize

########################################################
## enf_conservation_sol(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim,conserved_macroparams)
## This subroutine enforces conservation laws in solutions
##
##
##
##
##################################################
def enf_conservation_sol(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim,conserved_macroparams):
    # note that coll_oper comes in trimmed. So we need to untrim it, first before checking conservation.
    import my_readwrite
    coplong, isizen = my_readwrite.solution_untrim(sol[0,:], MM-2*Mtrim, Mtrim)
    #### Let us compute the violation of the conservation law in the collision integral
    import numpy as np
    cons_err=np.zeros((5))
    cons_err[0] = np.sum(coplong * nodes_gwts) - conserved_macroparams[0]
    cons_err[1] = np.sum(coplong * nodes_u * nodes_gwts) - conserved_macroparams[1]
    cons_err[2] = np.sum(coplong * nodes_v * nodes_gwts) - conserved_macroparams[2]
    cons_err[3] = np.sum(coplong * nodes_w * nodes_gwts) - conserved_macroparams[3]
    cons_err[4] = np.sum(coplong * (nodes_u**2 + nodes_v**2 + nodes_w**2) * nodes_gwts) - conserved_macroparams[4]
    #### Next we need to split the array of collision operator into two parts:
    #### the first part keep values abopve treshhol and the second part will keep values below treshhold
    ############################################################
    trshld = 0.00001
    max_value = np.amax(np.absolute(sol[0,:]))
    index_map = np.zeros((MM**3))
    ###
    coll_b=[]
    coll_s=[]
    nodes_u_b=[]
    nodes_u_s=[]
    nodes_v_b=[]
    nodes_v_s=[]
    nodes_w_b=[]
    nodes_w_s=[]
    nodes_gwts_b=[]
    nodes_gwts_s=[]

    for i in range(MM**3):
        if coplong[0,i]>=trshld*max_value:
            coll_b.extend([coplong[0,i]])
            nodes_u_b.extend([nodes_u[0,i]])
            nodes_v_b.extend([nodes_v[0,i]])
            nodes_w_b.extend([nodes_w[0,i]])
            nodes_gwts_b.extend([nodes_gwts[0,i]])
            index_map[i]=1
        else:
            coll_s.extend([coplong[0,i]])
            nodes_u_s.extend([nodes_u[0,i]])
            nodes_v_s.extend([nodes_v[0,i]])
            nodes_w_s.extend([nodes_w[0,i]])
            nodes_gwts_s.extend([nodes_gwts[0,i]])
            index_map[i]=0
    #the solution is now split in two parts.
    ## next we will make the matrix of the moment operator:
    mat = np.array(nodes_gwts_b).reshape((1,len(nodes_gwts_b)))
    z=np.array(nodes_gwts_b) * np.array(nodes_u_b)
    mat = np.concatenate((mat, z.reshape((1,len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * np.array(nodes_v_b)
    mat = np.concatenate((mat, z.reshape((1,len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * np.array(nodes_w_b)
    mat = np.concatenate((mat, z.reshape((1, len(nodes_gwts_b)))),axis=0)
    z=np.array(nodes_gwts_b) * (np.array(nodes_w_b)**2+np.array(nodes_u_b)**2+np.array(nodes_v_b)**2)
    mat = np.concatenate((mat, z.reshape((1, len(nodes_gwts_b)))),axis=0)
    ##
    from scipy import linalg
    U, s, Vh = linalg.svd(mat, full_matrices=False)
    ###
    z=cons_err
    Uinv=np.linalg.inv(U)
    y = np.dot(Uinv,z)
    y = y/s
    sol_corr = np.dot(Vh.T,y)
    coll_b = coll_b - sol_corr
    ############ Now we need to place the values of the collision operator back.
    j=0
    for i in range(MM**3):
        if index_map[i]==1:
            coplong[0,i]=coll_b[j]
            j=j+1
    ### Now we need to trim the array.
    sol_cons, newsize = my_readwrite.solution_trim(coplong,MM,Mtrim)

    #################### DEBUG
    cons_err[0] = np.sum(coplong * nodes_gwts)
    cons_err[1] = np.sum(coplong * nodes_u * nodes_gwts)
    cons_err[2] = np.sum(coplong * nodes_v * nodes_gwts)
    cons_err[3] = np.sum(coplong * nodes_w * nodes_gwts)
    cons_err[4] = np.sum(coplong * (nodes_u ** 2 + nodes_v ** 2 + nodes_w ** 2) * nodes_gwts)

    #############################

    return sol_cons, newsize


########################################################
## compute_conservative_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
## This subroutine enforces conservation laws in solutions
##
##
##
##
##################################################
def compute_conservative_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim):
    # note that sol array comes in trimmed. So we need to untrim it, first before checking conservation.
    import my_readwrite
    coplong, isizen = my_readwrite.solution_untrim(sol[0, :], MM - 2 * Mtrim, Mtrim)
    #### Let us compute the violation of the conservation law in the collision integral
    import numpy as np
    cons_moms = np.zeros((5))
    cons_moms[0] = np.sum(coplong * nodes_gwts)
    cons_moms[1] = np.sum(coplong * nodes_u * nodes_gwts)
    cons_moms[2] = np.sum(coplong * nodes_v * nodes_gwts)
    cons_moms[3] = np.sum(coplong * nodes_w * nodes_gwts)
    cons_moms[4] = np.sum(coplong * (nodes_u ** 2 + nodes_v ** 2 + nodes_w ** 2) * nodes_gwts)
    #### Next we need to split the array of collision operator into two parts:

    return cons_moms
