####################################################################################################################################################################
# This script is used to produce the Boltzmann collision operator from
# the convolution of two velocity distribution functions (f & g) with
# collision kernel (B). 
# Bdditionally, we require the number of velocity points (MM) and a choice
# of rank(s) for CP decomposition(s) of f and g i.e. (f_rank & g_rank).
# If no rank is given for g_rank, it will default to rank given by f_rank.
# When running this script, some aspects that may need changing are:
# (1) gamma which changes with number of velocity points (MM)
# (2) n_anc which defines the anchor point around which the tensor is defined
#
# Otherwise, the way this script operates is to CP decompose the tensors
# representing the velocity distribution functions, take the correct
# vectors from the respective factor matrices and convolve them with the
# correct vectors from 6D collision kernel. Next, we build factor matrices
# of our tensor (D) and weights (omega) which come from the weights
# of f, g, and B from CP decomposing them. Finally, we convert the CP form of our D tensor
# into a true tensor, reshape this tensor into a 1D array, apply gamma, and
# return the collision operator.
#
# Input (f,g) shapes each: (1, MM**3) - 2D
# Output (Q/CollOp) shape: (MM**3, )  - 1D
####################################################################################################################################################################
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.base import unfold

def cpQ(f, g, B_weights, list_B, MM, device, B_rank, f_rank, g_rank = None):
    
####################################################################################################################################################################
# We define the convolution function
####################################################################################################################################################################

    def conv (list_F, list_B, F_dim, B_dim, rank_F, rank_B, MM, n_anc):
        # n_anc is the anchor/reference point for the kernel
        # Convert lists to NumPy arrays if they aren't already
        c = torch.zeros(
            (MM, rank_B, rank_F),
            dtype=torch.float64,
            device=device
        )
        for n in range(MM):
            del_n =  (n_anc - n) # NOTE we are testing with a negative sign
            n_start = max(0, del_n)
            n_end = min(MM, MM + del_n)
            if n_start >= n_end:
                continue  # Skip if range is empty
            # Use NumPy array slicing
            n_range = torch.arange(n_start, n_end)
            F_slice = list_F[F_dim][:, n_range - del_n] 
            B_slice = list_B[B_dim][:, n_range]
            c[n] = torch.einsum("ij,kj->ik", F_slice, B_slice).T
        return c
    
################################################################################################################################################################################
# We define function to build D tensors which are "factor tensors"
################################################################################################################################################################################

    def build_D(c1, c2):
        return c1[:, :, :, None] * c2[:, :, None, :]
    
################################################################################################################################################################################
    
    # gamma1 = 2 * 1.141376e+2 / 6.4e-2 * (41.0 / 41.0) ** 3 #changed 15.0 to 41.0 in (x/41.0)**3
    gamma = 3.76**2 / 0.00313402 # mol_diam**2 / nodes_gwts
    
    
    if g_rank == None:
        g_rank = f_rank
    
    ##################################################################################
    #############  Next we load up F                               ##################
    ##################################################################################
    F = torch.tensor(f.reshape(MM, MM, MM), dtype=torch.float64, device=device) # NOTE is f alone? then change f to f[0]; what about reshape order='C'
    # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    F_CP = parafac(F, rank=f_rank, init='svd', normalize_factors=True) #NOTE we changed random to svd
    # print(F_CP[0])
    # print(F_CP[1][:10])
    del F # free up memory
    list_F = [unfold(factor_matrix, 1) for factor_matrix in F_CP[1]]
    #norm_F_CP[0] is factor weights
    #norm_F_CP[1] is factor matrices
    
 
    #################################################################################
    #############  Now we get G                                    #################
    #################################################################################

    G = torch.tensor(g.reshape(MM, MM, MM), dtype=torch.float64, device=device) # NOTE is g alone? then change g to g[0]
    # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    G_CP = parafac(G, rank=g_rank, init='svd', normalize_factors=True) #NOTE we changed random to svd
    del G # free up memory
    list_G = [unfold(factor_matrix, 1) for factor_matrix in G_CP[1]]
    #norm_G_CP[0] is factor weights
    #norm_G_CP[1] is factor matrices

    # recon_f = tl.cp_to_tensor(F_CP)
    # f_true = reshape(tl.tensor(f), (MM, MM, MM)).to(device=device, dtype=torch.float64)
    # err_f = norm(recon_f - f_true, 2) / norm(f_true, 2)
    # recon_g = tl.cp_to_tensor(G_CP)
    # g_true = reshape(tl.tensor(g), (MM, MM, MM)).to(device=device, dtype=torch.float64)
    # err_g = norm(recon_g - g_true, 2) / norm(g_true, 2)
    # print("Relative Errors for F and G:", err_f, err_g)
    #####################################################################################
    ##############  Now we convolve B with F                               ##############
    #####################################################################################
    anchor = 20
    c1 = conv(list_F, list_B, 0, 0, f_rank, B_rank, MM, anchor) #NOTE n_anc=anchor, 7 or 20 may change for different MM values
    c2 = conv(list_F, list_B, 1, 1, f_rank, B_rank, MM, anchor)
    c3 = conv(list_F, list_B, 2, 2, f_rank, B_rank, MM, anchor)
    c4 = conv(list_G, list_B, 0, 3, g_rank, B_rank, MM, anchor)
    c5 = conv(list_G, list_B, 1, 4, g_rank, B_rank, MM, anchor)
    c6 = conv(list_G, list_B, 2, 5, g_rank, B_rank, MM, anchor)

    #################################################################################
    # Finally, build collision operator after performing CP with the factor         #
    # matrices and weight for D    ##################################################
    #################################################################################
    D1 = build_D(c1, c4).reshape(MM, B_rank * f_rank * g_rank)
    D2 = build_D(c2, c5).reshape(MM, B_rank * f_rank * g_rank)
    D3 = build_D(c3, c6).reshape(MM, B_rank * f_rank * g_rank)
    del c1, c2, c3, c4, c5, c6 # free up memory

    omega = torch.einsum('i,j,k->ijk', B_weights, F_CP[0], G_CP[0]).reshape(-1) # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    fm_d = [D1, D2, D3]
    del D1, D2, D3 # free up memory
    # Ensure fm_d is a valid list of factor matrices
    CP_D = (omega, fm_d)  # Format as (weights, factors)
    del fm_d # free up memory
    D_tensor = tl.cp_to_tensor(CP_D)
    del CP_D
    ColOper = torch.flatten(D_tensor)  # ColOper is collision operator in form (MM ** 3, )
    del D_tensor # free up memory
    return gamma * ColOper
