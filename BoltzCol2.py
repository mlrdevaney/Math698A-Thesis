####################################################################################################################################################################
# This script is used to produce the Boltzmann collision operator from
# the convolution of two velocity distribution functions (f & g) with
# collision kernel (A). 
# Additionally, we require the number of velocity points (MM) and a choice
# of rank(s) for CP decomposition(s) of f and g i.e. (sigma_p & sigma_dp).
# If no rank is given for sigma_dp, it will default to rank given by sigma_p.
# When running this script, some aspects that may need changing are:
# (1) file_path and filename to specify the .pkl file where we obtain
#           the CP decomposed collision kernel A.
# (2) gamma which changes with number of velocity points (MM)
# (3) n_anc which defines the anchor point around which the tensor is defined
#
# Otherwise, the way this script operates is to CP decompose the tensors
# representing the velocity distribution functions, take the correct
# vectors from the respective factor matrices and convolve them with the
# correct vectors from 6D collision kernel. Next, we build factor matrices
# of our tensor (D) and weights (omega) which come from the weights
# of f, g, and A from CP decomposing them. Finally, we convert the CP form of our D tensor
# into a true tensor, reshape this tensor into a 1D array, apply gamma, and
# return the collision operator.
#
# Input (f,g) shapes each: (1, MM**3) - 2D
# Output (Q/CollOp) shape: (MM**3, )  - 1D
####################################################################################################################################################################

def boltzcol2 (f, g, MM, sigma_p, sigma_dp = None):
    import numpy as np
    import tensorly as tl
    import os
    import pickle
    from tensorly.decomposition import parafac
    from tensorly.base import fold, unfold, tensor_to_vec, vec_to_tensor
    from tensorly import ndim, cp_tensor, zeros, reshape, shape

####################################################################################################################################################################
# We define the convolution function
####################################################################################################################################################################

    def conv (list_F, list_A, F_dim, A_dim, rank_A, rank_F, MM, n_anc):
        # n_anc is the anchor/reference point for the kernel
        # Convert lists to NumPy arrays if they aren't already
        c = np.zeros((MM, rank_A, rank_F), dtype=np.float64)
        for n in range(MM):
            del_n = n_anc - n
            n_start = max(0, del_n)
            n_end = min(MM, MM+del_n)
            if n_start >= n_end:
                continue  # Skip if range is empty
            # Use NumPy array slicing
            n_range = np.arange(n_start, n_end)
            F_slice = list_F[F_dim][:, n_range - del_n]
            A_slice = list_A[A_dim][:, n_range]
            c[n] = np.einsum("ij,kj->ik", F_slice, A_slice).T
        return c
    
################################################################################################################################################################################
# We define function to build D tensors which are "factor tensors"
################################################################################################################################################################################

    def build_D(c1, c2):
        return c1[:, :, :, None] * c2[:, :, None, :]
    
################################################################################################################################################################################
    
    gamma = 1.141376e+2 / 6.4e-2 * (15.0 / 41.0) ** 3
    
    #################################################################################
    #############       First we load up Collision Kernel A          ################
    #################################################################################

    # Path to the saved .pkl file
    # NOTE pickle file path will change for different MM values
    file_path = '/Volumes/Devaney SSD/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/'
    filename = 'norm_A_CP_M41_R400.pkl'
    file_loc = os.path.join(file_path, filename)

    # Load the data
    with open(file_loc, "rb") as file:
        norm_A_CP = pickle.load(file)
    relative_error = norm_A_CP['relative_error']
    execution_time = norm_A_CP['execution_time']
    norm_A_CP = norm_A_CP['norm_A_CP']
    norm_A_CP[0] = norm_A_CP[0].numpy()
    norm_A_CP[1] = [factor.detach().cpu().numpy() for factor in norm_A_CP[1]]
    sigma = len(norm_A_CP[0]) # rank_A
    if sigma_dp == None:
        sigma_dp = sigma_p

    #norm_A_CP[0] is factor weights
    #norm_A_CP[1] is factor matrices
    list_A = [unfold(factor_matrix, 1) for factor_matrix in norm_A_CP[1]]

    ##################################################################################
    #############  Next we load up F1                               ##################
    ##################################################################################
    F1 = np.reshape(f[0], (MM, MM, MM), order='C') # NOTE is f alone? then change f to f[0]
    # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    F1_CP = parafac(F1, rank=sigma_p, init='random', normalize_factors=True) 
    list_F1 = [unfold(factor_matrix, 1) for factor_matrix in F1_CP[1]]
    #norm_F1_CP[0] is factor weights
    #norm_F1_CP[1] is factor matrices
 
    #################################################################################
    #############  Now we get F2                                    #################
    #################################################################################

    F2 = np.reshape(g[0], (MM, MM, MM), order='C') # NOTE is g alone? then change g to g[0]
    # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    F2_CP = parafac(F2, rank=sigma_dp, init='random', normalize_factors=True)
    list_F2 = [unfold(factor_matrix, 1) for factor_matrix in F2_CP[1]]
    #norm_F2_CP[0] is factor weights
    #norm_F2_CP[1] is factor matrices

    #####################################################################################
    ##############  Now we convolve A with F                               ##############
    #####################################################################################

    c1 = conv(list_F1, list_A, 0, 0, sigma, sigma_p, MM, 7) #NOTE n_anc=7 may change for different MM values
    c2 = conv(list_F1, list_A, 1, 1, sigma, sigma_p, MM, 7)
    c3 = conv(list_F1, list_A, 2, 2, sigma, sigma_p, MM, 7)
    c4 = conv(list_F2, list_A, 0, 3, sigma, sigma_dp, MM, 7)
    c5 = conv(list_F2, list_A, 1, 4, sigma, sigma_dp, MM, 7)
    c6 = conv(list_F2, list_A, 2, 5, sigma, sigma_dp, MM, 7)

    #################################################################################
    # Finally, build collision operator after performing CP with the factor         #
    # matrices and weight for D    ##################################################
    #################################################################################

    D1_tensor = build_D(c1, c4)
    D2_tensor = build_D(c2, c5)
    D3_tensor = build_D(c3, c6)

    omega = np.einsum('i,j,k->ijk', norm_A_CP[0], F1_CP[0], F2_CP[0]).ravel(order='C') # NOTE including normalize_factors=True requires we comment away norm and make for factor_matrix in norm to just CP
    D1 = reshape(D1_tensor, (MM, sigma * sigma_p * sigma_dp), 'C')
    D2 = reshape(D2_tensor, (MM, sigma * sigma_p * sigma_dp), 'C')
    D3 = reshape(D3_tensor, (MM, sigma * sigma_p * sigma_dp), 'C')
    fm_d = [D1, D2, D3]

    # Ensure fm_d is a valid list of factor matrices
    CP_D = (omega, fm_d)  # Format as (weights, factors)
    D_tensor = tl.cp_to_tensor(CP_D)  # Convert CP tensor to full tensor
    ColOper = reshape(D_tensor, (MM ** 3, ), 'C') # ColOper is collision operator in form (MM ** 3, )
    return gamma * ColOper
