print("Script launched")
##############################
###  08/14/2019 A. Alekseenko
###  This is main program for solving the problem of spatially homogeneous
###  rellaxation using learned collision operator.
###  The collision operator is trained offline. In this code we just set up the
###  network and load the weights.
###  Then the network is used to interpolate the values of the
###  collision operator
###
###  Euler time step is taken to advance the solution in time
###
###  Subroutines are added to enforce conservation laws.
##############################

#########################################################################
#### Small but important: a bunch of constants to match to the
#### deterministic solver for the 0D Boltzmann equation
beta_coeff = 1.0
#### End bunch of constants for Boltzmann
#########################################################################

#########################################################################
#### NEXT We read the initial data                              #########
####
##############
from pathlib import Path

import my_readwrite   ## import the entire module
import my_utils
import numpy as np
import os
import re
import time
import torch
torch.cuda.empty_cache()

###########################################################################
####  First we Set up mesh sizes and trimming: Copy and paste from the main.py
####  used to train the model above.
####
MM = 41
Mtrim = 0
dms=MM-2*Mtrim
####
####  END seting up mesh sizes and trim ####################################

run_id = 519
path = f'F:/BoltzmannData/RUNS M=41/sphomruns/good/{run_id}'
id_filename = f'CollTrnDta{run_id}_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat'
filepath = os.path.join(path, id_filename)

# let us load initial data
sol, coll, solsize = my_readwrite.my_read_sol_coll_trim(filepath,MM, Mtrim)  # this is the first file
## Retrieves solution as a numpy array in the shape [1,solsize]

# for bookkeeping purposes, we also need meshes from our dicscrete solution.
# specifically, we need arrays of velocity points and gauss weights. these
# points can be obtained from the DG-Boltzmann solution saves:
filename = 'F:/BoltzmannData/Boltzmann_Thesis_MD/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
#########################################################################


######################################################################
### setting up the Euler time stepping, macroparameter evaluations and
### solutions saves.
init_time = 0.0       # initial time
fin_time = 0.05   # final time 
dt = 0.0001      # time step
### some constants defining moments tracking and checkpoint saves:
num_save_sol = 10         # how many times solution is saved
num_eval_moments = 800    # how many times moments are tracked
delta_t_save_sol = (fin_time-init_time)/num_save_sol
delta_t_eval_moms = (fin_time-init_time)/num_eval_moments
######
running=init_time

############# SAVE INITIAL DATA and compute Moments
sol_utm, lsize = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
rec_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,running)

next_time_save_sol  = init_time + delta_t_save_sol
next_time_eval_moms = init_time + delta_t_eval_moms
############ end save initial data


######################################################################
#### THE MAIN LOOP
############################################################################
############################################################################
# ADDED NEW STUFF IN THIS BELOW CHUNK ############################################################################
############################################################################
import math
import pickle
import torch
import tensorly as tl
from cpQ import cpQ
from my_distributions import maxwellian
from pathlib import Path
from michaels_utils import huber_np, plot_moments, hermite_projection_np, build_ksis_np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
# device = "cpu"

tl.set_backend('pytorch')



Herm_Cons = False
enf_lyapunov = False
enf_lyapunov_entropy = False
alpha = 1e0
eps = 1e-12
d = 1.0

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
maxwell = maxwellian(moments,nodes_u,nodes_v,nodes_w)
if Herm_Cons:
    ksis = build_ksis_np(nodes_u, nodes_v, nodes_w, nodes_gwts, maxwell) # (B, N, 5)
file_loc = 'F:/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/norm_A_CP_M41_R400.pkl'

# Load the data
with open(file_loc, "rb") as file:
    norm_A_CP = pickle.load(file)
# print(norm_A_CP['relative_error'])
norm_A_CP = norm_A_CP['norm_A_CP']
A_weights = norm_A_CP[0].detach().clone().to(device=device, dtype=torch.float64)
A_factors = [factor.detach() for factor in norm_A_CP[1]]
sigma = len(A_weights) # rank_A
list_A = [factor_matrix.transpose(0, 1).to(torch.float64).to(device=device) for factor_matrix in A_factors]
rank = 50
runid_match = re.search(r'Dta(\d+)_', id_filename)
run_id = runid_match.group(1) if runid_match else "unknown"
fin_time_str = str(fin_time).replace('.', 'p')
output_folder = os.path.join('F:/BoltzmannData/Boltzmann_Thesis_MD', 'moments')
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, f'{run_id}_moments_{fin_time_str}_Herm{Herm_Cons}_Lin_{enf_lyapunov}_Ent_{enf_lyapunov_entropy}.txt')
torch.cuda.synchronize()
start = time.time()
count = 0
############################################################################
#############################################################################
############################################################################

while running < fin_time:
    ############################################################################
    # ADDED NEW STUFF IN THIS BELOW CHUNK ############################################################################
    ############################################################################
    count += 1
    if count % 10 == 0:
        torch.cuda.synchronize()
        print(f"Average time per iteration for past {count} iterations: {(time.time() - start)/count:.3f}(sec).")
    ffm = sol + maxwell # NOTE try changing ffm and delf to both sol
    delf = sol - maxwell
    ############################################################################
    #############################################################################
    ############################################################################
    ######## Record converved moments ###########
    conserved_macroparams=my_utils.compute_conservative_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
    coll_oper = cpQ(ffm, delf, A_weights, list_A, MM, device, sigma, rank, g_rank = None).cpu().numpy()
    # print(np.linalg.norm(coll_oper - coll)/np.linalg.norm(coll))
    if Herm_Cons: # this is something we want to toggle on/off even if it wasnt trained with Hermite to begin with
        coll_oper = hermite_projection_np(coll_oper, maxwell, ksis, nodes_gwts * math.sqrt(2/0.2))
    else:
        coll_oper = coll_oper
##  sol_5d = np.reshape(sol, (1, dms, dms, dms, 1))
##  coll_oper_5d = learncollision.predict(sol_5d)
##  coll_oper = np.reshape(coll_oper_5d, (1, dms*dms*dms))
##  coll_oper = coll_oper*(max_val-min_val)+min_val
    ##################################
    ## enforce conservation of mass on collision operator.
    enf_conservation_mass = False
    if enf_conservation_mass:
        coll_oper, coll_op_size = my_utils.enf_conservation(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts,MM,Mtrim)
    ## end enforce conservation of mass
    #################################
    sol[0,:] = sol[0,:] + dt*beta_coeff*coll_oper #NOTE - here should normally be +
        
    if enf_lyapunov:
        q_dot_delf = np.sum(coll_oper * delf) #we may want to include nodes_gwts eventually however at the present when we do it causes our plotted solutions to look bad
        delf_delf = np.sum(delf * delf * nodes_gwts) # def seems to need nodes for proper convergence
        ##sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu
        sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu

    if enf_lyapunov_entropy:
        Qlogf = np.sum(coll_oper * np.log(np.maximum(sol, 1e-16)))
        flogf = np.sum(sol * np.log(np.maximum(sol, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
        fMlogfM = np.sum(maxwell * np.log(np.maximum(maxwell, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
        flogdif = np.sum(sol * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
        dflogdif = np.sum(delf * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
        ##sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogf - fMlogfM), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu
        sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogdif), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu

    running+=dt

##    ###################################
##    ## filtering using SVD of solutions
##    do_filter_using_SV = False
##    if do_filter_using_SV:
##        sol[0,:] = np.dot(np.dot(sol[0,:],svect.T),svect)
##    ## end filtering using SVD of solutions
##    ####################################

    ## enforce conservation of mass on solutions.
    enf_conservation_on_solutions = False
    if enf_conservation_on_solutions:
        sol, sol_size = my_utils.enf_conservation_sol(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim,conserved_macroparams)
    ## end enforce conservation of mass

    #############################################
    ## Now we check if it is time to record the moments
    if running>next_time_eval_moms:
        sol_utm, iizizer = my_readwrite.solution_untrim(sol[0,:],MM-2*Mtrim,Mtrim)
        entry_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,running)
        rec_moments = np.concatenate((rec_moments, entry_moments), axis = 0)
        next_time_eval_moms =next_time_eval_moms+delta_t_eval_moms

    #############################################
    ## Add here subroutine to save solution
    ##
    if running>next_time_save_sol:
        my_readwrite.save_moments(output_filename,rec_moments)
        next_time_save_sol=next_time_save_sol+delta_t_save_sol
torch.cuda.synchronize()
end = time.time()
print("Time of Looping:", end-start)
print("Number of iterations:", count)
############ ALL done. Last save and quit
sol_utm, iisizer = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
entry_moments = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, running)
rec_moments = np.concatenate((rec_moments, entry_moments), axis=0)
my_readwrite.save_moments(output_filename,rec_moments)
path = Path(path)
##os.system(f'open "{output_path}"')
matching_files = [f for f in path.glob('*_macroerr.txt') if not f.name.startswith("._")]
for filepath in matching_files:
    truemacroerr = filepath
##    print("Opening:", filepath)
##    os.system(f'open "{truemacroerr}"')  # This opens the file in the default app (usually TextEdit)
plot_moments(output_filename, truemacroerr, "Tx", "Ty", run_ID=run_id, cons=Herm_Cons, lyap=enf_lyapunov, lyap_entrop=enf_lyapunov_entropy, CNN_model=None)


