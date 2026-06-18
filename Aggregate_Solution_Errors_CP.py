print("Script Launched.")

##############################
# Setup
##############################
MM = 41
Mtrim = 0
dms = MM - 2*Mtrim

import my_readwrite
import my_utils
import os, time, math, re, pickle, importlib
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

from michaels_utils import load_macro_file
import tensorly as tl
from tensorly import norm
from cpQ import cpQ
from my_distributions import maxwellian

tl.set_backend('pytorch')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################
# User Inputs
##############################
init_time = 0.0
fin_time = 0.0114
dt = 6e-4
runtime = init_time

base_data_dir = Path("F:/AFIT-ARCH/Nguyen_1/Data/111 New Training Data") #or 222 New Testing Data
output_dir = 'F:/BoltzmannData/Boltzmann_Thesis_MD/AGG_MOMS'
os.makedirs(output_dir, exist_ok=True)
##############################
# Maxwellian + encoder helper
##############################
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(
    "F:/BoltzmannData/Boltzmann_Thesis_MD/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat"
)
moments = np.array([[1,0,0,0,0.2]])
maxwell = maxwellian(moments, nodes_u, nodes_v, nodes_w)

top_level_dirs = sorted(
    [p for p in base_data_dir.iterdir() if p.is_dir()]
)
file_loc = 'F:/BoltzmannData/Boltzmann_Thesis_MD/CP A Pickle Files/norm_A_CP_M41_R400.pkl'
with open(file_loc, "rb") as file:
    norm_A_CP = pickle.load(file)
norm_A_CP = norm_A_CP['norm_A_CP']
A_weights = norm_A_CP[0].detach().clone().to(device=device, dtype=torch.float64)
A_factors = [factor.detach() for factor in norm_A_CP[1]]
sigma = len(A_weights) # rank_A
list_A = [factor_matrix.transpose(0, 1).to(torch.float64).to(device=device) for factor_matrix in A_factors]
rank = 10


num_save_sol = 10         # how many times solution is saved
num_eval_moments = 800    # how many times moments are tracked
delta_t_save_sol = (fin_time-init_time)/num_save_sol
delta_t_eval_moms = (fin_time-init_time)/num_eval_moments
############# SAVE INITIAL DATA and compute Moments
next_time_save_sol  = init_time + delta_t_save_sol
next_time_eval_moms = init_time + delta_t_eval_moms
##############################
# Output file
##############################
fin_time1 = str(fin_time).replace('.', 'p')
dt1 = str(dt).replace('.', 'p')


##############################
# Main Loop
##############################
j = 0

num_valid_runs = 0
sum_total_errors = 0.0
error_log_path = os.path.join(output_dir, f"MOM_ERR_@{fin_time1}_DT{dt1}_fgCPrank{rank}.txt")
moment_labels = [
    "dens","u","v","w","T","Tx","Ty","Tz",
    "mom3u","mom3v","mom3w",
    "mom4u","mom4v","mom4w",
    "mom5u","mom5v","mom5w",
    "mom6u","mom6v","mom6w"
]

with open(error_log_path, "w") as f:
    f.write("run_num " + " ".join(moment_labels) + "\n")

##############################
# Loop over runs
##############################
for subfolder in top_level_dirs:
    print(f"\nProcessing subfolder: {subfolder}")
    run_dirs = sorted(
            [
                p for p in subfolder.rglob("*")
                if p.is_dir() and (re.search(r"(\d+) copy", p.name) or re.search(r"(\d+)", p.name))
            ]
        )

    print(f"Found {len(run_dirs)} run directories")

    for run_path in run_dirs:
        match = re.search(r'(\d+)', run_path.name)
        if match:
            run_id = int(match.group(1))
        else:
            raise ValueError(f"Cannot determine run number from {run_path.name}")

        macro_files = list(run_path.glob("*_macroerr.txt"))
        if not macro_files:
            print(f"No macroerr file for {run_path.name}")
            continue

        true_data = load_macro_file(macro_files[0])

        times = true_data[:, 0]

        idx = np.where(np.isclose(times, fin_time, atol=1e-12))[0]
        if len(idx) == 0:
            print(f"Skipping {run_path.name}: no time {fin_time} in macro file")
            continue

        # match predicted moment dimension (6 moments)
        true_moments = true_data[idx[0], 1:21]

        dat_files = list(run_path.glob("*time0.0000000000_SltnColl.dat"))
        if not dat_files:
            continue
        datapath = str(dat_files[0])
        runtime = init_time
        sol, solsize = my_readwrite.my_read_solution_trim(datapath, MM, Mtrim)        
        sol_utm, _ = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
        rec_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,runtime)

        
        ##############################
        # Time stepping
        ##############################
        #    sol_utm, _ = my_readwrite.solution_untrim(sol[0,:], MM-2*Mtrim, Mtrim)
        #    pred = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, runtime)
        #    pred_moments = pred[-1,1:21]
        #    print(true_moments)
        #    print(pred_moments)
        #    exit()
        
        
        n_steps = int(round(fin_time / dt))
        
        output_filename = f"moments_{run_id}_ft-{fin_time1}_ts-{dt1}_fgCPrank{rank}.txt"
        output_path = os.path.join(output_dir, output_filename)
        print(f"Processing {run_path.name} ...")
        
        for i in range(n_steps):
            f1 = sol + maxwell
            f2 = sol - maxwell
            coll_oper = cpQ(f1, f2, A_weights, list_A, MM, device, sigma, rank, g_rank = None).cpu().numpy()
            sol[0,:] += dt * coll_oper
            #    delf = sol - maxwell
            #    q_dot_delf = np.sum(coll_oper * delf) #we may want to include nodes_gwts eventually however at the present when we do it causes our plotted solutions to look bad
            #    delf_delf = np.sum(delf * delf * nodes_gwts) # def seems to need nodes for proper convergence
                    
            
            #    if Lyapunov_Constraints:
            #        sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu

            #    if Lyapunov_Constraints:
            #        Qlogf = np.sum(coll_oper * np.log(np.maximum(sol, 1e-16)))
            #        flogf = np.sum(sol * np.log(np.maximum(sol, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
            #        fMlogfM = np.sum(maxwell * np.log(np.maximum(maxwell, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
            #        flogdif = np.sum(sol * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
            #        dflogdif = np.sum(delf * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
            #        sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogdif), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu
            runtime += dt
            ## Now we check if it is time to record the moments
            sol_utm, _ = my_readwrite.solution_untrim(
                sol[0,:],
                MM-2*Mtrim,
                Mtrim
            )

            entry_moments = my_utils.get_moments(
                sol_utm,
                nodes_u,
                nodes_v,
                nodes_w,
                nodes_gwts,
                runtime
            )

            rec_moments = np.concatenate(
                (rec_moments, entry_moments),
                axis=0
            )

        print(f"Finished time stepping for {run_path.name}. Now computing errors.")
        runtime1 = n_steps * dt

        ## Add here subroutine to save solution ##
        my_readwrite.save_moments(output_path, rec_moments)
        ##############################
        # Predicted moments
        ##############################
        sol_utm, _ = my_readwrite.solution_untrim(sol[0,:], MM-2*Mtrim, Mtrim)
        pred = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, runtime1)
        if pred.shape[1] < 21:
            raise ValueError(f"Predicted moments only have {pred.shape[1]} columns")
        pred_moments = pred[-1,1:21]   # drop time column

        ##############################
        # Mixed absolute/relative error
        ##############################
        if pred_moments.shape != true_moments.shape:
            print(f"Skipping {run_path.name}: shape mismatch {pred_moments.shape} vs {true_moments.shape}")
            continue

        errors = np.zeros_like(pred_moments)

        # Density should be compared against exact value 1
        errors[0] = np.abs(pred_moments[0] - 1.0)

        # u,v,w should be compared against exact value 0
        errors[1:4] = np.abs(pred_moments[1:4])

        # Everything else uses relative error against macroerr.txt values
        errors[4:] = (
            np.abs(pred_moments[4:] - true_moments[4:])
            / (np.abs(true_moments[4:]) + 1e-14)
        )

        # Overall model score:
        # use same mixed convention
        true_reference = np.copy(true_moments)

        # exact targets for first four moments
        true_reference[0] = 1.0
        true_reference[1:4] = 0.0

        diff = pred_moments - true_reference

        # absolute error for first four
        scaled_diff = np.zeros_like(diff)
        scaled_diff[0:4] = diff[0:4]

        # relative error for remaining moments
        scaled_diff[4:] = diff[4:] / (np.abs(true_moments[4:]) + 1e-14)

        tot_rel_error = np.linalg.norm(scaled_diff)

        sum_total_errors += tot_rel_error
        num_valid_runs += 1
        
        ##############################
        # Save
        ##############################
        with open(error_log_path, "a") as f:
            f.write(f"{run_id} " + " ".join([f"{e:.6e}" for e in errors]) + "\n")
print("\nDONE.")
