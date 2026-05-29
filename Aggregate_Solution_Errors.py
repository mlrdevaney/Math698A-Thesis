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

from AE_DeltaF_Class_v4 import Autoencoder
from my_distributions import maxwellian
from michaels_utils import parse_run_config, count_parameters, str2bool, hermite_projection_np, huber_np, build_ksis_np, get_encoded_delta_f, load_macro_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##############################
# User Inputs
##############################
init_time = 0.0
fin_time = 0.0114
dt = 6e-4
runtime = init_time

umb_folder = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CNNs"
folder = os.path.join(umb_folder, "AAA test folder")
base_data_dir = Path("/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/222 New Testing Data/sphomruns testing half")
output_dir = os.path.join(umb_folder, 'AGG_MOMS')
os.makedirs(output_dir, exist_ok=True)
##############################
# Maxwellian + encoder helper
##############################
nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(
    'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
)
moments = np.array([[1,0,0,0,0.2]])
maxwell = maxwellian(moments, nodes_u, nodes_v, nodes_w)

cnn_models = [p.name for p in Path(folder).iterdir() if p.is_dir()]
run_dirs = sorted([p for p in base_data_dir.iterdir() if p.is_dir()])



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
all_models = np.zeros(len(cnn_models))
alpha = 1.0
d = 1.0
beta_coeff = 1.0
eps = 1e-12
for CNN_model in cnn_models:

    print(f"\n===== MODEL: {CNN_model} =====")

    cnn_weights_directory = f"{folder}/{CNN_model}/LearnColOpCNNWeights.pt"
    cnn_config_path = os.path.join(folder, CNN_model, "run_config.txt")
    cnn_config = parse_run_config(cnn_config_path)

    Use_AE = str2bool(cnn_config.get("Use_AE", "1"))
    Bias = str2bool(cnn_config.get("Bias", "0"))
    SVD_Cons = str2bool(cnn_config.get("SVD_Cons", "0"))
    Herm_Cons = str2bool(cnn_config.get("Herm_Cons", "0"))
    Lyapunov_Constraints = str2bool(cnn_config.get("Lyapunov_Constraints", "0"))

    cnn_class_version = cnn_config.get("Class_Version") or 16
    module = importlib.import_module(f"CNN_CollOp_Class_v{cnn_class_version}")
    LearnColOpCNN = module.LearnColOpCNN

    ##############################
    # Load AE once per model
    ##############################
    AE_model_path = f"/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/{cnn_config.get('AE_model_path')}/LearnSolWeights.pt"
    ae_config = parse_run_config(os.path.join(os.path.dirname(AE_model_path), "run_config.txt"))

    code_len = int(ae_config.get("Code_Length", 64))
    hidden_layers = int(ae_config.get("Hidden_Layers", 3))

    AE_model = Autoencoder(MM**3, code_len, hidden_layers)
    AE_model.load_state_dict(torch.load(AE_model_path, map_location=device))
    AE_model.to(device).double().eval()

    dms_enc = int(round(code_len ** (1/3)))

    ##############################
    # Load CNN once
    ##############################
    dms_cnn = dms_enc if Use_AE else dms
    CNN = LearnColOpCNN(dms=dms_cnn, output_dim=MM**3, bias=Bias).to(device)
    CNN.load_state_dict(torch.load(cnn_weights_directory, map_location=device))
    CNN = CNN.double().eval()

    num_valid_runs = 0
    sum_total_errors = 0.0
    error_log_path = os.path.join(folder, f"MOM_ERR_@{fin_time1}_DT{dt1}_{CNN_model}.txt")
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
    for _, run_path in enumerate(run_dirs, start=1):
        match = re.search(r'/(\d+) copy', str(run_path))
        if match:
            run_id = int(match.group(1))
        else:
            raise ValueError("Cannot determine run number.")

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

        # match your predicted moment dimension (6 moments)
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
##        sol_utm, _ = my_readwrite.solution_untrim(sol[0,:], MM-2*Mtrim, Mtrim)
##        pred = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, runtime)
##        pred_moments = pred[-1,1:21]
##        print(true_moments)
##        print(pred_moments)
##        exit()
        
        ksis = build_ksis_np(nodes_u, nodes_v, nodes_w, nodes_gwts, maxwell) # (N, 5)
        
        n_steps = int(round(fin_time / dt))
        
        output_filename = f"AAA_moments_{run_id}_ft-{fin_time1}_ts-{dt1}_{CNN_model}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        for i in range(n_steps):
            delta_f = get_encoded_delta_f(sol, maxwell, AE_model, Use_AE, dms_enc, dms, device)
            coll_oper = CNN(delta_f).detach().cpu().numpy().squeeze()
##            if Herm_Cons:
##                coll_oper = hermite_projection_np(coll_oper, maxwell, ksis, nodes_gwts * math.sqrt(2/0.2))

            sol[0,:] += dt * coll_oper
##            delf = sol - maxwell
##            q_dot_delf = np.sum(coll_oper * delf) #we may want to include nodes_gwts eventually however at the present when we do it causes our plotted solutions to look bad
##            delf_delf = np.sum(delf * delf * nodes_gwts) # def seems to need nodes for proper convergence
##                  
##            
##            if Lyapunov_Constraints:
##                sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu
##
##            if Lyapunov_Constraints:
##                Qlogf = np.sum(coll_oper * np.log(np.maximum(sol, 1e-16)))
##                flogf = np.sum(sol * np.log(np.maximum(sol, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
##                fMlogfM = np.sum(maxwell * np.log(np.maximum(maxwell, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
##                flogdif = np.sum(sol * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
##                dflogdif = np.sum(delf * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
##                sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogdif), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu
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
    
##        print(true_moments)
##        print(pred_moments)
##        exit()

        ##############################
        # Relative error
        ##############################
        if pred_moments.shape != true_moments.shape:
            print(f"Skipping {run_path.name}: shape mismatch {pred_moments.shape} vs {true_moments.shape}")
            continue
        rel_error = np.abs(pred_moments - true_moments) / (np.abs(true_moments) + 1e-14)
        tot_rel_error = np.linalg.norm(pred_moments - true_moments) / (np.linalg.norm(true_moments) + 1e-14)
        sum_total_errors += tot_rel_error
        num_valid_runs += 1
        
        ##############################
        # Save
        ##############################
        with open(error_log_path, "a") as f:
            f.write(f"{run_id} " + " ".join([f"{e:.6e}" for e in rel_error]) + "\n")
    if num_valid_runs > 0:
        model_err = sum_total_errors / num_valid_runs
    else:
        model_err = np.nan

    all_models[j] = model_err
    j += 1
##    print(f"Model error {model_err*100:.3f}%")

print(all_models)
print("\nDONE.")
