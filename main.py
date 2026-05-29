print("Script Launched.")
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

###########################################################################
####  First we Set up mesh sizes and trimming: Copy and paste from the main.py
####  used to train the model above.
####
MM = 41
Mtrim = 0
dms=MM-2*Mtrim
####
####  END seeting up mesh sizes and trim ####################################

Sparse_AE = False
if Sparse_AE:
    divisor = 1000

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
import my_readwrite   ## import the entire module
import my_utils
import os
import time
import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch
import re
##from AE_DeltaF_Class_v3 import Autoencoder
from AE_DeltaF_Class_v4 import Autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from my_distributions import maxwellian
from datetime import datetime, timedelta
import pickle
import importlib
from michaels_utils import parse_run_config, count_parameters, str2bool, hermite_projection, relu, huber_np, plot_moments, build_ksis

fin_time = 0.3   # final time
run_list = [488]
cons_set = [True]
stab_df_set = [False, True]
stab_entrop_set = [False, True]
d = 1.0 
alpha = 1e0 # works for both lyapunov's
eps = 1e-12 # works for both lyapunov's
folder = "/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CNNs/AAA test folder"
test_folder = Path(folder)
cnn_models = [p.name for p in test_folder.iterdir() if p.is_dir()]
dt = 0.0001 # time step normally 0.0001
##dt = dt / (2 ** n)
seconds_per_run = fin_time / (dt * 9)
total_seconds = seconds_per_run * len(run_list) * len(cnn_models) * (len(cons_set) * len(stab_df_set) * len(stab_entrop_set))
completion_time = datetime.now() + timedelta(seconds=total_seconds)
print(f"Projected completion time: {completion_time.strftime('%H:%M:%S')}")


##idx = 0

##for n in range(idx+1):
for enf_lyapunov_entropy in stab_entrop_set:
    for enf_lyapunov in stab_df_set:
        if (enf_lyapunov_entropy and enf_lyapunov) or not (enf_lyapunov_entropy or enf_lyapunov):
            continue
        for Herm_Cons in cons_set:
            for run_ID in run_list:
                for CNN_model in cnn_models:
                    print("\n=================================================================")
                    print(f"Run {run_ID} for {CNN_model} started @ {datetime.now().strftime('%H:%M:%S')}")
                    #n = 0 # [0, 4] ends inclusive
                    path = f'/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/222 New Testing Data/sphomruns testing half/{run_ID} copy'
                    sol_coll_filename = f'CollTrnDta{run_ID}_2kc1su1sv1sw3NXU41MuUU41MvVU41MwWU_time0.0000000000_SltnColl.dat'
                    
                    cnn_weights_directory = f"{folder}/{CNN_model}/LearnColOpCNNWeights.pt"
                    datapath = os.path.join(path, sol_coll_filename)

                    #############################################
                    ### Parse CNN model info
                    #############################################
                    CNN_dir = os.path.dirname(cnn_weights_directory)
                    cnn_config_path = os.path.join(CNN_dir, "run_config.txt")
                    cnn_config = parse_run_config(cnn_config_path)

                    SVD_Cons = str2bool(cnn_config.get("SVD_Cons", "0"))
                    Weak_Projection = str2bool(cnn_config.get("Weak_Projection", "0"))
                    Bias = str2bool(cnn_config.get("Bias", "0"))
                    Use_AE = str2bool(cnn_config.get("Use_AE", "1"))
                    dropout = cnn_config.get("dropout_rate")
                    cnn_class_version = cnn_config.get("Class_Version") #NOTE
                    if cnn_class_version is None:
                        cnn_class_version = 16
                    module_name = f"CNN_CollOp_Class_v{cnn_class_version}"
                    module = importlib.import_module(module_name)
                    LearnColOpCNN = module.LearnColOpCNN
                    if cnn_class_version == 17:
                        from CNN_CollOp_Class_v17 import LearnColOpCNN, smooth_huber_relu, ICNN, Joint_Lyapunov_CNN_ICNN
                    AE_model = cnn_config.get("AE_model_path")
                    match = re.search("AEs", AE_model)
                    if not match:
                        AE_model = os.path.join("AEs", AE_model)

                    # let us load initial data
                    sol, solsize = my_readwrite.my_read_solution_trim(datapath,MM, Mtrim)  # this is the first file
                    ## Retrieves solution as a numpy array in the shape [1,solsize]

                    # for bookkeeping purposes, we also need meshes from our dicscrete solution.
                    # specifically, we need arrays of velocity points and gauss weights. these
                    # points can be obtained from the DG-Boltzmann solution saves:
                    filename='CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
                    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
                    #########################################################################


                    ######################################################################
                    ### setting up the Euler time stepping, macroparameter evaluations and
                    ### solutions saves.
                    init_time = 0.0       # initial time
                    ### some constants defining moments tracking and checkpoint saves:
                    num_save_sol = 10         # how many times solution is saved
                    num_eval_moments = 800    # how many times moments are tracked
                    delta_t_save_sol = (fin_time-init_time)/num_save_sol
                    delta_t_eval_moms = (fin_time-init_time)/num_eval_moments
                    ######
                    runtime=init_time

                    ############# SAVE INITIAL DATA and compute Moments
                    sol_utm, lsize = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
                    rec_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,runtime)

                    next_time_save_sol  = init_time + delta_t_save_sol
                    next_time_eval_moms = init_time + delta_t_eval_moms
                    ############ end save initial data


                    ######################################################################
                    #### THE MAIN LOOP
                    ############################################################################
                    ############################################################################
                    # ADD NEW STUFF IN THIS BELOW CHUNK ############################################################################
                    ############################################################################
                    # Convert initial time to string (e.g., 0.0 -> "0p0")
                    fin_time_str = str(fin_time).replace('.', 'p')
                    
                    #############################################
                    ### Parse AE model info
                    #############################################
                    AE_model_path = f"/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/{AE_model}/LearnSolWeights.pt"
                    AE_model_dir = os.path.dirname(AE_model_path)
                    ae_config_path = os.path.join(AE_model_dir, "run_config.txt")
                    ae_config = parse_run_config(ae_config_path)

                    if "Code_Length" in ae_config and "Hidden_Layers" in ae_config:
                        code_len = int(ae_config["Code_Length"])
                        hidden_layers = int(ae_config["Hidden_Layers"])
            ##            print(f"Loaded AE config from run_config.txt: code_len={code_len}, hidden_layers={hidden_layers}")
                    else:
                        print("No AE run_config found — falling back to regex extraction.")
                        match = re.search(r'CL(\d+)', AE_model_path)
                        if match:
                            code_len = int(match.group(1))
                        else:
                            code_len = 64
                        match = re.search(r'HL(\d+)', AE_model_path)
                        if match:
                            hidden_layers = int(match.group(1))
                        else:
                            hidden_layers = 3

                    AE_model = Autoencoder(solsize, code_len, hidden_layers)
                    AE_model.load_state_dict(torch.load(AE_model_path, map_location=device))
                    AE_model.to(device).double().eval()
                    dms_enc = int(round(code_len ** (1 / 3)))
                    
                    AE_used_for_CNN = cnn_config.get("AE_model_path")  # no colon, no trailing space

                    if AE_used_for_CNN is None:
                        print("⚠️ AE model path missing in CNN run_config.txt — skipping check")
                    else:
                        match_ae_using = re.search(r'delf-AE-(\d{2}_\d{2}_\d{4}-\d{2}_\d{2}_\d{2})', AE_model_path)
                        match_ae_used = re.search(r'delf-AE-(\d{2}_\d{2}_\d{4}-\d{2}_\d{2}_\d{2})', AE_used_for_CNN)
                        if match_ae_using and match_ae_used:
                            ae_used = match_ae_used.group(1)
                            ae_using = match_ae_using.group(1)
                            if ae_used != ae_using:
                                raise ValueError(f"Mismatch: CNN config expects AE {ae_used}, "
                                                 f"but AE model path is {ae_using}")
                        else:
                            print("⚠️ Could not parse AE datetime from CNN run_config.txt")

                    moments=np.zeros((1,5))
    ##                moments[0,0] = np.sum(sol * nodes_gwts)
    ##                moments[0,1] = np.sum(sol * nodes_u * nodes_gwts) / moments[0,0]
    ##                moments[0,2] = np.sum(sol * nodes_v * nodes_gwts) / moments[0,0]
    ##                moments[0,3] = np.sum(sol * nodes_w * nodes_gwts) / moments[0,0]
    ##                moments[0,4] = np.sum(sol * nodes_gwts * ((nodes_u - moments[0,1]) ** 2
    ##                                                              + (nodes_v - moments[0,2]) ** 2
    ##                                                              + (nodes_w - moments[0,3]) ** 2))\
    ##                                   / moments[0,0] / 3.0 * 2.0
                    moments[0,0]=1
                    moments[0,1]=0
                    moments[0,2]=0
                    moments[0,3]=0
                    moments[0,4]=0.2
                    
                    maxwell = maxwellian(moments, nodes_u, nodes_v, nodes_w)
                    

                    def get_encoded_delta_f(sol):
                        df = sol - maxwell
                        df = torch.tensor(df, dtype=torch.float64).to(device)
                        if Use_AE:
                            df = AE_model.encoder(df)
                            return df.view(1, 1, dms_enc, dms_enc, dms_enc)
                        else:
                            return df.view(1, 1, dms, dms, dms)
                            

                    nodes_path = 'CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
                    nodes1_u, nodes1_v, nodes1_w, nodes1_gwts = my_readwrite.read_nodes(nodes_path)
                    nodes1_u = torch.tensor(nodes1_u, dtype=torch.float64).to(device)
                    nodes1_v = torch.tensor(nodes1_v, dtype=torch.float64).to(device)
                    nodes1_w = torch.tensor(nodes1_w, dtype=torch.float64).to(device)
                    nodes1_gwts = torch.tensor(nodes1_gwts, dtype=torch.float64).to(device)
                    maxwell1 = torch.reshape(torch.tensor(maxwell, dtype=torch.float64), (1, -1))
                    ksis = build_ksis(nodes1_u, nodes1_v, nodes1_w, nodes1_gwts, maxwell1) # (B, N, 5)
                    gwts = (nodes1_gwts * math.sqrt(2/0.2))

                    if SVD_Cons or Weak_Projection:
                        with open('SVD_U_M41.pkl', 'rb') as svd_file:
                            U = pickle.load(svd_file)
                        U = U.to(device).double()
                    U_model = U if SVD_Cons else None

                    if Use_AE:
                        dms_cnn = dms_enc
                    else:
                        dms_cnn = dms
                    CNN = LearnColOpCNN(dms=dms_cnn, output_dim=solsize, bias=Bias).to(device)
                    CNN.load_state_dict(torch.load(cnn_weights_directory, map_location=device))
                    CNN = CNN.double()
                    CNN.eval()
                    # Define output directory and filename
                    output_dir = f"{folder}/{CNN_model}/moments"
                    os.makedirs(output_dir, exist_ok=True)
                    dtp = str(dt).replace('.', 'p')
                    output_filename = f"AAA_moments_{run_ID}_ft-{fin_time_str}_ts-{dtp}_H{int(Herm_Cons)}L{int(enf_lyapunov)}E{int(enf_lyapunov_entropy)}.txt"
                    output_path = os.path.join(output_dir, output_filename)        
                    ae_params = count_parameters(AE_model)
                    print(f"Number of trainable parameters in the AE model: {ae_params:,}")
                    cnn_params = count_parameters(CNN)
                    print(f"Number of trainable parameters in the CNN model: {cnn_params:,}")
                    if Sparse_AE:
                        count = 0
            ##        running = 0
                    coll_oper_dot_delf = []
                    coll_oper_norm = []
                    delta_f_norm = []
                    lyp_dot = []
                    loop_start = time.time()
                    ############################################################################
                    #############################################################################
                    ############################################################################
                    print("Beginning Euler Time-Stepping...")
                    start = time.process_time()
                    while runtime < fin_time:
            ##            iter_start = time.time()
                        ############################################################################
                        ######## Record converved moments ###########
                        conserved_macroparams=my_utils.compute_conservative_moments(sol, nodes_u, nodes_v, nodes_w, nodes_gwts, MM, Mtrim)
                        delta_f = get_encoded_delta_f(sol)
                        coll_oper = CNN(delta_f)
                        if SVD_Cons:
                            coll_oper -= (coll_oper @ U) @ U.T
                        elif Herm_Cons: # this is something we want to toggle on/off even if it wasnt trained with Hermite to begin with
                            coll_oper = hermite_projection(coll_oper, maxwell, ksis, nodes_gwts * math.sqrt(2/0.2))
                        else:
                            coll_oper = coll_oper
                        coll_oper = coll_oper.detach().cpu().numpy().squeeze()
                        ##################################
                        ## enforce conservation of mass on collision operator.
                        enf_conservation_mass = False
                        if enf_conservation_mass:
                            coll_oper, coll_op_size = my_utils.enf_conservation(coll_oper, nodes_u, nodes_v, nodes_w, nodes_gwts,MM,Mtrim)
                        ## end enforce conservation of mass
                        #################################
                        sol[0,:] = sol[0,:] + dt*beta_coeff*coll_oper
                        delf = sol - maxwell
                        q_dot_delf = np.sum(coll_oper * delf) #we may want to include nodes_gwts eventually however at the present when we do it causes our plotted solutions to look bad
                        delf_delf = np.sum(delf * delf * nodes_gwts) # def seems to need nodes for proper convergence
                            
                        if enf_lyapunov:
##                            sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu
                            sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(q_dot_delf + alpha * delf_delf, d) / (delf_delf + eps) * delf # trick is to use np.logaddexp not huber nor relu

                        if enf_lyapunov_entropy:
                            Qlogf = np.sum(coll_oper * np.log(np.maximum(sol, 1e-16)))
                            flogf = np.sum(sol * np.log(np.maximum(sol, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
                            fMlogfM = np.sum(maxwell * np.log(np.maximum(maxwell, 1e-16)) * nodes_gwts) # def seems to need nodes for proper convergence
                            flogdif = np.sum(sol * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
                            dflogdif = np.sum(delf * (np.log(np.maximum(sol, 1e-16)) - np.log(np.maximum(maxwell, 1e-16))) * nodes_gwts) # def seems to need nodes for proper convergence
##                            sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogf - fMlogfM), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu
                            sol[0,:] = sol[0,:] - dt*beta_coeff * huber_np(Qlogf + alpha * (flogdif), d) / (dflogdif + eps) * delf # trick is to use np.logaddexp not huber nor relu
                                
                        runtime=runtime+dt

                        norm_delf = np.linalg.norm(delf)
                        norm_q = np.linalg.norm(coll_oper)
##                        coll_oper_dot_delf.append(q_dot_delf)
##                        lyp_dot.append(delf_delf / q_dot_delf)
                        coll_oper_norm.append(norm_q)
                        delta_f_norm.append(norm_delf)
                        
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
                        if runtime>next_time_eval_moms:
                            sol_utm, iizizer = my_readwrite.solution_untrim(sol[0,:],MM-2*Mtrim,Mtrim)
                            entry_moments = my_utils.get_moments(sol_utm,nodes_u, nodes_v, nodes_w, nodes_gwts,runtime)
                            rec_moments = np.concatenate((rec_moments, entry_moments), axis = 0)
                            next_time_eval_moms =next_time_eval_moms+delta_t_eval_moms

                        #############################################
                        ## Add here subroutine to save solution
                        ##
                        if runtime>next_time_save_sol:
                            my_readwrite.save_moments(output_path,rec_moments)
                            next_time_save_sol=next_time_save_sol+delta_t_save_sol
    ##                    if Sparse_AE:
    ##                        count += 1
    ##                        if count % divisor == 0:
    ##                            def AE_delta_f(sol):
    ##                                df = sol - maxwell
    ##                                df = torch.tensor(df, dtype=torch.float64).to(device)
    ##                                df = AE_model(df)
    ##                                return df.detach().cpu().numpy()
    ##                            normb4 = np.linalg.norm(sol)
    ##                            print("Norm Before Using Sparse AE:", normb4)
    ##                            sol = maxwell + AE_delta_f(sol) # f=fM + (f-fM), hope here that by passing sol through full AE sparsely it may correct any issues of num. instabil.
    ##                            norm_after = np.linalg.norm(sol)
    ##                            print("Norm After Using Sparse AE:", norm_after)
    ##                            print("Percent of Solution Lost(-) or Gained(+) During Sparse AE'ing", -(normb4-norm_after)/normb4 * 100, "%")
                    ##    print(time.time()-iter_start, "sec")
            ##            running += time.time()-iter_start
            ##        print("Cumulative Run-Time:", running, "sec")
            ##        print("Average time per iteration using cumulative time per iteration:", running/count1)
                    end = time.process_time()
                    print(f"CPU processing time for Model: {CNN_model}; timesteps-{round(runtime/dt)} in {round(end - start, 3)} CPU sec")
##                    with open(f'Euler_Convergence_{run_ID}_{fin_time_str}_ts-{dt}_H_{int(Herm_Cons)}_L_{int(enf_lyapunov)}_LE_{int(enf_lyapunov_entropy)}.pkl', 'wb') as sol_file:
##                        pickle.dump(sol, sol_file, protocol=4)
                    total_runtime = time.time() - loop_start
                    print(f"Total run-time: {round(total_runtime, 3)} sec")
                    print(f"Average time per iteration: {round(total_runtime/(runtime/dt), 3)} sec")
                    print(f"Number of time-steps: {round(runtime/dt)}")
                    print(f"Run {run_ID} for {CNN_model} completed @ {datetime.now().strftime('%H:%M:%S')}")
                    ############ ALL done. Last save and quit
                    sol_utm, iisizer = my_readwrite.solution_untrim(sol[0,:], MM - 2 * Mtrim, Mtrim)
                    entry_moments = my_utils.get_moments(sol_utm, nodes_u, nodes_v, nodes_w, nodes_gwts, runtime)
                    rec_moments = np.concatenate((rec_moments, entry_moments), axis=0)
                    my_readwrite.save_moments(output_path,rec_moments)
                    path = Path(path)
                    ##os.system(f'open "{output_path}"')
                    matching_files = [f for f in path.glob('*_macroerr.txt') if not f.name.startswith("._")]
                    for filepath in matching_files:
                        truemacroerr = filepath
                    ##    print("Opening:", filepath)
                    ##    os.system(f'open "{truemacroerr}"')  # This opens the file in the default app (usually TextEdit)
                    plot_moments(output_path, truemacroerr, "Tx", "Ty", run_ID=run_ID, CNN_model=CNN_model, cons=Herm_Cons, lyap=enf_lyapunov, lyap_entrop=enf_lyapunov_entropy)

##                    plt.figure(figsize=(8,6))
##                    plt.plot(range(len(coll_oper_dot_delf)), coll_oper_dot_delf, marker='o', label=r'$\langle Q,\delta f \rangle$ vs timestep')
##                    plt.xlabel('Time-Step')
##                    plt.ylabel(r'$\langle Q,\delta f \rangle$')
##                    plt.title(f'{CNN_model}')
##                    plt.grid(True)
##                    plt.legend()
##                    plot_path = os.path.join(output_dir, f'dot_prod_{run_ID}_{fin_time_str}_H{int(Herm_Cons)}L{int(enf_lyapunov)}E{int(enf_lyapunov_entropy)}.png')
##                    plt.savefig(plot_path)
##                    plt.close()

                    plt.figure(figsize=(8,6))
                    plt.plot(range(len(coll_oper_norm)), coll_oper_norm, marker='o', label=r'$\|Q\|$ vs timestep')
                    plt.xlabel('Time-Step')
                    plt.ylabel(r'$\|Q\|$')
                    plt.title(f'{CNN_model}')
                    plt.grid(True)
                    plt.legend()
                    plot_path = os.path.join(output_dir, f'q_norm_{run_ID}_{fin_time_str}_H{int(Herm_Cons)}L{int(enf_lyapunov)}E{int(enf_lyapunov_entropy)}.png')
                    plt.savefig(plot_path)
                    plt.close()

                    plt.figure(figsize=(8,6))
                    plt.plot(range(len(delta_f_norm)), delta_f_norm, marker='o', label=r'$\|\delta f\|$ vs timestep')
                    plt.xlabel('Time-Step')
                    plt.ylabel(r'$\|\delta f\|$')
                    plt.title(f'{CNN_model}')
                    plt.grid(True)
                    plt.legend()
                    plot_path = os.path.join(output_dir, f'delf_norm_{run_ID}_{fin_time_str}_H{int(Herm_Cons)}L{int(enf_lyapunov)}E{int(enf_lyapunov_entropy)}.png')
                    plt.savefig(plot_path)
                    plt.close()

##                    plt.figure(figsize=(8,6))
##                    plt.plot(delta_f_norm, lyp_dot, marker='o', label=r'$\|\delta f\| ** 2 / \langle Q,\delta f \rangle$ vs $\|\delta f\|$')
##                    plt.xscale("log")
##                    plt.axhline(0.0, linestyle="--")
##                    plt.xlabel(r"$\|\delta f\|$ (log scale)")
##                    plt.ylabel(r"$\|\delta f\| ** 2 / \langle Q,\delta f \rangle$")
##                    plt.title(f'{CNN_model}')
##                    plt.grid(True, which="both")
##                    plt.legend()
##                    plot_path = os.path.join(output_dir, f'LYP_{run_ID}_{fin_time_str}_H{int(Herm_Cons)}L{int(enf_lyapunov)}E{int(enf_lyapunov_entropy)}.png')
##                    plt.savefig(plot_path)
##                    plt.close()

