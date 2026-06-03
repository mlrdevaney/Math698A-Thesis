##########################################################################################################
# This module is to read the solution files from a folder that contains all the solutions and save
# the solution data and collision data files.
# These pickle files will be loaded by another module to do the model training and model evaluation.
# 
##########################################################################################################
 
import numpy as np
import my_readwrite
import os
import pickle
import Utilities
from my_readwrite import read_nodes, solution_trim


def NormalLoadMaxwellDelfCollData(path):
    import my_readwrite
    import re
    from my_distributions import maxwellian

    names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)

    # Load nodes once
    filename = '/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
    match = re.search(r'sw(\d+)MuUU', filename)
    if match:
        MM = int(match.group(1))
    else:
        raise ValueError("Could not determine MM.")
    Mtrim = 0
    # First file
    solarry, collarry, solsize = my_readwrite.my_read_sol_coll_trim(names[0], MM, Mtrim) # shape also (1, 41^3)
    moments = np.zeros((1, 5))
    moments[0, 0] = np.sum(solarry * nodes_gwts)
    moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
    moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
    moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
    moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                   (nodes_v - moments[0, 2]) ** 2 +
                                                   (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
    fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
    delta_f = solarry - fm

    # Initialize training data arrays
    delta_f_data_train = delta_f
    maxwellian_data_train = fm
    coll_data_train = collarry

    num_samples = len(names)
    for i in range(1, num_samples):
        if i % 10 == 0:
            print("Processing " + str(i) + " of " + str(num_samples))

        solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(names[i], MM, Mtrim) # shape also (1, 41^3) unaltered
        assert solsize1 == solsize

        moments = np.zeros((1, 5))
        moments[0, 0] = np.sum(solarry * nodes_gwts)
        moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
        moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
        moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
        moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                       (nodes_v - moments[0, 2]) ** 2 +
                                                       (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
        fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
        delta_f = solarry - fm

        delta_f_data_train = np.concatenate((delta_f_data_train, delta_f), axis=0)
        maxwellian_data_train = np.concatenate((maxwellian_data_train, fm), axis=0)
        coll_data_train = np.concatenate((coll_data_train, collarry), axis=0)

    ########################################################################
    #               NORMALIZATION (dataset-wide)
    ########################################################################

    # Avoid extremely small std (stable division)
    eps = 1e-12

    # delta f
    df_mean = np.mean(delta_f_data_train)
    df_std = np.std(delta_f_data_train) + eps
    delta_f_data_train = (delta_f_data_train - df_mean) / df_std

    # maxwellians
    fm_mean = np.mean(maxwellian_data_train)
    fm_std = np.std(maxwellian_data_train) + eps
    maxwellian_data_train = (maxwellian_data_train - fm_mean) / fm_std

    # collision operator
    coll_mean = np.mean(coll_data_train)
    coll_std = np.std(coll_data_train) + eps
    coll_data_train = (coll_data_train - coll_mean) / coll_std

    return delta_f_data_train, maxwellian_data_train, coll_data_train, \
            (df_mean, df_std, fm_mean, fm_std, coll_mean, coll_std)

##    return delta_f_data_train, maxwellian_data_train, coll_data_train



def NormalCreateMaxwellDelfCollData(folder_pull_from, delta_f_FilePath, maxwell_FilePath, collFilePath, norms_path=None):

    delta_f_data_train, maxwell_data_train, coll_data_train, norms = \
        NormalLoadMaxwellDelfCollData(path=folder_pull_from)

    df_mean, df_std, fm_mean, fm_std, coll_mean, coll_std = norms

    # === Save arrays ONLY ===
    pickle.dump(delta_f_data_train, open(delta_f_FilePath, "wb"), protocol=4)
    pickle.dump(maxwell_data_train, open(maxwell_FilePath, "wb"), protocol=4)
    pickle.dump(coll_data_train, open(collFilePath, "wb"), protocol=4)

    # === Save norms separately (optional) ===
    if norms_path is not None:
        pickle.dump(
            {
                "df_mean": df_mean, "df_std": df_std,
                "fm_mean": fm_mean, "fm_std": fm_std,
                "coll_mean": coll_mean, "coll_std": coll_std
            },
            open(norms_path, "wb"),
            protocol=4
        )




################################################################################################################################################
################################################################################################################################################
################################################################################################################################################




def LoadMaxwellDelfCollData(path):
    import my_readwrite
    import re
    from my_distributions import maxwellian

    names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)

    # Load nodes once
    filename = '/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
    match = re.search(r'sw(\d+)MuUU', filename)
    if match:
        MM = int(match.group(1))
    else:
        raise ValueError("Could not determine MM.")
    Mtrim = 0
    # First file
    solarry, collarry, solsize = my_readwrite.my_read_sol_coll_trim(names[0], MM, Mtrim) # shape also (1, 41^3)
    moments = np.zeros((1, 5))
    moments[0, 0] = np.sum(solarry * nodes_gwts)
    moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
    moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
    moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
    moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                   (nodes_v - moments[0, 2]) ** 2 +
                                                   (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
    fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
    delta_f = solarry - fm

    # Initialize training data arrays
    delta_f_data_train = delta_f
    maxwellian_data_train = fm
    coll_data_train = collarry

    num_samples = len(names)
    for i in range(1, num_samples):
        if i % 10 == 0:
            print("Processing " + str(i) + " of " + str(num_samples))

        solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(names[i], MM, Mtrim) # shape also (1, 41^3) unaltered
        assert solsize1 == solsize

        moments = np.zeros((1, 5))
        moments[0, 0] = np.sum(solarry * nodes_gwts)
        moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
        moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
        moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
        moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                       (nodes_v - moments[0, 2]) ** 2 +
                                                       (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
        fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
        delta_f = solarry - fm

        delta_f_data_train = np.concatenate((delta_f_data_train, delta_f), axis=0)
        maxwellian_data_train = np.concatenate((maxwellian_data_train, fm), axis=0)
        coll_data_train = np.concatenate((coll_data_train, collarry), axis=0)

    return delta_f_data_train, maxwellian_data_train, coll_data_train

def CreateMaxwellDelfCollData(folder_pull_from, delta_f_FilePath, maxwell_FilePath, collFilePath):
    delta_f_data_train, maxwell_data_train, coll_data_train = LoadMaxwellDelfCollData(path=folder_pull_from)    
    with open(delta_f_FilePath, "wb") as delta_f_train_file:
        pickle.dump(delta_f_data_train, delta_f_train_file, protocol=4)

    with open(maxwell_FilePath, "wb") as maxwell_train_file:
        pickle.dump(maxwell_data_train, maxwell_train_file, protocol=4)
    
    with open(collFilePath, "wb") as coll_train_file:
        pickle.dump(coll_data_train, coll_train_file, protocol=4)



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


def LoadDelfCollData(path):
    import my_readwrite
    import re
    from my_distributions import maxwellian

    names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)

    # Load nodes once
    filename = '/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
    match = re.search(r'sw(\d+)MuUU', filename)
    if match:
        MM = int(match.group(1))
    else:
        raise ValueError("Could not determine MM.")
    Mtrim = 0
    # First file
    solarry, collarry, solsize = my_readwrite.my_read_sol_coll_trim(names[0], MM, Mtrim) # shape also (1, 41^3)
    moments = np.zeros((1, 5))
    moments[0, 0] = np.sum(solarry * nodes_gwts)
    moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
    moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
    moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
    moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                   (nodes_v - moments[0, 2]) ** 2 +
                                                   (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
    fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
    delta_f = solarry - fm

    # Initialize training data arrays
    delta_f_data_train = delta_f
    coll_data_train = collarry

    num_samples = len(names)
    for i in range(1, num_samples):
        if i % 10 == 0:
            print("Processing " + str(i) + " of " + str(num_samples))

        solarry, collarry, solsize1 = my_readwrite.my_read_sol_coll_trim(names[i], MM, Mtrim) # shape also (1, 41^3) unaltered
        assert solsize1 == solsize

        moments = np.zeros((1, 5))
        moments[0, 0] = np.sum(solarry * nodes_gwts)
        moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
        moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
        moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
        moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                       (nodes_v - moments[0, 2]) ** 2 +
                                                       (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
        fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
        delta_f = solarry - fm

        delta_f_data_train = np.concatenate((delta_f_data_train, delta_f), axis=0)
        coll_data_train = np.concatenate((coll_data_train, collarry), axis=0)

    return delta_f_data_train, coll_data_train

def CreateDelfCollData(folder_pull_from, delta_f_FilePath, collFilePath):
    delta_f_data_train, coll_data_train = LoadDelfCollData(path=folder_pull_from)    
    with open(delta_f_FilePath, "wb") as delta_f_train_file:
        pickle.dump(delta_f_data_train, delta_f_train_file, protocol=4)
    
    with open(collFilePath, "wb") as coll_train_file:
        pickle.dump(coll_data_train, coll_train_file, protocol=4)
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


def LoadDelfData(path):
    import my_readwrite
    import re
    from my_distributions import maxwellian

    names = my_readwrite.my_get_soltn_file_names_time(path, cutoff_time)

    # Load nodes once
    filename = '/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/CollTrnDta180_1su1sv1sw41MuUU41MvVU41MwWU_nodes.dat'
    nodes_u, nodes_v, nodes_w, nodes_gwts = my_readwrite.read_nodes(filename)
    match = re.search(r'sw(\d+)MuUU', filename)
    if match:
        MM = int(match.group(1))
    else:
        raise ValueError("Could not determine MM.")
    Mtrim = 0
    # First file
    solarry, solsize = my_readwrite.my_read_solution_trim(names[0], MM, Mtrim) # shape also (1, 41^3)
    moments = np.zeros((1, 5))
    moments[0, 0] = np.sum(solarry * nodes_gwts)
    moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
    moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
    moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
    moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                   (nodes_v - moments[0, 2]) ** 2 +
                                                   (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
    fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
    delta_f = solarry - fm

    # Initialize training data arrays
    delta_f_data_train = delta_f

    num_samples = len(names)
    for i in range(1, num_samples):
        if i % 10 == 0:
            print("Processing " + str(i) + " of " + str(num_samples))

        solarry, solsize1 = my_readwrite.my_read_solution_trim(names[i], MM, Mtrim) # shape also (1, 41^3) unaltered
        assert solsize1 == solsize

        moments = np.zeros((1, 5))
        moments[0, 0] = np.sum(solarry * nodes_gwts)
        moments[0, 1] = np.sum(solarry * nodes_u * nodes_gwts) / moments[0, 0]
        moments[0, 2] = np.sum(solarry * nodes_v * nodes_gwts) / moments[0, 0]
        moments[0, 3] = np.sum(solarry * nodes_w * nodes_gwts) / moments[0, 0]
        moments[0, 4] = np.sum(solarry * nodes_gwts * ((nodes_u - moments[0, 1]) ** 2 +
                                                       (nodes_v - moments[0, 2]) ** 2 +
                                                       (nodes_w - moments[0, 3]) ** 2)) / moments[0, 0] / 3.0 * 2.0
        fm = maxwellian(moments, nodes_u, nodes_v, nodes_w).reshape(1, -1)
        delta_f = solarry - fm

        delta_f_data_train = np.concatenate((delta_f_data_train, delta_f), axis=0)

    return delta_f_data_train

def CreateDelfData(folder_pull_from, delta_f_FilePath):
    delta_f_data_train = LoadDelfData(path=folder_pull_from)    
    with open(delta_f_FilePath, "wb") as delta_f_train_file:
        pickle.dump(delta_f_data_train, delta_f_train_file, protocol=4)


import argparse
parser = argparse.ArgumentParser(description="Data Procurement")
parser.add_argument("--MM", type=int, default=41)
parser.add_argument("--Mtrim", type=int, default=0)
parser.add_argument("--cutoff_time", type=float, default=1.0)
parser.add_argument("--folder_pull_from", type=str, default="/Volumes/Devaney SSD/AFIT:ARCH/Nguyen_1/Data/111 New Training Data")
parser.add_argument("--delta_f_data_file", type=str, default=None)
parser.add_argument("--maxwell_data_file", type=str, default=None)
parser.add_argument("--coll_data_file", type=str, default=None)
parser.add_argument("--norm_data_file", type=str, default=None)

args = parser.parse_args()

MM = args.MM
Mtrim = args.Mtrim
cutoff_time = args.cutoff_time
folder = args.folder_pull_from

# Now build the default filenames if user didn't provide them
if args.delta_f_data_file is None:
    delta_f_data_file = f"Data/08xs_delta_f_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pkl"
else:
    delta_f_data_file = args.delta_f_data_file

if args.maxwell_data_file is None:
    maxwell_data_file = f"Data/08xs_maxwellian_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pkl"
else:
    maxwell_data_file = args.maxwell_data_file

if args.coll_data_file is None:
    coll_data_file = f"Data/08xs_coll_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pkl"
else:
    coll_data_file = args.coll_data_file

if args.norm_data_file is None:
    norm_data_file = f"Data/08xs_norm_data_MM_{MM}_MT_{Mtrim}_CT_{cutoff_time}.pkl"
else:
    norm_data_file = args.norm_data_file

##NormalCreateMaxwellDelfCollData(folder, delta_f_data_file, maxwell_data_file, coll_data_file, norm_data_file)
CreateMaxwellDelfCollData(folder, delta_f_data_file, maxwell_data_file, coll_data_file)
##CreateDelfCollData(folder, delta_f_data_file, coll_data_file)
##CreateDelfData(folder, delta_f_data_file)
