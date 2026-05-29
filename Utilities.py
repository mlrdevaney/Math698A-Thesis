##########################################################################################################
# This module provides various utility functions that can be used by other modules.
#
##########################################################################################################

import os
import pickle

data_dir = 'Data'

def RemoveSavedModels(dirpath):
	if (os.path.exists(dirpath)):
		files = os.listdir(dirpath)
		for file in files:
			os.remove(dirpath + '/' + file)
	else:
    		os.mkdir(dirpath)
    		
    	
    		
def LoadPickleSolData(data_file):
	f = open(data_file, "rb")
	sol_data = pickle.load(f)
	f.close()
	return sol_data

def CheckRequiredDir():		
	if (os.path.exists(data_dir) == False):
		os.mkdir(data_dir)

def LoadSettings():
    settings = {}
    with open("settings.txt") as f:
        for line in f:
            (key, val) = line.split()
            settings[key] = val
    return settings

