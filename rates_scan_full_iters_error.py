
import os,sys

import json
sys.path.append(os.path.abspath('..'))
import numpy as np
import random
import glob
import rate_methods as RM
from scipy import interpolate, optimize
from scipy.stats import ks_1samp, ks_2samp
from scipy.stats import gamma as gamma_func
import multiprocessing as mp
from functools import partial


if len(sys.argv) != 3:
    print('Include the directory and number of runs as arguments.')
    exit()

gammas = np.arange(0.,1.01,0.05)
log_rates = np.arange(-3.5,8.1,.1)

directory = sys.argv[1]
runs = [f"run{i+1}" for i in range(int(sys.argv[2]))]
colvar_name = 'COLVAR_modified_short'
log_name = colvar_name
plog_len = 1
beta = 0.3967641346

colvars = []
plogs = []
for run in runs:
    colvars.append(f"{directory}/{run}/{colvar_name}")
    plogs.append(f"{directory}/{run}/{log_name}")


# Load all colvar files
colvars_count = len(colvars)
colvars_maxrow_count = None

data = [] # data[i][j,k] is column k of simulation i at the time of row j.
final_times = np.zeros((colvars_count, 2)) # final_times[i,0] is simulation i's transition time while final_times[i,1] is the iMetaD rescaled time.
i = 0
for colvar in colvars:
    current_colvar = np.loadtxt(colvar)
    data.append(current_colvar)
    colvars_maxrow_count = data[-1].shape[0] if colvars_maxrow_count is None or colvars_maxrow_count < data[-1].shape[0] else colvars_maxrow_count
    final_times[i,:] = np.array([data[-1][-1][0],0]) # final_times[i,0] is simulation i's transition time while final_times[i,1] is the iMetaD rescaled time.
    i = i+1
k = 1. # Uninitialized value for the iMetaD CDF rate. This allows bootstrapping and KS ranges in the KTR and EATR parts of this function. Do not use regularization without running iMetaD CDF.

acc = RM.calc_acc(data, 0, 1, beta)
final_times[:,1] = final_times[:,0]*acc

# Count transitions
event = []
for plog in plogs:
    with open(plog,'r') as f:
        if len(f.readlines()) > plog_len:
            event.append(True)
        else:
            event.append(False)
event = np.array(event)
M = event.sum() # Number of transitions
N = len(event) # Total number of simulations

t = final_times[:,0]
cores = 4

random.seed(a=12345) 
CDF = True
MLE_guess = False

log_rate = np.log10(1/13e-9)

opt_gammas = []
for i in range(100):
    resample = random.choices(data, k=len(data))
    colvars_rows = [len(colv) for colv in resample]
    colvars_maxrow_count = np.max(colvars_rows)
    SSEs = []
    for gamma in gammas:
        v_data, ix_col = RM.inst_bias(resample, colvars_count, colvars_maxrow_count, beta, 1)
        spline = RM.EATR_calculate_avg_acc(gamma, v_data, beta, ix_col)
        ecdfx = np.sort(t[event])
        ecdfy = np.arange(1, event.sum() + 1) / len(event)
        SSEs.append(RM.EATR_leastsq_cost(10**(log_rate-12), gamma, ecdfx, ecdfy, spline))
    opt_gammas.append(gammas[np.argmin(SSEs)])


print(f'{directory[-1]} {np.mean(opt_gammas)} {np.std(opt_gammas)}')

# Want: rate, likelihood, and CDF SSE for best rate in each gamma

#np.savetxt(f'scan_results_{directory}.dat', results, header='# gamma \tlog10_rate \tlog_l \tSSE')
