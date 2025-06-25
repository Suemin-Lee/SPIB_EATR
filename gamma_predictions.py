import numpy as np
import sys
import json

if len(sys.argv) != 2:
    sys.exit(f"USAGE: python {sys.argv[0]} [parameter file]")

# params_file is a JSON file with system (directory), altname (for acc. factor file), iter_nums (number of iterations), runs (list of run numbers), lograte (ln k0), and output_filename. Edit everything necessary to make sure all time units are consistent.
with open(sys.argv[1],'r') as params_file:
    params = json.load(params_file)
locals().update(params)

# Load the acceleration factors α from file
acc_factors = []
for i in range(iter_nums):
        with open(f'{altname}_data/accelerated_data_iter_{i}.npy','rb') as f:
            acc_factors.append(np.lib.format.read_array(f))

# Load the biased and rescaled residence times
sim_times = []
res_times = []
for i in range(iter_nums):
#     print(i)
    res_times.append([])
    sim_times.append([])
    for j, run in enumerate(runs[i]):
        data = np.loadtxt(f'{system}/SPIB{i}/run{run+1}/COLVAR_modified')
        sim_times[i].append(data[-1,0]/1e12)
        res_times[i].append(data[-1,0]*acc_factors[i][j]/1e12)

# Calculate the covariance between the rescaled residence times and the acceleration factors α
covars = np.array([np.cov(np.stack((acc_factors[i],sim_times[i]),axis=0))[0,1] for i in range(iter_nums)])

# Calculate the mean acceleration factors and mean rescaled residence times
logaccs = np.array([np.log(np.mean(accs)) for accs in acc_factors])
mean_res = np.array([np.mean(res) for res in res_times])

# Calculate final predictions
iters = list(range(iter_nums))
pred1 = -np.log(mean_res)/logaccs + 1 - lograte/logaccs
pred2 = -np.log(mean_res-covars)/logaccs + 1 - lograte/logaccs

print('γ prediction 1: ',pred1)
print('γ prediction 2: ',pred2)

# Save predictions to file
np.savetxt(output_filename, np.vstack((iters,pred1,pred2)).T )