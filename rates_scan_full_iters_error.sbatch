#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=3:00:00
#SBATCH --job-name=fkbpdmso
#SBATCH --output=rates_scan_full_iters_error.out

echo "iter gamma std"
python rates_scan_full_iters_error.py SPIB0 4
python rates_scan_full_iters_error.py SPIB1 8
python rates_scan_full_iters_error.py SPIB2 8
python rates_scan_full_iters_error.py SPIB3 8
python rates_scan_full_iters_error.py SPIB4 8



