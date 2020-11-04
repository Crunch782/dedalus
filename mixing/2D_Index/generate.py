"""
=========================================================================

This program generates the batch script for running the DAL computations
Enter parameters below and the required script is created

=========================================================================
"""

file = open("run.sh", "w")

# Algorithm Parameters
T       = '2.0'        #Target Optimization Time        (eg 2.0)
Re      = '50.0'       #Reynolds number                 (eg 50.0)
K       = '1.5'        #Step Factor (if too small)      (eg 2.0)
r       = '0.5'        #Step factor (if too big)        (eg 0.5)
e0init  = '0.1'        #Initial Step                    (eg 1.0)
LS      = '1'          #Line Search/Not                 (eg 1)
LSI     = '1'          #Line Search Interpolation/Not   (eg 1)
proj    = '1'          #Projection onto surface/Not     (eg 1)
method  = 'rot'        #rot/Lag update on u0            (eg 'rot')
dmethod = 'conj'       #conj/grad direction update      (eg 'conj')
powit   = '0'          #Power iteration                 (eg 0)
N       = '200'         #Max number of loops             (eg 50)
start   = 'rand'       #rand/cont IC                    (eg 'rand')
s       = '0.5'        #s index value                   (eg 0.5)
if start == 'cont':
    resMin  = '0.005'    #Minimum res achieved in last    (determined from a previous output, eg 0.157...)

arr = [T, Re, K, r, e0init, LS, LSI, proj, method, dmethod, powit, N, start, s]
if start == 'cont':
    arr.append(resMin)

# Batch Script Parameters
sTre = 'Re='+Re+'_s='+s+'_T='+T
N = 1
n = 1
cpus = 2
cutoff = 8

# Batch Script Parameters as Strings
projectname = str(sTre)
bigN = str(N)
smalln = str(n)
ncpus = str(cpus)
cutTime = str(8)

file.write("#!/bin/bash\n")
file.write("#! Account name for group, use SL2 for paying queue:\n")
file.write("#SBATCH -J "+projectname+"\n")
file.write("#SBATCH -A CAULFIELD-SL3-CPU\n")
file.write("#SBATCH -p skylake-himem\n\n")

file.write("#SBATCH -N "+bigN+"\n")
file.write("#SBATCH -n "+smalln+"\n")
file.write("#SBATCH --cpus-per-task="+ncpus+"\n")
file.write("#SBATCH --time="+cutTime+":00:00\n\n")

file.write("#SBATCH --error=./Output/"+projectname+".err\n")
file.write("#SBATCH --output=./Output/"+projectname+".out\n")
file.write("#SBATCH --mail-type=BEGIN\n")
file.write("#SBATCH --mail-type=END\n\n\n")

file.write("#! Modify the environment seen by the application. For this example we need the default modules.\n")
file.write(". /etc/profile.d/modules.sh                # This line enables the module command\n")
file.write("module purge                               # Removes all modules still loaded\n")
file.write("module load rhel7/default-peta4         # REQUIRED - loads the basic environment\n\n")
file.write("unset CONDA_SHLVL\n\n")
file.write("module load miniconda/3\n")
file.write("source /usr/local/software/master/miniconda/3/etc/profile.d/conda.sh\n")
file.write("conda activate dedalus\n\n\n")

file.write("#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.\n")
file.write("#! In this example, each job will be passed its index, so each output file will contain a different value\n\n")
file.write("#!echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope\n\n")

str1 = 'python3 Optimization.py '
sp = '\n'
str2 = ''

for i in range (0, len(arr)):
	str2 = str2+arr[i]
	if i < len(arr) - 1:
		str2 = str2 + ' '

command = str1+str2

file.write(command)

file.close()
