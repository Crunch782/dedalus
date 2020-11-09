import os
import time
import numpy as np

arr = []

for i in range(0,10):
    arr.append(int(np.ceil((23./36.)*(i+1))))


k = 0

for i in range(k, k+1):
	s = 1+i
	for j in range(0, 10):
		T = 1+j
		com = "python3 generate.py "+str(s)+" "+str(T)+" 50.0 "+str(arr[j])
		os.system(com)
		os.system("sbatch run.sh")
		time.sleep(1)

