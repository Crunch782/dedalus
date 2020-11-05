import os
import time

for i in range(0, 5):
	s = (1+i)*0.1
	for i in range(0, 5):
		T = (1+i)*0.1
		com = "python3 generate.py "+str(s)+" "+str(T)+" 50.0"
		os.system(com)
		os.system("sbatch run.sh")
		time.sleep(2.5)

