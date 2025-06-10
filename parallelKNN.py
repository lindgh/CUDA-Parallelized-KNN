import numpy as np
import time 
from numba import cuda
from collections import Counter

def numbaParallelKNN(x_train, y_train, x_val, y_val, k = 3):
	#START TIME
	print("\nGPU PARALLELIZED KNN KERNEL")

	CORRECT = 0;
	RANGED_CORRECT = 0; 

	train_d = cuda.to_device(x_train) 
	
	
