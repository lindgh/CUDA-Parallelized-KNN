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
	
	#for each validation review, get cosine distance from each training tf-idf vector
	for i in range(len(x_val)): 	
		testVector = x_val[i:i+1] #the review to test cosine distance
		test_d = cuda.to_device(testVector)
		distances_d = cuda.device_array(x_train.shape[0],dtype=np.float32)

		#print("does this even work")

		
