import numpy as np
import time 
from numba import cuda
from collections import Counter

@cuda.jit
def cosineSim(train_vectors, test_vector, distances):
	idx = cuda.grid(1)
	distances[idx] = 0

def numbaParallelKNN(x_train, y_train, x_val, y_val, k):
	#START TIME
	print("\nGPU PARALLELIZED KNN KERNEL")

	CORRECT = 0;
	RANGED_CORRECT = 0; 

	train_d = cuda.to_device(x_train.astype(np.float32)) 
	
	#for each validation review, get cosine distance from each training tf-idf vector
	for i in range(len(x_val)): 	
		testVector = x_val[i:i+1] #the review to test cosine distance
		test_d = cuda.to_device(testVector.astype(np.float32))
		distances_d = cuda.device_array(x_train.shape[0],dtype=np.float32)

		#print("does this even work")
		BLOCK_SIZE = 32
		dim_grid = (x_train.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
		
		#invoke numba kernel
		cosineSim[dim_grid , BLOCK_SIZE](train_d, test_d, distances_d) 

		cosineDistances = distances_d.copy_to_host()
		closestReviews = np.argsort(cosineDistances)[:k]
		closestLabels = y_train[closestReviews]
		
		#MAKE PREDICTION BY MOST COMMON CLASS OF K NEIGHBORS
		predictedRating = Counter(closestLabels).most_common(1)[0][0]	
		
		#KEEP TRACK OF HOW MANY CORRECT
		if predictedRating == y_val[i]:
			CORRECT+=1
		#KEEP TRACK OF HOW MANY CORRECT (with tolerance) 
		if predictedRating == y_val[i] + 1 or predictedRating == y_val[i] - 1 or predictedRating == y_val[i]: 
			RANGED_CORRECT+=1
		 
