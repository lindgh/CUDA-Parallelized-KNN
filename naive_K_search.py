#implement best k search

import numpy as np
from collections import Counter
from naive_knn import predict_knn

def naive_bestKsearch(x_train, y_train, y_val, krange): 
	BEST_K = -1
	BEST_ACCURACY = -1
	BEST_RANGED_K = -1
	BEST_RANGED_ACCURACY = -1
	
	print("\n--- NAIVE FIND BEST K ---")
	#loop through all k values and find the best 
	for k in krange:
		correct = 0
		ranged_correct = 0
		
		pred = predict_knn(X_train, Y_train, X_val[i], k)
       		actual = Y_val[i]
       		if pred == actual:
           		correct += 1
       		if abs(pred - actual) <= 1:
           		ranged_correct += 1
