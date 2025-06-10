#implement best k search


import numpy as np
from collections import Counter
from naive_knn import predict_knn

def naive_bestKsearch(x_train, y_train, x_val, y_val, krange):
    BEST_K = -1
    BEST_ACCURACY = -1
    BEST_RANGED_K = -1
    BEST_RANGED_ACCURACY = -1

    print("\n--- NAIVE FIND BEST K ---")
    #loop through all k ranges    
    for k in range(1,krange + 1):
        correct = 0
        ranged_correct = 0
	
	#calculate KNN
        for i in range(len(x_val)):
            pred = predict_knn(x_train, y_train, x_val[i], k)
            actual = y_val[i]

            if pred == actual:
                correct += 1
            if abs(pred - actual) <= 1:
                ranged_correct += 1

        accuracy = correct / len(y_val)
        ranged_accuracy = ranged_correct / len(y_val)

        print(f"K={k:3d} | Accuracy={accuracy:.4f} | Ranged Accuracy ={ranged_accuracy:.4f}")

        if accuracy > BEST_ACCURACY:
            BEST_ACCURACY = accuracy
            BEST_K = k
        if ranged_accuracy > BEST_RANGED_ACCURACY:
            BEST_RANGED_ACCURACY = ranged_accuracy
            BEST_RANGED_K = k

    print(f"\nBEST STRICT K: {BEST_K}, ACCURACY: {BEST_ACCURACY:.4f}")
    print(f"BEST RANGED K: {BEST_RANGED_K}, RANGED ACCURACY: {BEST_RANGED_ACCURACY:.4f}")
    print("\n--- END NAIVE FIND BEST K ---")

