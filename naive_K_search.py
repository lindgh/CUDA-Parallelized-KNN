#implement best k search


import numpy as np
import time

from collections import Counter

def naive_bestKsearch(x_train, y_train, x_val, y_val, krange, predict_device):
    #for making graph:
    accuracies = []
    ranged_accuracies = []
    #########################
    BEST_K = -1
    BEST_ACCURACY = -1
    BEST_RANGED_K = -1
    BEST_RANGED_ACCURACY = -1

    # print("\n--- NAIVE FIND BEST K ---")

    start_overall = time.time()
    #loop through all k ranges    
    for k in krange:
        correct = 0
        ranged_correct = 0
        start_individual = time.time()	
	#calculate KNN
        for i in range(len(x_val)):
            pred = predict_device(x_train, y_train, x_val[i], k)
            actual = y_val[i]

            if pred == actual:
                correct += 1
            if abs(pred - actual) <= 1:
                ranged_correct += 1

        end_individual = time.time()
        accuracy = correct / len(y_val)
        ranged_accuracy = ranged_correct / len(y_val)
	
        #for graphs
        accuracies.append(accuracy)
        ranged_accuracies.append(ranged_accuracy)

        print(f"K={k:3d} | Accuracy={accuracy:.4f} | Ranged Accuracy ={ranged_accuracy:.4f} | Time: {end_individual - start_individual:.2f} seconds")

        if accuracy > BEST_ACCURACY:
            BEST_ACCURACY = accuracy
            BEST_K = k
        if ranged_accuracy > BEST_RANGED_ACCURACY:
            BEST_RANGED_ACCURACY = ranged_accuracy
            BEST_RANGED_K = k

    end_overall = time.time()
    print(f"\nBEST STRICT K: {BEST_K}, ACCURACY: {BEST_ACCURACY:.4f}")
    print(f"BEST RANGED K: {BEST_RANGED_K}, RANGED ACCURACY: {BEST_RANGED_ACCURACY:.4f}")
    print(f"Total time: {end_overall - start_overall:.2f} seconds") 
    #print("\n--- END NAIVE FIND BEST K ---")
    return accuracies, ranged_accuracies

