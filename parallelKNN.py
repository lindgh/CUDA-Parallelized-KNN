import numpy as np
import time
from numba import cuda
from collections import Counter

@cuda.jit
def cosineSim(train_vectors, test_vector, distances):
        idx = cuda.grid(1)
                
        if idx < train_vectors.shape[0]:
                dot = 0.0
                normal_a = 0.0
                normal_b = 0.0

                for j in range(train_vectors.shape[1]):
                        a = train_vectors[idx, j]
                        b = test_vector[0, j]
                        dot += a * b
                        normal_a += a * a
                        normal_b += b * b

                denominator = (normal_a ** 0.5) * (normal_b ** 0.5)
                distances[idx] = 1 - (dot / denominator) if denominator != 0 else 1.0

def numbaParallelKNN(x_train, y_train, x_val, y_val, k):
        #START TIME
        print("\n=== Running GPU KNN ===")
        start = time.time()
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
        total = len(x_val)
        end = time.time()
        print(f"Strict Accuracy: {CORRECT}/{total} = {CORRECT/total:.4f}")
        print(f"Ranged (+/-1) Accuracy: {RANGED_CORRECT}/{total} = {RANGED_CORRECT/total:.4f}")
        print(f"Total GPU time: {end - start:.2f} seconds")
        print("\n=== Running GPU KNN ===")

def predict_knn_gpu(x_train, y_train, x_test_row, k):
        BLOCK_SIZE = 32
        dim_grid = (x_train.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE

        train_d = cuda.to_device(x_train.astype(np.float32))
        test_d = cuda.to_device(x_test_row[np.newaxis, :].astype(np.float32))
        distances_d = cuda.device_array(x_train.shape[0], dtype=np.float32)

        cosineSim[dim_grid , BLOCK_SIZE](train_d, test_d, distances_d)

        cosineDistances = distances_d.copy_to_host()
        closestReviews = np.argsort(cosineDistances)[:k]
        closestLabels = y_train[closestReviews]
        predictedRating = Counter(closestLabels).most_common(1)[0][0]
        return predictedRating
