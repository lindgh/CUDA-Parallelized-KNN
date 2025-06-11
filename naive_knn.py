#add naive implementation here

import numpy as np
import time
from collections import Counter

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    A_norm = np.linalg.norm(a)
    B_norm = np.linalg.norm(b)
    if A_norm == 0 or B_norm == 0:
        return 0
    return dot / (A_norm * B_norm)


def predict_knn(X_train, Y_train, X_test, k):
    distances = []
    for i in range(len(X_train)):
        similarity = cosine_similarity(X_train[i], X_test)
        distances.append((similarity, Y_train[i]))
    
    #now we sort so that the first one is the highest similarity 
    distances.sort(reverse = True)
    top_k = [label for _, label in distances[:k]] 
    most_common = Counter(top_k).most_common(1)[0][0]
    return most_common


def run_naive_knn(X_train, Y_train, X_val, Y_val, k):
    print("\n=== Running Naive CPU Version of KNN ===\n")
    start = time.time()

    correct = 0
    ranged_correct = 0

    for i in range(len(X_val)):
        pred = predict_knn(X_train, Y_train, X_val[i], k)
        actual = Y_val[i]
        if pred == actual:
            correct += 1
        if abs(pred - actual) <= 1:
            ranged_correct += 1

    total = len(X_val) 
    end = time.time()

    print(f"Strict Accuracy: {correct}/{total} = {correct/total:.4f}")
    print(f"Ranged (+/-1) Accuracy: {ranged_correct}/{total} = {ranged_correct/total:.4f}")
    print(f"Total time: {end - start:.2f} seconds") 
    print("\n=== Ending Naive CPU Version of KNN ===\n")
