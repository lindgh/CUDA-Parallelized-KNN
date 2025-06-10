import numpy as np
import time 
from numba import cuda
from collections import Counter

def numbaParallelKNN(x_train, y_train, x_val, y_val, k = 3):
	#START TIME

	
