import numpy as np
from implementations import *
from data_preprocessing import *
from data_io import *
from validation import *

y = np.array([
       0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

print(y.size)

res = stratify_sampling(y, 3, True)

for i in range(3):
       print(y[res[i]])
