import numpy as np

traning_data= np.loadtxt("traning_data.txt").reshape(10, 10)
ans_data = np.loadtxt("ans_data.txt")
print(traning_data)
print(ans_data)
