import numpy as np

b = 1000 # наблюдений
a = 10 # признаков
traning_data= np.loadtxt("traning_data.txt").reshape(a, b)
traning_data = traning_data.astype(int)
ans_data = np.loadtxt("ans_data.txt").reshape(b, 1)
ans_data = ans_data.astype(int)
print(traning_data)
print(ans_data)
