import numpy as np

Arr = np.array([[1, 2], [3, 4], [5, 6]])

print(Arr)
fout = open("work_file.txt", 'w')
for row in Arr:
    np.savetxt(fout, row)
fout.close()

Arr_in = np.loadtxt("work_file.txt").reshape(3,2)
print(Arr_in)


