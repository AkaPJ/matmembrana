import numpy as np
import time
from matplotlib import pyplot


def creasis(N, mu):
    diagonal = np.ones(N * N) * (1 + 4 * mu)
    M = np.diag(diagonal)
    vdi = -np.ones(N * N - 1) * mu;
    for i in range(0, N * N - 1, N):
        vdi[i] = 0;

    M = M + np.diag(vdi, 1) + np.diag(vdi, -1)
    vdd = -np.ones(N * N - N) * mu;
    M = M + np.diag(vdd, -N) + np.diag(vdd, N);
    return (M)


N = 20
M = creasis(N, -1)
pyplot.spy(M)
fil, col = M.shape
b = np.random.rand(N * N, 1)
start_time = time.perf_counter()
for vuelta in range(50):
    x0 = np.zeros([N * N, 1])
    for i in range(fil):
        for j in range(col):
            x0[i] = x0[i] + M[i, j] * b[j]

end_time = time.perf_counter()
print("Program finished in {} seconds".format(end_time - start_time))
error = (M @ b - x0)
print('error =', np.linalg.norm(error))