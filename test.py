import matplotlib.pyplot as plt
import numpy as np
nz=100
nx=80

A1=np.ones([nz,nx])
A2=np.ones([nz,nx])

for ix in range(nx):
    A1[:10,ix]*=np.linspace(0,1,10)

plt.imshow(A1)
plt.colorbar()
plt.show()
