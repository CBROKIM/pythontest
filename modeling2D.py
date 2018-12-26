import sh
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

np.set_printoptions(precision=3, suppress=True)
vel=np.floor(np.random.rand(40,40)*5+15)
nPMLs=[10,10,10,10]
velPML=sh.Velpad(vel,nPMLs,0.2)
[Nz,Nx]=np.shape(velPML)
S=sh.Smake(velPML,20.0,0.1,nPMLs)
plt.imshow(velPML,aspect='auto')
plt.colorbar()
plt.show()
spz=30
spx=20
f=np.zeros(np.shape(velPML),dtype=complex)
f[spz+nPMLs[0],spx+nPMLs[2]]=1j
from scipy.sparse.linalg import splu
lu=splu(S)
u=lu.solve(f.ravel())
u=u.reshape(Nz,Nx)
#u=u[nPMLs[0]+1:Nz-nPMLs[3],nPMLs[1]+1:Nx-nPMLs[2]]
plt.imshow(u.real, aspect='auto')
plt.colorbar()
plt.show()
