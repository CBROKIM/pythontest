import ModelingOOP as MOP
import numpy as np
import matplotlib.pyplot as plt

filename = 'truevel.370x120.25m.marmousi'
with open(filename,'rb') as f:
    vel=np.fromfile(f,dtype=np.float32)
    vel=np.reshape(vel,[370,120])
    vel=vel.T
a=MOP.Modeling(100,vel,[10,10,10,10],0.5)
a.MakeWavefield([10,a.nx/2],40.0)
a.PrintWavefield()
'''
plt.figure()
plt.imshow(a.velpad, aspect='auto')
plt.colorbar()

plt.figure()
for i in range(5):
    plt.subplot(6,1,i+1)
    plt.plot(a.khs, a.coefs[:,i])
#plt.subplot(6,1,6)
#plt.plot(range(len(a.khs)),a.khs)
plt.show()
'''
