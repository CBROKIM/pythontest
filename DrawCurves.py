import numpy as np
import matplotlib.pyplot as plt
import sh
import math
i=0;
Vph=np.empty([1000,30])
khs=np.linspace(0.1,np.pi,1000)
khs=np.random.rand(1000)*np.pi
angles=np.linspace(0,np.pi/4,30)
coefs=sh.Coef2D(khs)
for kh in khs:
    #coef=sh.Coef2D(np.ones(1)*kh)
#    print(coef)
    j=0;
    coef=coefs[i,:]
    coef=coef.T
    for angle in angles:
        A=math.cos(kh*math.cos(angle))+math.cos(kh*math.sin(angle))
        B=math.cos(kh*math.cos(angle))*math.cos(kh*math.sin(angle))
        Vph[i,j]=math.sqrt(-(coef[3]*(A-2)+coef[4]*(2*B-A))\
                /(coef[0]+coef[1]*A+coef[2]*B))/kh
        j+=1
    i+=1
Vph=Vph.T

plt.figure()
print(coefs.shape)
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.scatter(khs,coefs[:,i])
#plt.show()
plt.figure()
for i in range(len(angles)):
    plt.scatter(khs,Vph[i])#,'g')
plt.show()

