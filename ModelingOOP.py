import scipy.linalg as la
import scipy as sp
import numpy as np
import math
from scipy.sparse import csr_matrix

class Modeling:
    def __init__(self, dx, vel,nPMLs=[10,10,10,10],padFactor=0.2):
        self.nz,self.nx = vel.shape
        self.dx=dx
        self.vel=vel
        self.nPMLs=nPMLs
        self.factor=padFactor

    def MakeWavefield(self, spxz, freq):
        spxz = [int(x) for x in spxz]
        velpad = self.VelPadding()
        Nz,Nx = velpad.shape
        S = Smake(velpad, freq, self.dx, self.nPMLs)
        f = np.zeros(Nz*Nx) 
        f[spxz[0]*Nx+spxz[1]]=1.0 #Nx*sz+sx
        from scipy.sparse.linalg import splu
        lu=splu(S)
        self.wavefield = lu.solve(f)
        self.wavefield = self.wavefield.reshape(Nz,Nx)

    def PrintWavefield(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(np.real(self.wavefield))
        plt.show()

    def VelPadding(self):
        from scipy.ndimage import gaussian_filter as gf
        nPMLs=self.nPMLs
        factor=self.factor
        vel=self.vel
        velpad= np.pad(vel,[nPMLs[:2],nPMLs[2:]],'edge')
        Nz,Nx = np.shape(velpad)
        for iz in range(nPMLs[0]):
            velpad[iz,:]=gf(velpad[iz,:],(nPMLs[0]-iz)*factor)
        for iz in range(nPMLs[1]):
            velpad[Nz-iz-1,:]=gf(velpad[Nz-iz-1,:],(nPMLs[1]-iz)*factor)
        for ix in range(nPMLs[2]):
            velpad[:,ix]=gf(velpad[:,ix],(nPMLs[2]-ix)*factor)
        for ix in range(nPMLs[3]):
            velpad[:,Nx-ix-1]=gf(velpad[:,Nx-ix-1],(nPMLs[3]-ix)*factor)
        velpad=np.floor(velpad)
        return velpad

def Smake(vel, freq, dx, nPMLs=[10,10,10,10]):
    [nz,nx] = vel.shape
    
    veluniq, inverse_idx = np.unique(vel, return_inverse=True)
    khs = freq / veluniq*dx
    coefs = Coef2D(khs)
    coefs = coefs[inverse_idx]
    coefs[:,0:2]*=dx*dx
    k2=np.power(freq/vel,2)
    
    A1=np.ones([nz,nx],dtype=complex)
    A2=np.ones([nz,nx],dtype=complex)
    c0=2.0
    etaz0=1j*c0*math.log(1000)/freq/(nPMLs[0]*dx)\
            *np.power(np.linspace(0,1,nPMLs[0]),2)
    etaz1=1j*c0*math.log(1000)/freq/(nPMLs[1]*dx)\
            *np.power(np.linspace(0,1,nPMLs[1]),2)
    etaz2=1j*c0*math.log(1000)/freq/(nPMLs[2]*dx)\
            *np.power(np.linspace(0,1,nPMLs[2]),2)
    etaz3=1j*c0*math.log(1000)/freq/(nPMLs[3]*dx)\
            *np.power(np.linspace(0,1,nPMLs[3]),2)
    for ix in range(nx):
        A1[:nPMLs[0],ix]        +=vel[:nPMLs[0],ix]	*etaz0.T
        A1[nz-nPMLs[1]:nz,ix]   +=vel[nz-nPMLs[1]:nz,ix]*etaz1
    for iz in range(nz):
        A2[iz,:nPMLs[2]]        +=vel[iz,:nPMLs[2]]	*etaz2.T
        A2[iz,nx-nPMLs[3]:nx]   +=vel[iz,nx-nPMLs[3]:nx]*etaz3
    A12=(A2/A1+A1/A2)/2
    k2A12=k2*A1*A2
    
    A12=A12.flatten()
    k2A12=k2A12.flatten()
    A1=A1.flatten()
    A2=A2.flatten()
    Full    = range(nx*nz)
    Hal1    = nx*np.repeat(range(nz-1),nx)   + np.tile(range(nx),  nz-1)    #for 2 and 8
    Hal2    = nx*np.repeat(range(nz),  nx-1) + np.tile(range(nx-1),nz)      #for 4 and 6
    Both    = nx*np.repeat(range(nz-1),nx-1) + np.tile(range(nx-1),nz-1)
    row=[]
    col=[]
    val=[]

#center
    row.extend(Full)
    col.extend(Full)
    val.extend((8*coefs[:,3]*A12.flatten()-4*coefs[:,0]*k2A12.flatten()).flatten())
#edges
    #2 and 8    +-nx
    valtemp=-2*coefs[:,3]*A2/A1\
            +2*coefs[:,4]*A1/A2\
            -2*coefs[:,1]*k2A12
    row.extend(Hal1+nx)
    col.extend(Hal1)
    val.extend(valtemp[Hal1+nx])
    row.extend(Hal1)
    col.extend(Hal1+nx)
    val.extend(valtemp[Hal1])
    #4 and 6    +-1
    valtemp=-2*coefs[:,3]*A1/A2\
            +2*coefs[:,4]*A2/A1\
            -2*coefs[:,1]*k2A12
    row.extend(Hal2+1)
    col.extend(Hal2)
    val.extend(valtemp[Hal2+1])
    row.extend(Hal2)
    col.extend(Hal2+1)
    val.extend(valtemp[Hal2])
#corners
    valtemp=-2*coefs[:,4]*A12\
            -1*coefs[:,2]*k2A12
    row.extend(Both+nx+1)
    col.extend(Both)
    val.extend(valtemp[Both+nx+1])  #1
    row.extend(Both+1)
    col.extend(Both+nx)
    val.extend(valtemp[Both+1])     #3
    row.extend(Both+nx)
    col.extend(Both+1)
    val.extend(valtemp[Both+nx])    #7
    row.extend(Both)
    col.extend(Both+nx+1)
    val.extend(valtemp[Both])       #9
    return csr_matrix((val,(row,col)),shape=(nx*nz,nx*nz))

def Coef2D(khs):
    x=[1, 0.9808, 0.9239, 0.8315, 0.7071]
    y=[0, 0.1951, 0.3827, 0.5556, 0.7071]
    nkh=np.size(khs)
    coefs = sp.empty([nkh,5])
    for ikh in range(nkh):
        kh=khs[ikh]
        Mat=sp.empty((5,3))
        RHS=sp.empty(5)
        for i in range(5):
            A=math.cos(kh*x[i])+math.cos(kh*y[i])        
            B=math.cos(kh*x[i])*math.cos(kh*y[i])       
            Mat[i]=([A-2,B-1,(A-B-1)*2])
            RHS[i]=A/2-B-kh*kh/4
        MatMat=Mat.T @ Mat
        MatRHS=Mat.T @ RHS
        sol=la.solve(MatMat+sp.eye(3)*la.norm(Mat)*0.000000000001,MatRHS)
        sol[[0,1]]=sol[[0,1]]/kh/kh
        sol=np.append([0.25-(2*sol[0]+sol[1])],sol)
        sol=np.append(sol,0.5-sol[3])
        coefs[ikh,:] = sol
    return coefs


