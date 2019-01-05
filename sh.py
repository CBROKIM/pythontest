import math
import scipy.linalg as la
import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix
def Smake(vel, omega, dx,nPMLs):
    [nz,nx]=np.shape(vel)
#    k2=np.power(omega,2)/np.power(vel,2)
    k2=np.power(omega/vel,2)

    Full    = range(nz*nx)
    Hal1    = nz*np.repeat(range(nx),  nz-1) + np.tile(range(nz-1),nx)
    Hal2    = nz*np.repeat(range(nx-1),nz)   + np.tile(range(nz),  nx-1)
    Both    = nz*np.repeat(range(nx-1),nz-1) + np.tile(range(nz-1),nx-1)

#   set PML
    A1=np.ones([nz,nx],dtype=complex)
    A2=np.ones([nz,nx],dtype=complex)
    c0=5.0
    etaz0=1j*c0*math.log(1000)/omega/(nPMLs[0]*dx)\
            *np.power(np.linspace(0,1,nPMLs[0]),2)
    etaz1=1j*c0*math.log(1000)/omega/(nPMLs[1]*dx)\
            *np.power(np.linspace(0,1,nPMLs[1]),2)
    etaz2=1j*c0*math.log(1000)/omega/(nPMLs[2]*dx)\
            *np.power(np.linspace(0,1,nPMLs[2]),2)
    etaz3=1j*c0*math.log(1000)/omega/(nPMLs[3]*dx)\
            *np.power(np.linspace(0,1,nPMLs[3]),2)
    for ix in range(nx):
        A1[:nPMLs[0],ix]        +=vel[:nPMLs[0],ix]	*etaz0.T
        A1[nz-nPMLs[1]:nz,ix]   +=vel[nz-nPMLs[1]:nz,ix]*etaz1
    for iz in range(nz):
        A2[iz,:nPMLs[2]]        +=vel[iz,:nPMLs[2]]	*etaz2.T
        A2[iz,nx-nPMLs[3]:nx]   +=vel[iz,nx-nPMLs[3]:nx]*etaz3
    A12=(A2/A1+A1/A2)/2
    k2A12=k2*A1*A2

    veluniq, inverse_index = np.unique(vel, return_inverse=True)
    coefs=Coef2D(omega/veluniq*dx)
    
    #
    print(coefs)
    #

    coef=coefs[inverse_index]
    coef[0:2,:]*=dx*dx
    a=coef[:,0].reshape(nz,nx)
    b=coef[:,1].reshape(nz,nx)
    c=coef[:,2].reshape(nz,nx)
    d=coef[:,3].reshape(nz,nx)
    e=coef[:,4].reshape(nz,nx)
    #
    row5 = Full
    col5 = Full
    val5 = 8*d[:]*A12 - 4*a[:]*k2A12
    #
    row2 = Hal1+1
    col2 = Hal1
    val2 = -2*d[1:,:]*A2[1:,:]/A1[1:,:]\
           +2*e[1:,:]*A1[1:,:]/A2[1:,:]\
           -2*b[1:,:]*k2A12[1:,:]
    #
    row8 = Hal1
    col8 = Hal1+1
    val8 = -2*d[:nz-1,:]*A2[:nz-1,:]/A1[:nz-1,:]\
           +2*e[:nz-1,:]*A1[:nz-1,:]/A2[:nz-1,:]\
           -2*b[:nz-1,:]*k2A12[:nz-1,:]
    #
    row4 = Hal2+nz
    col4 = Hal2
    val4 = -2*d[:,1:]*A1[:,1:]/A2[:,1:]\
           +2*e[:,1:]*A2[:,1:]/A1[:,1:]\
           -2*b[:,1:]*k2A12[:,1:]
    #
    row6 = Hal2
    col6 = Hal2+nz
    val6 = -2*d[:,:nx-1]*A1[:,:nx-1]/A2[:,:nx-1]\
           +2*e[:,:nx-1]*A2[:,:nx-1]/A1[:,:nx-1]\
           -2*b[:,:nx-1]*k2A12[:,:nx-1]
    #
    row1 = Both+nz+1
    col1 = Both
    val1 = -2*e[1:,1:]*A12[1:,1:]\
             -c[1:,1:]*k2A12[1:,1:]
    #
    row3 = Both+1
    col3 = Both+nz
    val3 = -2*e[1:,:nx-1]*A12[1:,:nx-1]\
             -c[1:,:nx-1]*k2A12[1:,:nx-1]
    #
    row7 = Both+nz
    col7 = Both+1
    val7 = -2*e[:nz-1,1:]*A12[:nz-1,1:]\
             -c[:nz-1,1:]*k2A12[:nz-1,1:]
    #
    row9 = Both
    col9 = Both+nz+1
    val9 = -2*e[:nz-1,:nx-1]*A12[:nz-1,:nx-1]\
             -c[:nz-1,:nx-1]*k2A12[:nz-1,:nx-1]
#    val=np.concatenate((val1,val2,val3,val4,val5,val6,val7,val8,val9),axis=None)
    val=np.concatenate((val1.T,val2.T,val3.T,val4.T,val5.T,val6.T,val7.T,val8.T,val9.T),axis=None)
    col=np.concatenate((col1,col2,col3,col4,col5,col6,col7,col8,col9))#,axis=None)
    row=np.concatenate((row1,row2,row3,row4,row5,row6,row7,row8,row9))#,axis=None)
    Mat=csr_matrix((val,(row,col)),shape=(nx*nz,nx*nz))
    return Mat

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

def Velpad(vel,nPMLs,factor):
    from scipy.ndimage import gaussian_filter as gf
    velPML= np.pad(vel,[[nPMLs[0],nPMLs[1]],[nPMLs[2],nPMLs[3]]],'edge')
    velPML= np.pad(vel,[nPMLs[:2],nPMLs[2:]],'edge')
    Nz,Nx = np.shape(velPML)
    for iz in range(nPMLs[0]):
        velPML[iz,:]=gf(velPML[iz,:],(nPMLs[0]-iz)*factor)
    for iz in range(nPMLs[1]):
        velPML[Nz-iz-1,:]=gf(velPML[Nz-iz-1,:],(nPMLs[1]-iz)*factor)
    for ix in range(nPMLs[2]):
        velPML[:,ix]=gf(velPML[:,ix],(nPMLs[2]-ix)*factor)
    for ix in range(nPMLs[3]):
        velPML[:,Nx-ix-1]=gf(velPML[:,Nx-ix-1],(nPMLs[3]-ix)*factor)
    return velPML

