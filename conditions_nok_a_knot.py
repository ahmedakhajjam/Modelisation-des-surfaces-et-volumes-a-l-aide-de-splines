# -----------AKHAJJAM AHMED------------------------
#  Modeling using spline functions
#  Application on the Modeling of surfaces and volumes
#  Master "Modeling and scientific computing"
#---------------2020/2021-------------------------
#import scipy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
# number of patches in both direction
m=6 # x
n=5 # y
# fit knots
xx = np.linspace(0, 10, m)
yy = np.linspace(0, 10, n)
x=np.reshape(np.tile(xx,(n,1)).T,(n*m))+(np.random.random((n*m))*2.0-1.0)*0.2
y=np.tile(yy,m)+(np.random.random((n*m))*2.0-1.0)*0.2
z=(np.random.random((n*m))*2.0-1.0)
Px=np.concatenate((x,np.tile([0],(m+2)*(n+2)-(m*n))))
Py=np.concatenate((y,np.tile([0],(m+2)*(n+2)-(m*n))))
Pz=np.concatenate((z,np.tile([0],(m+2)*(n+2)-(m*n))))
# passing matrix with free end conditions
phi=np.zeros(((n+2)*(m+2),(m+2)*(n+2)))
# interpolation equations
for j in range(m):
    for i in range(n):
        phi[i+(j)*n,i+(j)*(n+2)]=1
        phi[i+(j)*n,i+(j)*(n+2)+1]=4
        phi[i+(j)*n,i+(j)*(n+2)+2]=1
        phi[i+(j)*n,i+(j+1)*(n+2)]=4
        phi[i+(j)*n,i+(j+1)*(n+2)+1]=16
        phi[i+(j)*n,i+(j+1)*(n+2)+2]=4
        phi[i+(j)*n,i+(j+2)*(n+2)]=1
        phi[i+(j)*n,i+(j+2)*(n+2)+1]=4
        phi[i+(j)*n,i+(j+2)*(n+2)+2]=1
# x- y- corner
phi[n*m,0]=1
phi[n*m,1]=-1
phi[n*m,n+2]=-1
phi[n*m,n+2+1]=1
# x- y+ corner
phi[n*m+1,n]=-1
phi[n*m+1,n+1]=1
phi[n*m+1,n+n+2]=1
phi[n*m+1,n+n+2+1]=-1
# x+ y- corner
phi[n*m+2,(n+2)*(m+2)-1-n-3-n]=-1
phi[n*m+2,(n+2)*(m+2)-1-n-2-n]=1
phi[n*m+2,(n+2)*(m+2)-1-1-n]=1
phi[n*m+2,(n+2)*(m+2)-1-n]=-1
# x+ y+ corner
phi[n*m+3,(n+2)*(m+2)-1-n-2-1]=1
phi[n*m+3,(n+2)*(m+2)-1-n-2]=-1
phi[n*m+3,(n+2)*(m+2)-1-1]=-1
phi[n*m+3,(n+2)*(m+2)-1]=1
# not-a-knot at y- border
for i in range(n):
    phi[n*m+4+i,i]=-1
    phi[n*m+4+i,i+1]=-4
    phi[n*m+4+i,i+2]=-1
    phi[n*m+4+i,i+1*(n+2)]=4
    phi[n*m+4+i,i+1+1*(n+2)]=16
    phi[n*m+4+i,i+2+1*(n+2)]=4
    phi[n*m+4+i,i+2*(n+2)]=-6
    phi[n*m+4+i,i+1+2*(n+2)]=-24
    phi[n*m+4+i,i+2+2*(n+2)]=-6
    phi[n*m+4+i,i+3*(n+2)]=4
    phi[n*m+4+i,i+1+3*(n+2)]=16
    phi[n*m+4+i,i+2+3*(n+2)]=4
    phi[n*m+4+i,i+4*(n+2)]=-1
    phi[n*m+4+i,i+1+4*(n+2)]=-4
    phi[n*m+4+i,i+2+4*(n+2)]=-1
# not-a-knot at y+ border
for i in range(n):
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)]=-1
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+1]=-4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+2]=-1
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+(n+2)]=4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+(n+2)+1]=16
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+(n+2)+2]=4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+2*(n+2)]=-6
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+2*(n+2)+1]=-24
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+2*(n+2)+2]=-6
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+3*(n+2)]=4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+3*(n+2)+1]=16
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+3*(n+2)+2]=4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+4*(n+2)]=-1
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+4*(n+2)+1]=-4
    phi[n*m+4+n+i,i+(m+2)*(n+2)-5*(n+2)+4*(n+2)+2]=-1
# not-a-knot at x- border
for i in range(m):
    phi[n*m+4+2*n+i,i*(n+2)]=-1
    phi[n*m+4+2*n+i,i*(n+2)+1]=+4
    phi[n*m+4+2*n+i,i*(n+2)+2]=-6
    phi[n*m+4+2*n+i,i*(n+2)+3]=+4
    phi[n*m+4+2*n+i,i*(n+2)+4]=-1
    phi[n*m+4+2*n+i,i*(n+2)+1*(n+2)]=-4
    phi[n*m+4+2*n+i,i*(n+2)+1*(n+2)+1]=+16
    phi[n*m+4+2*n+i,i*(n+2)+1*(n+2)+2]=-24
    phi[n*m+4+2*n+i,i*(n+2)+1*(n+2)+3]=+16
    phi[n*m+4+2*n+i,i*(n+2)+1*(n+2)+4]=-4
    phi[n*m+4+2*n+i,i*(n+2)+2*(n+2)]=-1
    phi[n*m+4+2*n+i,i*(n+2)+2*(n+2)+1]=+4
    phi[n*m+4+2*n+i,i*(n+2)+2*(n+2)+2]=-6
    phi[n*m+4+2*n+i,i*(n+2)+2*(n+2)+3]=+4
    phi[n*m+4+2*n+i,i*(n+2)+2*(n+2)+4]=-1
# not-a-knot at x+ border
for i in range(m):
    phi[n*m+4+m+2*n+i,i*(n+2)+n+2-5]=-1
    phi[n*m+4+m+2*n+i,i*(n+2)+1+n+2-5]=+4
    phi[n*m+4+m+2*n+i,i*(n+2)+2+n+2-5]=-6
    phi[n*m+4+m+2*n+i,i*(n+2)+3+n+2-5]=+4
    phi[n*m+4+m+2*n+i,i*(n+2)+4+n+2-5]=-1
    phi[n*m+4+m+2*n+i,i*(n+2)+1*(n+2)+n+2-5]=-4
    phi[n*m+4+m+2*n+i,i*(n+2)+1*(n+2)+1+n+2-5]=+16
    phi[n*m+4+m+2*n+i,i*(n+2)+1*(n+2)+2+n+2-5]=-24
    phi[n*m+4+m+2*n+i,i*(n+2)+1*(n+2)+3+n+2-5]=+16
    phi[n*m+4+m+2*n+i,i*(n+2)+1*(n+2)+4+n+2-5]=-4
    phi[n*m+4+m+2*n+i,i*(n+2)+2*(n+2)+n+2-5]=-1
    phi[n*m+4+m+2*n+i,i*(n+2)+2*(n+2)+1+n+2-5]=+4
    phi[n*m+4+m+2*n+i,i*(n+2)+2*(n+2)+2+n+2-5]=-6
    phi[n*m+4+m+2*n+i,i*(n+2)+2*(n+2)+3+n+2-5]=+4
    phi[n*m+4+m+2*n+i,i*(n+2)+2*(n+2)+4+n+2-5]=-1
# control points
phi_inv=np.linalg.inv(phi)
Qx=36*phi_inv.dot(Px)
Qy=36*phi_inv.dot(Py)
Qz=36*phi_inv.dot(Pz)
# figure plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,color='black')
t=7 # patch discretization
U=np.linspace(0,1,num=t)
V=np.linspace(0,1,num=t)
u,v = np.meshgrid(U, V)
for pv in range(0,n-1):
    for pu in range (0,m-1):
        V1=(1-v)**3
        V2=3*v**3-6*v**2+4
        V3=-3*v**3+3*v**2+3*v+1
        V4=v**3
        U1=(1-u)**3
        U2=3*u**3-6*u**2+4
        U3=-3*u**3+3*u**2+3*u+1
        U4=u**3
        param_x=((V1*(Qx[pv+pu*(n+2)]*U1+Qx[pv+n+2+pu*(n+2)]*U2+Qx[pv+2*(n+2)+pu*(n+2)]*U3+Qx[pv+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V2*(Qx[pv+1+pu*(n+2)]*U1+Qx[pv+1+n+2+pu*(n+2)]*U2+Qx[pv+1+2*(n+2)+pu*(n+2)]*U3+Qx[pv+1+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V3*(Qx[pv+2+pu*(n+2)]*U1+Qx[pv+2+n+2+pu*(n+2)]*U2+Qx[pv+2+2*(n+2)+pu*(n+2)]*U3+Qx[pv+2+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V4*(Qx[pv+3+pu*(n+2)]*U1+Qx[pv+3+n+2+pu*(n+2)]*U2+Qx[pv+3+2*(n+2)+pu*(n+2)]*U3+Qx[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36;
        param_y=((V1*(Qy[pv+pu*(n+2)]*U1+Qy[pv+n+2+pu*(n+2)]*U2+Qy[pv+2*(n+2)+pu*(n+2)]*U3+Qy[pv+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V2*(Qy[pv+1+pu*(n+2)]*U1+Qy[pv+1+n+2+pu*(n+2)]*U2+Qy[pv+1+2*(n+2)+pu*(n+2)]*U3+Qy[pv+1+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V3*(Qy[pv+2+pu*(n+2)]*U1+Qy[pv+2+n+2+pu*(n+2)]*U2+Qy[pv+2+2*(n+2)+pu*(n+2)]*U3+Qy[pv+2+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V4*(Qy[pv+3+pu*(n+2)]*U1+Qy[pv+3+n+2+pu*(n+2)]*U2+Qy[pv+3+2*(n+2)+pu*(n+2)]*U3+Qy[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36;
        param_z=((V1*(Qz[pv+pu*(n+2)]*U1+Qz[pv+n+2+pu*(n+2)]*U2+Qz[pv+2*(n+2)+pu*(n+2)]*U3+Qz[pv+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V2*(Qz[pv+1+pu*(n+2)]*U1+Qz[pv+1+n+2+pu*(n+2)]*U2+Qz[pv+1+2*(n+2)+pu*(n+2)]*U3+Qz[pv+1+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V3*(Qz[pv+2+pu*(n+2)]*U1+Qz[pv+2+n+2+pu*(n+2)]*U2+Qz[pv+2+2*(n+2)+pu*(n+2)]*U3+Qz[pv+2+3*(n+2)+pu*(n+2)]*U4)) \
        		+(V4*(Qz[pv+3+pu*(n+2)]*U1+Qz[pv+3+n+2+pu*(n+2)]*U2+Qz[pv+3+2*(n+2)+pu*(n+2)]*U3+Qz[pv+3+3*(n+2)+pu*(n+2)]*U4)))/36;
        ax.plot_surface(param_x,param_y,param_z,color='green',alpha=1)
#plt.plot(Qx,Qy,Qz,".-",color='green',label='points controle',alpha=0.9)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(-5,5)
plt.savefig('bicubic_parametric_surface_not-a-knot.png')
plt.show()
