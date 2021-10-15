#AKHAJJAM AHMED 
#MÈMOIRE DE FIN D’ÉTUDES
#Pour obtenir le Diplôme de
#Master “Modélisation et calcul scientifique”
#Modélisation à l’aide de fonctions splines
#Application sur la Modélisation des surfaces et volumes
import matplotlib.pyplot as plt
import numpy as np
# number of patches in both direction
m=7# x
n=8# y
# fit knots
xx = np.linspace(0, 10, m)
yy = np.linspace(0, 10, n)
x=np.reshape(np.tile(xx,(n,1)).T,(n*m))+(np.random.random((n*m))*2.0-1.0)*0.1
y=np.tile(yy,m)+(np.random.random((n*m))*2.0-1.0)*0.1
z=(np.random.random((n*m))*2.0-1.0)
Px=np.concatenate((x,np.tile([0],(m+2)*(n+2)-(m*n))))
Py=np.concatenate((y,np.tile([0],(m+2)*(n+2)-(m*n))))
pu=np.concatenate((z,np.tile([0],(m+2)*(n+2)-(m*n))))
# passing matrix
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
# y- border
for i in range(m):
    phi[n*m+i,(i+1)*(n+2)]=1
    phi[n*m+i,(i+1)*(n+2)+1]=-2
    phi[n*m+i,(i+1)*(n+2)+2]=1
# y+ border
for i in range(m):
    phi[n*m+m+i,(i+1)*(n+2)+(n+2-3)]=1
    phi[n*m+m+i,(i+1)*(n+2)+(n+2-2)]=-2
    phi[n*m+m+i,(i+1)*(n+2)+(n+2-1)]=1
# x+ border
for i in range(n):
    phi[n*m+2*m+i,1+i]=1
    phi[n*m+2*m+i,1+i+n+2]=-2
    phi[n*m+2*m+i,1+i+2*(n+2)]=1
# x- border
for i in range(n):
    phi[n*m+2*m+n+i,-1-(1+i)]=1
    phi[n*m+2*m+n+i,-1-(1+i+(n+2))]=-2
    phi[n*m+2*m+n+i,-1-(1+i+2*(n+2))]=1
# x- y- corner
phi[n*m+2*m+2*n,0]=1
phi[n*m+2*m+2*n,n+2+1]=-2
phi[n*m+2*m+2*n,2*(n+2)+2]=1
# x- y+ corner
phi[n*m+2*m+2*n+1,(n+2)-1]=1
phi[n*m+2*m+2*n+1,(n+2)+((n+2)-1)-1]=-2
phi[n*m+2*m+2*n+1,2*(n+2)+((n+2)-2)-1]=1
# x+ y- corner
phi[n*m+2*m+2*n+2,(m+2)*(n+2)-(n+2)]=1
phi[n*m+2*m+2*n+2,(m+2)*(n+2)-2*(n+2)+1]=-2
phi[n*m+2*m+2*n+2,(m+2)*(n+2)-3*(n+2)+2]=1
# x+ y+ corner
phi[n*m+2*m+2*n+3,(m+2)*(n+2)-2*(n+2)-3]=1
phi[n*m+2*m+2*n+3,(m+2)*(n+2)-(n+2)-2]=-2
phi[n*m+2*m+2*n+3,(m+2)*(n+2)-1]=1
# control points
phi_inv=np.linalg.inv(phi)
Qx=36*phi_inv.dot(Px)
Qy=36*phi_inv.dot(Py)
Qz=36*phi_inv.dot(pu)
# figure plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,'.',color='black')
t=11 # patch discretization
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
        ax.plot_surface(param_x,param_y,param_z,color='blue',alpha=0.9)
#plt.plot(Qx,Qy,Qz,".-",color='blue',label='control points',markersize=8.0)
#plt.plot(x,y,z,'o',color='green')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(10,-10)
#plt.legend(loc='upper left', ncol=2)
plt.savefig('bicubic_parametric_surface_free_end.png')
plt.show()
