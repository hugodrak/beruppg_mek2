import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

L = 1
R = 0.05
r1 = 0.04
r2 = 0.05
omega = 100
g = 9.82
#Masses
dens = 7800
v1 = 4/3 * np.pi * r1**3
v2 = 4/3 * np.pi * r2**3 - v1
m1 = v1 * dens
m2 = v2 * dens
#print(f"vol 1: {v1}, vol2: {v2}, mass1: {m1}, mass2: {m2}")
m = m1 + m2
#I2 = 2/3 * m2 * r2**2
#Moments of inertia
I1 = 2/5 * m1 * r1**2
Itot = 2/5 * m * r2**2
I2 = Itot - I1
#print(f"I2: {I2},,, Steiner: {I2 + m2 * (L+r2)**2}")
compact = (I2 + m * ((L + r2)**2)) / (m * (L+r2))
# print(I2)
# I1 = 2/5 * m2 * r2**2 - 2/5 * m1 * r1**2
# print(I1)
# print(I2-I1)

def h (t, y):
    y_prim = [0,0]

    y_prim[0] = y[1]
    y_prim[1] = -((R * omega**2 * np.sin(omega * t) - g) * y[0] + R * omega**2 * np.cos(omega* t)) / compact
    return y_prim


#initial
y_0 = [np.pi/180, 0]
#y2_0 = [0, 0]
#y3_0 = [5, -1]

#stop time
tend = 5

def omegatest(o, n, x):
    global omega
    omega = o
    for i in range(n):
        sol = solve_ivp(h, [0, tend], y_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
        plt.plot(sol.t,sol.y[0,:], label = str(np.round(omega,5)))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('phi(t)')
        plt.title(r"$ \omega $")
        #plt.show()
        omega += x
    plt.show()

omegatest(100, 1, 1)
#omegatest(90.9, 5, -.001)
#testat värde då det kukar ur cirka 90.89

#solve ODE
sol = solve_ivp(h, [0, tend], y_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
#sol2 = solve_ivp(h, [0, tend], y2_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
#sol3 = solve_ivp(h, [0, tend], y3_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))

xs = []
ys = []

# for i, t in enumerate(sol.t):
#     phi = sol.y[0][i] #nuvarande phi-värde
#     x = (L + r2) * np.sin(phi)
#     y = (L+r2) * np.sin(phi)
#     xs.append(x)
#     ys.append(y)

# #positionplot
# plt.plot(xs, ys)
# plt.show()



#plot
plt.plot(sol.t,sol.y[0,:])
#plt.plot(sol2.t,sol2.y[0,:], label = str(np.round(y2_0[0],4)))
plt.legend()
plt.xlabel('time')
plt.ylabel('phi(t)')
plt.show()
