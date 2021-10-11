import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

w0 = 3
g = 9.81
a = 3
b = 4

def f(x):
    return a * np.sin(x)**2 + b * x**2

def f_prim(x):
    return a * np.sin(2*x) + 2 * b * x

def f_bis(x):
    return 2*a * np.cos(2*x) + b


def h (t, y):
    y_prim = [0,0]

    y_prim[0] = y[1]
    y_prim[1] = ((w0**2)*y[0] - f_prim(y[0])*f_bis(y[0])*(y[1]**2) - f_prim(y[0])*g) / (1 + f_prim(y[0])**2)
    return y_prim

#initial
y_0 = [0.5, 0]
tend = 20

sol = solve_ivp(h, [0, tend], y_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))

def w0test(w, n, x):
    global w0
    w0 = w
    for i in range(n):
        sol = solve_ivp(h, [0, tend], y_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
        plt.plot(sol.t,sol.y[0,:], label = str(np.round(w0,5)))
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('xi(t)')
        plt.title("w0")
        #plt.show()
        w0 += x
    plt.show()

#w0test(11.314,8,0.00001)

#11.32 (11.315?) kritiskt värde

#y2_0 = [5, 3]
#y3_0 = [5, -1]

xs = []
ys = []
zs = f(sol.y[0])  # f(xi)
for i, t in enumerate(sol.t):
    xi = sol.y[0][i]
    x = xi * np.cos(w0 * t)
    y = xi * np.sin(w0 * t)
    xs.append(x)
    ys.append(y)


fig = plt.figure()
three_D = fig.add_subplot(111, projection='3d')
three_D.plot(xs, ys, zs)

three_D.set_xlabel('$X$', fontsize=20)
three_D.set_ylabel('$Y$', fontsize=20)
three_D.zaxis.set_rotate_label(False)
three_D.set_zlabel('$Z$', fontsize=20)
three_D.set_title(f'w0: {w0}', fontsize=18, fontweight='bold')

#solve ODE
#sol = solve_ivp(f, [0, tend], y_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
#sol2 = solve_ivp(h, [0, tend], y2_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
#sol3 = solve_ivp(h, [0, tend], y3_0, method = "Radau", t_eval=np.linspace(0, tend, 10000))
"""
#plot
plt.plot(sol.t,sol.y[0,:])
#plt.plot(sol2.t,sol2.y[0,:])
#plt.plot(sol3.t,sol3.y[0,:])
plt.xlabel('time')
plt.ylabel('xi(t)')"""
plt.show()
