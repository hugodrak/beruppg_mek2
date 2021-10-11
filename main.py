#! /usr/bin/python3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import sin, cos, pi



def mass(dens, vol):
    return dens*vol

global m, L, r2, R, g, omega, I
L = 1 #m
R = 0.05  #m
r1 = 0.04 #m
r2 = 0.05 #m
# tröghetsmom för hela = stor kula - liten kula
dens = 7800 # kg/m^3
g = 9.81
omega = 100#100 # rad/s, 15.915 hz
# m = m1 + m2
m1 = mass(dens, (4/3)*pi*r1**2)
m2 = mass(dens, (4/3)*pi*r2**2)
m = m1 + m2
I1 = (2/5)*(m1*r1**2)
I = (2/5)*(m2*r2**2)
I2 = I-I1

denom = (I2 + m * ((L + r2)**2)) / (m * (L+r2))


def H(t, Y):
    p = Y[0] # phi(t)
    pp = Y[1] # phi'(t) phi prim

    # Y[0] == y1, Y[1] = y2
    # func = ((w0**2)*y1 - fp(y1)*fpp(y1)*(y2**2) - fp(y1)*g) / (fp(y1)**2 + 1)

    #func = (m*(L+r2)*(g*sin(p)+omega**2*R*cos(p-omega*t))) / (I-m*(L+R)**2)
    func = -((R*omega**2*sin(omega*t)-g)*p + R*omega**2*cos(omega*t))/denom
    return [pp, func]



def solver():
    # initial cond
    # phi and phi'
    phi0 = [pi/180, 0]

    # time end
    tend = 5

    sol = solve_ivp(H, [0, tend], phi0, method='Radau', t_eval=np.linspace(0, tend, 2000))

    # # calculate pos in fixed coordinate system
    # xs = []
    # ys = []
    # #zs = f(sol.y[0])  # f(xi)x
    # for i, t in enumerate(sol.t):
    #     phi = sol.y[0][i]
    #     x = phi * np.cos(omega * t)
    #     y = phi * np.sin(omega * t)
    #     xs.append(x)
    #     ys.append(y)
    return sol

def main(start, end, step):
    # 2D
    legends = []
    prev_diff = 1e10
    curr_omega = 0
    global omega
    max_dev = 0
    for o in range(start, end+step, step):
        #global omega
        omega = o
        sol = solver()
        #x = sol.t
        y = sol.y[0,:]
        for phi in y:
            if abs(phi) > max_dev:
                max_dev = abs(phi)
                curr_omega = omega
        #y_min, y_max = np.min(y), np.max(y)
        #diff = abs(y_max-y_min)
        # plt.plot(x, y, linewidth=1)
        # legends.append(f'$ \Omega$: {omega}; d: {diff}')

    #global omega
    omega = curr_omega
    sol = solver()
    x = sol.t
    y = sol.y[0,:]

    plt.plot(x, y, linewidth=1)
    deg = (max_dev*180)/pi
    legends.append(f'$ \Omega$: {omega}; d: {round(deg, 4)}')

    plt.xlabel("t", fontsize=20)
    plt.ylabel(r"$ \phi(t)$", fontsize=20)
    plt.legend(legends)
    #plt.title(f'$ \Omega$: {omega}', fontsize=18, fontweight='bold')
    plt.show()


def test():
    # L = 1 #m
    # R = 0.05  #m
    # r1 = 0.04 #m
    # r2 = 0.05 #m
    # # tröghetsmom för hela = stor kula - liten kula
    # dens = 7800 # kg/m^3
    #
    # omega = 100 # rad/s, 15.915 hz

    # m = m1 + m2
    # m1 = mass(dens, (4/3)*math.pi*r1**2)
    # m2 = mass(dens, (4/3)*math.pi*r2**2)
    # m = m1 + m2

    # phi(0) = 1deg, phi'(0)=0
    # 1: calculate max dev of pendulum from vertical, ie:
    # ie: max(phi(t))

    # 2: min(omega)






    pass

main(100, 100, 1)
