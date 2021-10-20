#! /usr/bin/python3
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import sin, cos, pi

def mass(dens, vol):
    return dens*vol

def to_deg(rad):
    return (rad*180)/pi

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

    func = -((R*omega**2*sin(omega*t)-g)*p + R*omega**2*cos(omega*t))/denom
    return [pp, func]



def solver(tend):
    # initial cond
    # phi and phi'
    phi0 = [pi/180, 0]

    # time end
    #tend = 15

    sol = solve_ivp(H, [0, tend], phi0, method='Radau', t_eval=np.linspace(0, tend, 50*tend))

    return sol

def main(start, end, step, tend, title):
    # 2D
    legends = []
    prev_diff = 1e10
    curr_omega = 0
    global omega
    max_dev = 0
    for o in np.arange(start, end, step):
        o = round(o, 3)
        omega = o
        sol = solver(tend)
        print("d")
        y = sol.y[0,:]
        x = sol.t

        y_min, y_max = to_deg(np.min(y)), to_deg(np.max(y))

        max_dev = max(abs(y_min), abs(y_max))
        plt.plot(x, y, linewidth=1)
        legends.append(f'$ \Omega $: {omega}; max:{round(y_max, 2)}$ \degree $; min:{round(y_min, 2)}$ \degree $; Max Dev:{round(max_dev,3)}$ \degree $')
    plt.title(title)
    plt.xlabel("t", fontsize=20)
    plt.ylabel(r"$ \phi(t)$", fontsize=20)
    plt.legend(legends)
    plt.show()

main(100, 101, 1, 10, "Part 1")
main(90.82, 90.86, .01, 26, "Part 2")
