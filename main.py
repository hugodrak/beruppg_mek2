import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

def mass(dens, vol):
    return dens*vol


def phi_biz(m, L, R2, g, RA, I, phi, omega, t):
    pass


def main():
    L = 1 #m
    R = 0.05  #m
    r1 = 0.04 #m
    r2 = 0.05 #m
    # tröghetsmom för hela = stor kula - liten kula
    dens = 7800 # kg/m^3

    omega = 100 # rad/s, 15.915 hz

    # m = m1 + m2
    m1 = mass(dens, (4/3)*math.pi*r1**2)
    m2 = mass(dens, (4/3)*math.pi*r2**2)
    m = m1 + m2

    # phi(0) = 1deg, phi'(0)=0
    # 1: calculate max dev of pendulum from vertical, ie:
    # ie: max(phi(t))

    # 2: min(omega)
