import numpy as np
import sympy as sp
import control as ctrl
from ipydex import IPS

# x1 = x
# x2 = x_dot
# x3 = phi
# x4 = phi_dot

# parameters of cartpole model
g = 9.8
mc = 1.0 # mass cart
mp = 0.1 # mass pole
total_mass = mc + mp
l = 0.5  # actually half the pole's length #! wtf does that mean? 

A = np.array([[0,1,0,0],
              [0,0,0,0],
              [0,0,0,1],
              [0,0,g/l,0]])

B = np.array([[0,1,0,1/l]]).T
a = 1.2
b = 3.4
poles = np.r_[-1.5+a*1j, -1.5-a*1j, -1.3 + b*1j, -1.3 - b*1j]
# F = ctrl.place(A, B, poles)
F = ctrl.place(A, B, np.linspace(-6, -8, 4))
print(F)

poles = np.linalg.eigvals(A-B@F)
print(poles)

