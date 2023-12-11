import numpy as np
import matplotlib.pyplot as plt
import scipy

# Problem 1

data = np.genfromtxt('Plutonium.csv', delimiter = ',')
t = data[0, :]
P = data[1, :]
h = t[1] - t[0]

def forward_sec(i):
    return (P[i + 1] - P[i]) / h
def backward_sec(i):
    return (P[i] - P[i - 1]) / h
def aug_forward_sec(i):
    return (-3 * P[i] + 4 * P[i + 1] - P[i + 2]) / 2 * h
def aug_backward_sec(i):
    return (3 * P[i] - 4 * P[i - 1] + P[i - 2]) / 2 * h
def aug_central_sec(i):
    return (P[i + 1] - P[i - 1]) / (h * 2)

A1 = h
A2 = forward_sec(0)
A3 = backward_sec(40)
A4 = aug_forward_sec(0)
A5 = aug_backward_sec(40)
A6 = np.zeros(len(t))
A7 = np.zeros(len(t))

A6[0] = aug_forward_sec(0)
A6[len(t) - 1] = aug_backward_sec(len(t) - 1)
for i in range (1, len(t) - 1):
    A6[i] = aug_central_sec(i)
for i in range (len(t)):
    A7[i] = (-1 / P[i]) * A6[i]
    
A8 = np.average(A7)
A9 = np.log(2) / A8

# Problem 2

def LHR(x, y, delta_x):
    return delta_x * np.sum(y[: -1])

def RHR(x, y, delta_x):
    return delta_x * np.sum(y[1 :])

def MPR(x, S, delta_x):
    mid = 0
    for k in range(len(x) - 1):
        mid = mid + delta_x * S((x[k] + x[k + 1]) / 2)
    return mid

def Trap(x, y, delta_x):
    return (LHR(x, y, delta_x) + RHR(x, y, delta_x)) / 2

def Simp(y, delta_x):
    return (delta_x / 3) * (y[0] + 4 * np.sum(y[1 : -1 : 2]) + 2 * np.sum(y[2 : -2 : 2]) + y[-1])

mu = 85
sigma = 8.3
S = lambda x: (1/ np.sqrt(2 * np.pi * sigma ** 2)) * np.e ** ((-(x - mu) ** 2) / (2 * sigma ** 2))

A11 = scipy.integrate.quad(S, 110, 130)[0]

steps = np.zeros(16)
for i in range (1, 17):
    steps[i - 1] = 2 ** -i

A12 = np.zeros(16)
A13 = np.zeros(16)
A14 = np.zeros(16)
A15 = np.zeros(16)
A16 = np.zeros(16)

for i in range (16):
    x = np.arange(110, 130 + steps[i], steps[i])
    y = S(x)
    A12[i] = LHR(x, y, steps[i])
    A13[i] = RHR(x, y, steps[i])
    A14[i] = MPR(x, S, steps[i])
    A15[i] = Trap(x, y, steps[i])
    A16[i] = Simp(y, steps[i])

print(A11)
print(A12)
print(A13)
print(A14)
print(A15)
print(A16)