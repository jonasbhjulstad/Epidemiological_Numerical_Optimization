import numpy as np
from casadi import *
def collocation_coeffs(d):
    # Use a breakpoint in the code line below to debug your script.

    tau_root = [0] + collocation_points(d, "legendre")
    B = np.zeros((d+1,1))
    C = np.zeros((d+1, d+1))
    D = np.zeros((d+1,1))

    for j in range(d+1):

        p = np.poly1d([1])
        for r in range(d+1):
            if r!=j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        D[j] = p(1.0)

        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        pint = np.polyint(p)
        B[j] = pint(1.0)

    # print("Collocation coefficients: ", C, D, B)
    return B, C, D, tau_root

def legendre_polynomials(d):
    tau_root = [0] + collocation_points(d, "legendre")
    p_list = []
    for j in range(d+1):

        p = np.poly1d([1])
        for r in range(d+1):
            if r!=j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])
        p_list.append(p)
    return p_list, tau_root