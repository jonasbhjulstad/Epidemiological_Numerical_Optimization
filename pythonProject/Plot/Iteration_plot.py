import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import pandas as pd
from Plot_tools.Contour_Objective import ObjectivePlot

def objective(x,u):
    Wu = 100
    return x**2 + Wu/(u**2)
if __name__ == '__main__':
    df = pd.read_pickle('../data/Direct_Collocation_Iteration_data.pck')
