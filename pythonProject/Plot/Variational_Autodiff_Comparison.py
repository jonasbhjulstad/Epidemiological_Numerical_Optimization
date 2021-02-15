import numpy as np
import pickle as pck
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
import pandas as pd
from Integrators.Autodiff_SIR_Sensitivity import Autodiff_SIR_Sensitivity
from Integrators.Variational_SIR_Sensitivity import Variational_SIR_Sensitivity
def read_data(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pck.load(openfile))
            except EOFError:
                break
    return objects

def largest_sense_elem(pd_col):
    return max([matrix.max(np.matrix(k)) for k in pd_col.values])

def error_stats(df_auto, df_var, iter):
    A_auto = largest_sense_elem(df_auto['A'])
    B_auto = largest_sense_elem(df_auto['B'])
    A_var = largest_sense_elem(df_var['A'])
    B_var = largest_sense_elem(df_var['B'])

    Traj_var = np.max(np.max(df_var.loc[:,['S','I','R']].values))
    Traj_auto = np.max(np.max(df_auto.loc[:,['S','I','R']].values))

    Traj_diff = Traj_var-Traj_auto
    A_diff = A_auto - A_var
    B_diff = B_auto - B_var
    return pd.DataFrame({'A_auto': A_auto, 'B_auto': B_auto, 'Traj_auto': Traj_auto,'A_var': A_var,'B_var': B_var, 'Traj_var': Traj_var, 'Traj_diff': Traj_diff, 'A_diff': A_diff, 'B_diff': B_diff}, index=[iter])
if __name__ == '__main__':

    load = False
    data_true = pd.read_pickle('../data/True_Trajectory.pck')
    data_true = data_true.rename(columns={'A': 'A_true', 'B': 'B_true', 'S': 'S_true', 'I': 'I_true', 'R': 'R_true'})
    # N_true = data_true.shape[0]
    # t_end = data_true.index[-1]

    diff_stats = pd.DataFrame([], columns=['A_auto', 'B_auto','Traj_auto', 'A_var', 'B_var', 'Traj_var'])
    i = 1
    if load:
        data_auto = pd.read_pickle('../data/Autodiff_sense.pck')
        data_var = pd.read_pickle('../data/Variational_sense.pck')

    l = 1
    enum_list = np.logspace(0,-5,6)
    for i in enum_list:
        print("Iteration %d of %d, dt:%.6f" %(l,len(enum_list),i))
        l += 1
        data_auto = Autodiff_SIR_Sensitivity(i, stop=1)
        data_var = Variational_SIR_Sensitivity(i, stop=1)

        true_df_fit = pd.merge_asof(data_auto, data_true, left_index=True, right_index=True,direction='nearest').loc[:, data_true.columns]
        true_df_fit = true_df_fit.rename(columns={'A_true':'A', 'B_true': 'B', 'S_true': 'S', 'I_true': 'I', 'R_true': 'R'})

        diff_auto = data_auto-true_df_fit
        diff_var = data_var-true_df_fit

        diff_stats = diff_stats.append(error_stats(diff_auto, diff_var, i))


    plt.style.use('seaborn-white')


    fig, ax = plt.subplots(2, 1)



    ax[0].set(xlabel='time (s)', ylabel='Individuals',
           title='Sensitivity Comparison')
    ax[0].grid()

    ax[1].plot()
    ax[1].set(xlabel='time (s)', ylabel='Individuals')
    ax[1].grid()
    plt.show()
