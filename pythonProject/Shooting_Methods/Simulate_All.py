import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Single_Shooting import Single_Shooting_RK4
from Multiple_Shooting import Multiple_Shooting_RK4
from Collocation.Direct_Collocation_IPOPT import Direct_Collocation
from Plot_tools.RK4_Plot import *


if __name__ == '__main__':
    for method in ['IPOPT', 'SQP']:
        for param in ['Social_Distancing', 'Vaccination']:
            print('Running ' + method + ', ' + param)
            fname = Single_Shooting_RK4(param, method=method)
            Trajectory_Plot(fname)
            Bounds_Objective_Plot(fname)
            
            for b in [True, False]:
                fname = Multiple_Shooting_RK4(param, method=method, traj_initial=b)
                Trajectory_Plot(fname)
                Iterate_Objective_Constraint_Plot(fname)
                Multiple_Shooting_Trajectory_Plot(fname)
                Constraint_Objective_Plot(fname)

    
    


