
import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from IPOPT.Single_Shooting import Primal_Dual_Single_Shooting
from IPOPT.Multiple_Shooting import Primal_Dual_Multiple_Shooting
from Single_Shooting import Single_Shooting_RK4
from Multiple_Shooting import Multiple_Shooting_RK4
from Collocation.Col_IPOPT import Direct_Collocation
from PMP.Multiple_Shooting_PMP import Multiple_shooting_PMP
from Plot_tools.Collocation_Plot import *
from Plot_tools.RK4_Plot import *


if __name__ == '__main__':


    # for param in ['Social_Distancing', 'Isolation', 'Vaccination']:
    #     fname = Multiple_shooting_PMP(param)
    #     Iterate_Objective_Bounds_Plot(fname)
    #     Multiple_Shooting_Trajectory_Plot(fname)
    #     Constraint_Objective_Plot(fname)
    # for method in ['IPOPT', 'SQP']:
    
    
    #     for param in reversed(['Vaccination']):

    #         print('Running ' + method + ', ' + param)
    #         fname = Direct_Collocation(param, method=method)
    #         Direct_Collocation_Trajectory_Plot(fname)


    
    # for param in ['Social_Distancing', 'Isolation', 'Vaccination']:
    #     fname = Primal_Dual_Single_Shooting(param)
    #     Trajectory_Plot(fname)
    #     Bounds_Objective_Plot(fname)
    #     Iterate_Objective_Bounds_Plot(fname)

    # for param in ['Social_Distancing', 'Isolation', 'Vaccination']:
    #         fname = Primal_Dual_Multiple_Shooting(param, traj_initial=True)
    #         Iterate_Objective_Constraint_Plot(fname)
    #         Multiple_Shooting_Trajectory_Plot(fname)
    #         Constraint_Objective_Plot(fname)

    for method in ['IPOPT', 'SQP']:
    
    
        for param in ['Social_Distancing', 'Isolation', 'Vaccination']:

            # print('Running ' + method + ', ' + param)
            # fname = Single_Shooting_RK4(param, method=method)
            # Trajectory_Plot(fname)
            # Bounds_Objective_Plot(fname)
            # Iterate_Objective_Bounds_Plot(fname)
            
            fname = Multiple_Shooting_RK4(param, method=method, traj_initial=True)
            Trajectory_Plot(fname)
            Iterate_Objective_Constraint_Plot(fname)
            Multiple_Shooting_Trajectory_Plot(fname)
            Constraint_Objective_Plot(fname)


    
    


