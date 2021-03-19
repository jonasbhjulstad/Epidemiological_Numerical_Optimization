from Single_Shooting import SingleShoot_main
from Single_Shooting_SQP import SingleShootSQP_main
from Multiple_Shooting import MultiShoot_main
from Multiple_Shooting_SQP import MultiShootSQP_main
from Collocation.Direct_Collocation_IPOPT import DirectCollocationMain


if __name__ == '__main__':

    for param in ['Social Distancing', 'Isolation', 'Vaccination']:
        # SingleShootSQP_main(param)
        # SingleShoot_main(param)
        MultiShoot_main(param, traj_initial=True)
        MultiShoot_main(param, traj_initial=False)
        MultiShootSQP_main(param, traj_initial=True)
        MultiShootSQP_main(param, traj_initial=False)
        DirectCollocationMain(param)