import sys
from os.path import dirname, abspath
parent = dirname(dirname(abspath(__file__)))
sys.path.append(parent)
from Single_Shooting import Single_Shooting_RK4
from Multiple_Shooting import Multiple_Shooting_RK4
from Collocation.Direct_Collocation_IPOPT import DirectCollocationMain


if __name__ == '__main__':

    for param in ['Social_Distancing', 'Vaccination']:
        Single_Shooting_RK4(param, method='IPOPT')
        Multiple_Shooting_RK4(param, method='IPOPT')



