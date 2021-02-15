import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
if __name__ == '__main__':

    N_pop = 5.3e6
    u_min = 0.5
    u_max = 6.5
    Wu = N_pop**2/(u_max-u_min)/10
    x = np.linspace(0,N_pop, 100)
    u = np.linspace(u_min, u_max, 100)
    cost = lambda I, u: I**2 - Wu*u**2
    X, Y = np.meshgrid(x, u)
    Z = cost(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()