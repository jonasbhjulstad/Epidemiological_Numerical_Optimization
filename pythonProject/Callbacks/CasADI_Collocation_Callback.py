#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#! Callback
#! =====================
from casadi import *
from numpy import *
import xarray as xr
from Collocation.collocation_coeffs import collocation_coeffs
from Plot_tools.Collocation_Plot import collocation_plot
class DataCallback(Callback):
  def __init__(self, name, params):
    Callback.__init__(self)

    self.nx = params['nx']
    self.nu = params['nu']
    self.ng = params['ng']
    self.d = params['d']

    _, _, _, self.tau_root = collocation_coeffs(self.d)
    self.tgrid = params['tgrid']
    self.tgrid_radau = np.concatenate([self.tau_root + tk for tk in self.tgrid])
    self.f_sols = []
    self.g_sols = []
    self.lam_x_sols = []
    self.lam_g_sols = []
    self.lam_p_sols = []
    self.x_sols = []
    self.iter = []

    self.construct(name, {})

  def get_n_in(self): return nlpsol_n_out()
  def get_n_out(self): return 1
  def get_name_in(self, i): return nlpsol_out(i)
  def get_name_out(self, i): return "ret"

  def get_sparsity_in(self, i):
    n = nlpsol_out(i)
    if n=='f':
      return Sparsity. scalar()
    elif n in ('x', 'lam_x'):
      return Sparsity.dense(self.nx)
    elif n in ('g', 'lam_g'):
      return Sparsity.dense(self.ng)
    else:
      return Sparsity(0,0)
  def eval(self, arg):
    # Create dictionary
    if self.iter == []:
      self.iter = [0]
    else:
      self.iter.append(self.iter[-1] + 1)
    darg = {}
    for (i,s) in enumerate(nlpsol_out()): darg[s] = arg[i]
    x_sol = darg['x'].full()
    x_sol = [a[0] for a in x_sol]
    self.x_sols.append(x_sol)
    self.f_sols.append(darg['f'].full()[0][0])
    self.g_sols.append(darg['g'].full())
    self.lam_x_sols.append(darg['lam_x'].full())
    self.lam_g_sols.append(darg['lam_g'].full())
    self.lam_p_sols.append(darg['lam_p'].full())
    return [0]

  def iter_sol_to_arrays(self, f_filter_x, file=None):
    Nx_filtered = f_filter_x.n_out()
    Nk = len(self.x_sols)
    x_plot_list = []
    u_plot_list = []
    thetas_list = []

    for x_sol in self.x_sols:
      x_plot, u_plot, thetas = f_filter_x(x_sol)
      x_plot_list.append(x_plot.full())
      u_plot_list.append(u_plot.full())
      thetas_list.append(thetas.full())

    x_plot_Array = xr.DataArray(np.array(x_plot_list), coords=[self.iter, ['S', 'I', 'R'], self.tgrid], dims=['iteration', 'group','time'])
    u_plot_Array = xr.DataArray(np.array(u_plot_list).squeeze(), coords=[self.iter, self.tgrid[:-1]], dims=['iteration', 'time'])
    thetaArray = xr.DataArray(thetas_list, coords=[self.iter,['S', 'I', 'R'], self.tgrid_radau[:-4]], dims=['iteration','group',  'time'])
    return x_plot_Array, u_plot_Array, thetaArray





