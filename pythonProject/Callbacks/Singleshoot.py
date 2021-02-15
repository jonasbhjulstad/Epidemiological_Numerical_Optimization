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


class Singleshoot_CB(Callback):
  def __init__(self, name, nx, ng, nu,iter_step, opts={}):
    Callback.__init__(self)

    self.nx = nx
    self.nu = nu
    self.ng = ng
    self.nx_ode = 3

    self.x_sols = []
    self.f_sols = []
    self.g_sols = []
    self.lam_x_sols = []
    self.lam_g_sols = []
    self.lam_p_sols = []
    self.iter = []
    self.iter_step = iter_step
    self.I_thetas = []
    # Initialize internal objects
    self.construct(name, opts)

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
      self.iter.append(self.iter[-1] + self.iter_step)
    darg = {}
    for (i,s) in enumerate(nlpsol_out()): darg[s] = arg[i]
    x_sol = [float(elem) for elem in darg['x'].full()]

    self.x_sols.append(x_sol)
    self.f_sols.append(darg['f'].full()[0][0])
    self.g_sols.append(darg['g'].full())
    self.lam_x_sols.append(darg['lam_x'].full())
    self.lam_g_sols.append(darg['lam_g'].full())
    self.lam_p_sols.append(darg['lam_p'].full())
    return [0]

