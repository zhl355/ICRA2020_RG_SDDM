#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LTI solver for RGS with output peak computation.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import expm
from my_utils import get_dir_mat
from opt_solver import find_eta_max_opt_scaling
from traj_est import find_eta_max_analytic

class LTI_Error(Exception):
    """# User Defined Exceptions for LTI solver
    """
    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "LTI_Error exception: {0}".format(self.msg)
        else:
            return "LTI_Error exception"


class RGS_LTISolver:
    """
    Class for robot-governor system simulation state update
    """

    def __init__(self,
                 design_paras, Pd, PV0,
                 dt=0.05):
        """ Init the solver according discretiziation time
        'dt', and got upadate matrices and corresponding update costants
        for LTI system state update equation.
        """
        self.dt = dt
        self.PV = PV0
        self.Pd = Pd
        self.c1 = Pd[0][0]
        self.c2 = Pd[1][1]
        self.design_paras = design_paras
        self.eta_max_log_ANL = []
        self.eta_max_log_OPT = []

        self.inaccurate_indices = []
        self.solver_err_indices = []
        self.numerical_err_indices = []
        self.eta_bound_err_indices = []


        print('solver using total energy gains')
        kg, kv, zeta = design_paras
        self.kv = kv
        self.zeta = zeta
        self.A = np.array([[0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0],
                           [-2*kv, 0, -zeta, 0, 2*kv, 0],
                           [0, -2*kv, 0, -zeta, 0, 2*kv],
                           [0, 0, 0, 0, -kg, 0],
                           [0, 0, 0, 0, 0, -kg]])
        print('Matrix A init')
        print(self.A)

        self.B = np.array([[0, 0],
                           [0, 0],
                           [0, 0],
                           [0, 0],
                           [kg, 0],
                           [0, kg]])
        print('Matrix B init')
        print(self.B)

        self.alpha, self.beta \
            = RGS_LTISolver.comp_const(self.A, self.B, dt)

        self.update_cnt = 0

    @staticmethod
    def comp_const(A, B, dt):
        """Compute equation update costants.
        """
#        A, B, dt = self.A, self.B, self.dt
        n = A.shape[0]
        expAdt = expm(A*dt)
        invA = inv(A)
        I6 = np.eye(n)
        alpha = expAdt
        beta = invA @ (expAdt - I6) @ B
        return alpha, beta

    @staticmethod
    def lsim(x0, alpha, beta, tvec, uvec, dt=0.05):
        """ Solve linear system states given simulation time vector and
        control input vector, return system state history.
        Time interval in tvec must match 'self.ts'.
        """
        nT = len(tvec)
        # intialization of variables
        xhist = np.zeros((nT, len(x0)))
        xhist[0] = x0
        xlast = x0
        # update for short time dt assume control is constant in dt interval
        for tidx in range(nT-1):
            if len(uvec) > 0:
                x_next = alpha @ xlast + beta @ uvec[tidx]
            else:
                x_next = alpha @ xlast

            xhist[tidx+1] = x_next
            xlast = x_next

        return xhist

    def get_max_dev(self, x0, T, xg, debug_info = False):
        """
        Given initial condition and local projected goal, compute maximal
        deviation along the trajectory wrt. local reference frame x-axis
        centered at xg(xg_bar), point to the opposite direction of xr0.
        Input:
            # x0: I.C
            # T: time horizon
            # xg_static: static governor position
        # Output:
            # maximum deivation and relative points coordinates
              wrt local reference frames
        """

        xvec0 = x0
        dt = self.dt
        xr0 = xvec0[0:2]
        xv0 = xvec0[2:4]
        nT = np.int(T/dt) + 1
        tvec = np.linspace(0,T,num=nT)
        uvec =  np.tile(xg,(nT,1))
        xhist = RGS_LTISolver.lsim(xvec0, self.alpha, self.beta, tvec, uvec)
        # compute projection on reference frame along the path
        u = xhist[:,0:2] - xg
        v = xr0 - xg
        v_perp = np.array([v[1], -v[0]])

        if norm(v) > 0:
            w1 = np.dot(u,v)/norm(v)**2
            up_path = w1.reshape(len(u),1) * v

            up_perp_path = u -  up_path
            up_len_path = norm(up_path,axis=1)
            up_perp_len_path = norm(up_perp_path,axis=1)
            da_idx, db_idx = np.argmax(up_len_path), np.argmax(up_perp_len_path);
            da, db = up_len_path[da_idx], up_perp_len_path[db_idx]
            da_pt, db_pt = xhist[da_idx, 0:2], xhist[db_idx, 0:2]

            e1, e2 = v/norm(v), v_perp/norm(v_perp)
            # compute extreme pts of bounding box
            p1 = xg + da * e1 + db * e2
            p2 = xg - da * e1 + db * e2
            p3 = xg - da * e1 - db * e2
            p4 = xg + da * e1 - db * e2
        else:
            da = 0
            up_perp_len_path = norm(u,  axis=1)
            db = np.max(up_perp_len_path)
            da_idx = 0
            db_idx = np.argmax(up_perp_len_path)

            if norm(xv0) > 0:
                e2 = xv0 / norm (xv0)
            else:
                e2 = 0
                print('already at equlibrium')

            p1 = xg + db * e2
            p2 = p1
            p3 = xg - db * e2
            p4 = p3
            da_pt, db_pt = xhist[da_idx, 0:2], xhist[db_idx, 0:2]

        box_bound_pts = [p1,p2,p3,p4]
        if debug_info == True:
            print('da_idx %d, db_idx %d  ' %(da_idx,db_idx),  end='')
            print('da_pt = [%.2f %.2f], db_pt = [%.2f %.2f]'\
               % (da_pt[0], da_pt[1], db_pt[0], db_pt[1]))
            print('box_bound_pts is %s' %(box_bound_pts))
        return box_bound_pts, da,db, da_pt, db_pt, xhist

    def update(self, xvec, u, filtering=True):
        """ One step LTI sytem update
        """
        xr, xv, xg = xvec[0:2], xvec[2:4], xvec[4:6]
        alpha_t = xg -  xr
        Q = get_dir_mat(self.c1, self.c2, alpha_t)
        eta_max_ANL, _ = find_eta_max_analytic(xvec, Q, 2 * self.kv, self.zeta)

        if len(self.eta_max_log_OPT) > 0:
            eta_opt_bk = self.eta_max_log_OPT[-1]
        else:
            eta_opt_bk = 0

        s0 = np.hstack((xg - xr, xv))
        Abar = self.A[0:4, 0:4]
        opt_status, _, eta_max_OPT, _ = find_eta_max_opt_scaling(s0, Q, Abar)
        # smoothing result reducing numerical solver noise
        if opt_status == 0:
            self.inaccurate_indices.append(self.update_cnt)
            if filtering is True:
                if eta_max_OPT > 1.1 * eta_opt_bk or eta_max_OPT < 0.9 * eta_opt_bk:
                    eta_max_OPT = eta_max_ANL

        else:
            if np.isinf(eta_max_OPT) is True:
                eta_max_OPT = eta_opt_bk
                self.solver_err_indices.append(self.update_cnt)


        xvec_new = self.alpha @ xvec + self.beta @ u
        self.PV = Q
        self.eta_max_log_ANL.append(eta_max_ANL)
        self.eta_max_log_OPT.append(eta_max_OPT)
        self.update_cnt +=1

        return xvec_new, Q, eta_max_ANL

    def update_simple(self, xvec, u):
        """ One step LTI sytem update
        """
        xvec_new = self.alpha @ xvec +  self.beta @ u
        return xvec_new
