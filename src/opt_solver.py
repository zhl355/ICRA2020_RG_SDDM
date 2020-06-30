#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Optimization Solver for Problems Related to Govornor
# Ref: https://web.stanford.edu/~boyd/lmibook/lmibook.pdf
"""
Robot Governor System Simulation Optimization library. 
Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM

Known Issue: SDP Numerical Problem founding output bounds.
"""
# python built in package
import numpy as np
from scipy.linalg import sqrtm
from numpy import linalg as LA
# third party
import cvxpy as cp
# personal
from my_utils import get_dir_mat
from my_utils import debug_print, listmin
from my_utils import force2D_arr, flist1D
from traj_est import find_eta_max_analytic


class OptError(Exception):
    """ User Defined Exceptions for optimization.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "OptError exception: {0}".format(self.msg)
        else:
            return "OptError exception"


def find_eta_max_opt(s0, P_dir, Abar, ignore_opt_error=True,
                     sdp_solver='SCS',
                     sdp_verbose=False):
    """
    HAVING NUMERICAL ISSUE !!!
    Find eta_max by formulated as output peak  by constructing invariant
    ellipse using 6.2.2 LMI book Boyd. We want to find PSD matirx minimize
    bounds on output z satisfies the following dynamics
    sdot = As
    z = Cz s
    In particular We want to have Cz = P_dir^(1/2) * T
    where Q = directional energy matrix, s = [x,dotx]
    xp = Ts = [I 0]s
    therefore z^T z = s^T T^T P_dir T s^T = x^T P_dir x
    if we can ensure max z^T z <= E0 = s0^T P_dir s0 we are in good shape
    """
    # small_nd = -1e-16 * np.eye(4)
    # small_pd = 1e-16 * np.eye(6)
    # print('-----------------------------------------')
    # print('Find find_eta_max_opt non-scaling')

    small_nd = 0
    small_pd = 0

    # state to position transformation
    T = np.block([[np.eye(2)], [np.zeros((2, 2))]]).T
    Cz = sqrtm(P_dir) @ T  # output linear transformation
    # s0 = xvec0[0:4] # initial condition of all system (x, xdot)

    P_boundz = cp.Variable((4, 4), PSD=True)  # optimized variables
    delta = cp.Variable()                  # optimized variables
    obj = cp.Minimize(delta)
    LMI_646 = cp.bmat([[P_boundz, Cz.T], [Cz, delta * np.eye(2)]])
    constraints = [(Abar.T @ P_boundz + P_boundz @ Abar) << small_nd]
    constraints = constraints + [LMI_646 >> small_pd]
    constraints = constraints + [s0.T @ P_boundz @ s0 <= 1]
    prob = cp.Problem(obj, constraints)
    opt_status = -1

    try:
        # prob.solve(solver=sdp_solver, verbose=sdp_verbose)
        prob.solve(solver=sdp_solver, verbose=sdp_verbose)

    except cp.error.SolverError:
        print('Solver Fatal Error! something wrong when find eta_max')
        print('return prob')
        print('DEBUG solver')
        print('prob.solver_stats')
        print(prob.solver_stats)
        print('prob.status')
        print(prob.status)
        print("----More info about s0, P_dir, Abar----")
        print(s0)
        print(P_dir)
        print(Abar)

        print("\n\n ----------- Try alternative solver ---------------\n\n")
        if sdp_solver == 'MOSEK':
            try:
                prob.solve(solver='SCS', verbose=False)
            except cp.error.SolverError:
                print('MOSEK->SCS Fatal Error Again! Give up')
                return -100, prob, np.inf, None
        else:
            try:
                prob.solve(solver='MOSEK', verbose=True)
            except cp.error.SolverError:
                print('SCS->MOSEK Solver Error Again! Give up')
                return -10, prob, np.inf, None

    if prob.status != 'optimal':

        eta_max_OPT = -1
        if ignore_opt_error is True and prob.status == 'optimal_inaccurate':
            opt_status = 0
            P_boundz = P_boundz.value
            eta_max_OPT = prob.value
            # print("using inaccurate solution = %.4f" %(eta_max_OPT))
        else:
            print("\n\n ---SCS/MOSEK Failed Try Alt Solver ----------\n\n")
            try:
                prob.solve(solver='MOSEK', verbose=True)
            except cp.error.SolverError:
                print('SCS->MOSEK, SolverError give up ')
                print("----More info about s0, P_dir, Abar----")
                print(s0)
                print(P_dir)
                return -10, prob, np.inf, None

            if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
                print("MOSEK works!!! prob.status = %s" %prob.status)
                opt_status = 50
                P_boundz = P_boundz.value
                eta_max_OPT = prob.value
            else:
                return -50, prob, np.inf, None

    else:
        opt_status = 100
        # get back values from cvx variables
        P_boundz = P_boundz.value
        # eta_max_OPT is the maximum delta value on ellipse
        eta_max_OPT = prob.value
        # print('optimal delta (eta_max) is %.2f' %(prob.value))

    return opt_status, prob, eta_max_OPT, P_boundz

def find_mve_2Dpts(pts_array):
    """
    Find minimum voluem ellipse cover a collection of 2D points
    using cvx parser, problem set up see [Boyd 8.4 Extremal Volume Ellisoids]
    Input:
    #   pts_array: 2D pts array
    Output: ellipse (x-xc).T P^-1 (x-xc) <= 1 covering those pts
    #   P: PSD matrix associated the ellipse
    #   xc: center of the ellipse
    """
    if len(pts_array) <= 2:
        raise OptError('find_mve: two few inputs points, at least 3 pts')
    # Create optimization variables, objective, and constraints
    # b = cp.Variable((2,1))
    b = cp.Variable(2)
    A = cp.Variable((2, 2), PSD=True)
    obj = cp.Minimize(-cp.log_det(A))  # objective
    constraints = []
    for x in pts_array:
        constraints = constraints + [cp.norm(A @ x + b) <= 1]

    constraints = constraints + [A.T == A]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    A, b = A.value, b.value

    if prob.status == 'optimal' or prob.status == 'optimal_inaccurate':
        #        print("optimal value", prob.value)
        #        print('optimal matrix A')
        #        print(A)
        #        print ('b is %s' %(b))
        P, xc = LA.inv(A @ A), - LA.inv(A) @ b
        return P, xc
    else:
        print("\nstatus:", prob.status)
        raise OptError('find_mve_failed')
    
def find_eta_max_opt_scaling(s0, P_dir, Abar):
    """
    Find eta_max by formulated as output peak  by constructing invariant
    ellipse using 6.2.2 LMI book Boyd. We want to find PSD matirx minimize
    bounds on output z satisfies the following dynamics

    Scaled version of find_eta_max_opt to overcome numerical
    problem. Consider normalized IC first and then scale it
    up if IC constraint is not satisfied
    """

    eta_max_OPT = np.inf
    opt_status = -1

    small_nd = 0
    small_pd = 0

    scale_factor = LA.norm(s0)
    s0_scaled = s0 / scale_factor


    # state to position transformation
    T = np.block([[np.eye(2)], [np.zeros((2, 2))]]).T
    Cz = sqrtm(P_dir) @ T  # output linear transformation

    P_boundz = cp.Variable((4, 4), PSD=True)  # optimized variables
    delta = cp.Variable()                     # optimized variables
    obj = cp.Minimize(delta)
    LMI_646 = cp.bmat([[P_boundz, Cz.T], [Cz, delta * np.eye(2)]])
    constraints = [(Abar.T @ P_boundz + P_boundz @ Abar) << small_nd]
    constraints = constraints + [LMI_646 >> small_pd]
    constraints = constraints + [s0_scaled.T @ P_boundz @ s0_scaled <= 1]

    prob = cp.Problem(obj, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)

    except cp.error.SolverError:
        print('SolverError! something wrong when find eta_max')
        print('return prob')
        print('DEBUG solver')
        print('prob.solver_stats')
        print(prob.solver_stats)
        print('prob.status')
        print(prob.status)
        return opt_status, prob, np.inf, np.eye(2)

    if prob.status != 'optimal':
        P_boundz = np.zeros(4)
        print('DEBUG solver')
        print('prob.solver_stats')
        print(prob.solver_stats)
        print('prob.status')
        print(prob.status)
        print('WARNING OPT solver encouter problem !\n')
    else:
        opt_status = 100
        P_boundz = P_boundz.value
        ic_quad = s0.T @ P_boundz @ s0
        if ic_quad > 1:
            # print("scale up s0.T @ P_boundz @ s0 = %.4f" %ic_quad)
            pass
        eta_max_OPT = prob.value * max(1, ic_quad)

    return opt_status, prob, eta_max_OPT, P_boundz


def dist_pt2circle_Pnorm(pt, circle_vec, norm_mat):
    """
    Compute distance between a point and a circle vec
    (center_x,center_y, radius) distance is measured in Pnorm
    induced by psd matrix P (norm_mat) |x|_norm_mat = sqrt(x^T norm_mat x)
    """
    debug_level = -1
    msg = 'circle_vec is ' + str(circle_vec)
    debug_print(debug_level, msg)
    xc, r = circle_vec[0:2], circle_vec[-1]
    debug_print(debug_level, 'dist_pt2seg')
    # optimzation variables
    # x = cp.Variable(2,1)                 # closet position inside circle
    x = cp.Variable(2)
    xbar = x - xc
    constraints = [cp.norm(xbar) <= r]     # x inside the circle
    obj = cp.Minimize(cp.quad_form(pt - x, norm_mat))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        raise OptError('dist_pt2circle_Pnorm failed')
    else:
        xstar = x.value
        dstar = np.sqrt(prob.value)
        dmsg1 = 'optimal dist pt %s to circle %s is %.4f from zstar %s' \
            % (pt, circle_vec, dstar, xstar)
        debug_print(debug_level, dmsg1)
    return dstar, xstar


def dist_pt2circle_arr_Pnorm(pt, circle_arr, norm_mat, extra_info=False):
    """
    Compute distance between a point and a set of n circle encapulated in a
    n*3 2D array, distance is measure in Pnorm
    """
    debug_level = -1
    debug_print(debug_level, 'dist_pt2circle_arr_Pnorm')
    dstar_list = []
    xstar_list = []
    circle_arr2D = force2D_arr(circle_arr)
    for item in circle_arr2D:
        dstar, xstar = dist_pt2circle_Pnorm(pt, item, norm_mat)
        dstar_list.append(dstar)
        xstar_list.append(xstar)
    # find minimum distance and corresponding optimal pt
    # (closest pt in obstacles)

    dist_min_idx, dstar = listmin(dstar_list)
    xstar = xstar_list[dist_min_idx]
    # dmsg1 = 'xstar_list' + str(xstar_list)
    dmsg2 = 'dist_pt2circle_arr_Pnorm dstar_list' + str(flist1D(dstar_list))
    # debug_print(1, dmsg1)
    debug_print(-1, dmsg2)
    if extra_info is False:
        return dstar, xstar
    else:
        return dstar, xstar, dstar_list


def dist_pt2seg_Pnorm(pt, seg, norm_mat):
    """
    Compute distance between a point and a line segment in quadratic norm
    induced by psd matrix P (norm_mat) |x|_norm_mat = sqrt(x^T norm_mat x)
    """
    debug_level = -1
    # optimzation variables
    alpha = cp.Variable()
    z1, z2 = seg
    z = alpha * z1 + (1 - alpha) * z2       # z on the segment
    constraints = [alpha <= 1, alpha >= 0]
    obj = cp.Minimize(cp.quad_form(pt - z, norm_mat))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        raise OptError('dist_ellipse_line_segments failed')
    else:
        a = alpha.value
        zstar = a * z1 + (1-a) * z2
        dstar = np.sqrt(prob.value)
        dmsg1 = 'dist_pt2seg_Pnorm pt %s to seg %s is %.6f from zstar %s' \
            % (pt, seg, dstar, zstar)
        debug_print(debug_level, dmsg1)
    return dstar, zstar

def dist_pt2seglist_Pnorm(pt, segment_list, norm_mat, debug_level=-100):
    """ Compute distance between a point and a collection of line
    segments. For example to a wall.
    """
    debug_level = -1
    zstar_list = []
    dstar_list = []
    debug_print(debug_level, 'dist_pt2seglist_Pnorm')
    if type(segment_list[0]) == list:
        # many segments in the list
        for seg in segment_list:
            # compute distance to each line segment
            #            print('check line segment %s' % (segment))
            dval, zval = dist_pt2seg_Pnorm(pt, seg, norm_mat)
            zstar_list.append(zval)
            dstar_list.append(dval)

        # find the minimum distance and corresponding optimal pts
        dist_min_idx, dstar = listmin(dstar_list)
        zstar = zstar_list[dist_min_idx]
        dmsg1 = 'zstar_list' + str(zstar_list)
        dmsg2 = 'dstar_list' + str(dstar_list)
        debug_print(debug_level, dmsg1)
        debug_print(debug_level, dmsg2)
        # if show_debug is True:
        #     print('zstar_list, dstar_list')
        #     print(zstar_list)
        #     print(dstar_list)
        # else:
        #     pass
    else:
        # only one segment in the list
        segment = segment_list
        dstar, zstar = dist_pt2seg_Pnorm(pt, segment, norm_mat)
        dmsg3 = 'dstar is %.2f from pt %s to zstar %s'\
            % (dstar, pt, zstar)
        debug_print(debug_level, dmsg3)
    return dstar, zstar


def dist_ellipse_circle(circle_vec, Ps_inv, xc,
                        norm_mat=np.eye(2), show_debug=False):
    """
    Compute distance between a ellipse and a circle_vec = [xc1,xc2,radius]
    Circle = {x | (x-xc)^T (x-xc) <= r^2}.
    """
    debug_print(-1, 'dist_ellipse_line_segment')
    zc, r = circle_vec[0:2], circle_vec[2]
    # optimzation variables
    x = cp.Variable(2)
    z = cp.Variable(2)

    xbar = x - xc                           # pt inside the ellipse
    zbar = z - zc                           # pt inside the circle
    constraints = [cp.quad_form(xbar, Ps_inv) <= 1]    #
    constraints = constraints + [cp.norm(zbar) <= r]
    if np.array_equal(norm_mat, np.eye(2)) is True:
        debug_print(-1, 'using 2norm')
        obj = cp.Minimize(cp.norm(x - z))
    else:
        P = norm_mat
        debug_print(-1, 'using P norm ')
        v_xz = x - z
        obj = cp.Minimize(cp.quad_form(v_xz, P))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        raise OptError('dist_ellipse_circle failed')
    else:
        xstar = x.value
        zstar = z.value
        if show_debug is True:
            print('\nDB:dist_ellipse_circle')
            print('optimal distance to %s' % (circle_vec))
            print("optimal xstar %s to zstar %s" % (xstar, zstar))
        if np.array_equal(norm_mat, np.eye(2)) is True:
            dstar = prob.value
        else:
            dstar = np.sqrt(prob.value)

    return dstar, xstar, zstar


def dist_ellipse_line_segment(seg, Ps_inv, xc,
                              norm_mat=np.eye(2), show_debug=False):
    """
    Compute distance between a ellipse and a line segment in quadratic norm
    or (Euclidean norm as a special case) induced by psd matrix P (norm_mat),
    |x|_norm_mat = sqrt(x^T norm_mat x)
    Ellipse = {x | (x-xc)^T Ps^{-1} (x-xc) <= 1}.
    """
    # optimzation variables
    x = cp.Variable(2)
    alpha = cp.Variable()
    msg1 = 'seg is ' + str(seg)
    debug_print(-10, msg1)
    z1, z2 = seg
    rho = 0
    xbar = x - xc                           # x inside the ellipse
    z = alpha * z1 + (1 - alpha) * z2       # z on the segment
    constraints = [cp.quad_form(xbar, Ps_inv) <= 1]    #
    constraints = constraints + [alpha <= 1, alpha >= 0]
    debug_print(-1, 'dist_ellipse_line_segment')
    # using 2norm
    if np.array_equal(norm_mat, np.eye(2)) is True:
        debug_print(-1, 'using 2norm')
        obj = cp.Minimize(cp.norm(x - z) + rho * cp.norm(z - z2))
    else:
        P = norm_mat
        debug_print(-1, 'using P norm ')
        v_xz = x - z
        obj = cp.Minimize(cp.quad_form(v_xz, P) + rho * cp.norm(z - z2))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        raise OptError('dist_ellipse_line_segments failed')
    else:
        a = alpha.value
        xstar = x.value
        zstar = a * z1 + (1-a) * z2
        if show_debug is True:
            print('\nDB:dist_ellipse_line_segment')
            print('optimal distance to seg %s is %.6f ' % (seg, obj.value))
            print("optimal xstar %s to zstar %s" % (xstar, zstar))
        # P norm need to sqrt of the result
        if np.array_equal(norm_mat, np.eye(2)) is True:
            dstar = prob.value
        else:
            dstar = np.sqrt(prob.value)

    return dstar, xstar, zstar


def proj_ellipse_interseg_Pnorm(seg, Ps_inv, xc, show_debug=False):
    """
    Find the projection point on intersecting segment by
    finding the minimum distance ||x - z2|| (z2:=sB 'right' end of segment)
    where x inside ellipse and lies in the segment.
    """
    status = 0
    dist_xsB = 0
    xstar = np.zeros(2)
    # optimzation variables
    # x = cp.Variable(2,1)                            # pt inside ellipse
    x = cp.Variable(2)
    alpha = cp.Variable()
    z1, z2 = seg
    xbar = x - xc
    # constrain pt inside the ellipse
    constraints = [cp.quad_form(xbar, Ps_inv) <= 1]
    constraints = constraints \
        + [alpha <= 1, alpha >= 0, x == alpha * z1 + (1 - alpha) * z2]

    # obj = cp.Minimize(cp.norm(x - z2, norm_mat))
    obj = cp.Minimize(cp.norm(x - z2))
    prob = cp.Problem(obj, constraints)


    try:
        prob.solve(solver=cp.MOSEK, verbose=False)

    except cp.error.SolverError:
        print('SolverError! something wrong when find proj_ellipse_interseg_Pnorm')
        print('return prob')
        print('DEBUG solver')
        print('prob.solver_stats')
        print(prob.solver_stats)
        print('prob.status')
        print(prob.status)


    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        print('seg is %s' % (seg))
        print('Ps_inv is')
        print(Ps_inv)
        print('xc is')
        print(xc)

        # raise OptError('proj_ellipse_interseg failed')
        print('WARNING proj_ellipse_interseg failed')
        status = -1
    else:
        xstar = x.value
        if show_debug is True:
            print('\nDB:proj_ellipse_interseg')
            print('|xstar - xc|_2 is %.2f' % (LA.norm(xstar - xc)))
            print("optimal xstar %s to sB %s" % (xstar, z2))

        if prob.value >= 0:
            dist_xsB = np.sqrt(prob.value)
        else:
            dmsg1 = 'proj_ellipse_interseg_Pnorm \
                    prob.value = %.6f runtime error' % prob.value
            debug_print(-1, dmsg1)
            dist_xsB = 0

    return status, dist_xsB, xstar


def dist_ellipse_pt_Pnorm(pt, Ps_inv, xc,
                          norm_mat=np.eye(2), show_debug=False):
    """
    Compute distance between a ellipse and a point in quadratic norm
    or (Euclidean norm as a special case) induced by psd matrix P (norm_mat),
    |x|_norm_mat = sqrt(x^T norm_mat x)
    Ellipse = {x | (x-xc)^T P^{-1} (x-xc) <= 1}.
    """
    # optimzation variables
    # x = cp.Variable(2,1)
    x = cp.Variable(2)
    msg1 = 'pt is ' + str(pt)
    debug_print(-10, msg1)
    xbar = x - xc                           # x inside the ellipse
    constraints = [cp.quad_form(xbar, Ps_inv) <= 1]    #
    debug_print(-1, 'dist_ellipse_line_segment')

    obj = cp.Minimize(cp.quad_form(x - pt, norm_mat))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print("\nstatus:", prob.status)
        raise OptError('dist_ellipse_pt_Pnorm failed')
    else:
        xstar = x.value
        if show_debug is True:
            print('\nDB:dist_ellipse_pt_Pnorm')
            print('optimal distance to pt %s is %.6f ' % (pt, obj.value))
            print("optimal xstar %s to pt %s" % (xstar, pt))
            dstar = np.sqrt(prob.value)

    return dstar, xstar


def vel_aligned(xr, xv, xg, eps_angle_deg=1e-2):
    """ Check if velocity direction is aligned with vector xr-->xg
    """
    rg = xg - xr
    rg_ort = np.rad2deg(np.arctan2(rg[1], rg[0]))
    # orientation of velocity
    vel_ort = np.rad2deg(np.arctan2(xv[1], xv[0]))
    angle_diff = abs(rg_ort-vel_ort)
    if angle_diff < eps_angle_deg:
        return True
    else:
        print('rg_ort is %.2f deg and vel_ort is %.2f' % (rg_ort, vel_ort))
        print('velocity is not perfect aligned, angle_diff is %.2f deg' %
              (angle_diff))
        return False


def find_eta_max_by_inspection_IC(c1, xr, xv, xg, Q):
    """ find eat max by inspecting whether velocity is aligned
    with vector xr-->xg. If so, we have a short cut to get
    eta_max = max_{t >= t0} x(t)^T P_dir x(t)
    eta_max happens at t=0, and eta_max = c1 ||xr - xg||_2^2
    """
    status = -1
    eta_max = -1
    align_flag = vel_aligned(xr, xv, xg)
    if align_flag is True:
        # eta_max = c1 * LA.norm(xr - xg)**2
        v = xr - xg
        eta_max = v.T @ Q @ v
        status = 100

    return status, eta_max

#%% module test 02/14/2020 some numerical error could happens

if __name__ == '__main__':
    s0 = np.array([-3.036,  -5.5661, 0.676,  1.2394]) # SCS Inaccurate (Failed before rounding) # MOSEK OPTIMAL
    # s0 = np.array([ 7.3308, 13.4397,  2.9188,  5.3512]) # SCS: optimal_inaccurate, MOSEK: UNKNOWN
    # s0 = np.array([7.8825, 14.4512,  5.8827, 10.785]) # SCS: optimal_inaccurate(infeasible_inaccurate before round), MOSEK: UNKNOWN,
    zeta = 2 * np.sqrt(2)
    design_parameter = [1, 1, zeta]
    Q = get_dir_mat(1, 4, s0[0:2])
    Abar = np.array(
            [[ 0,      0,      1,       0.    ],
             [ 0,      0,      0,       1.    ],
             [-2,      0,     -zeta,    0.    ],
             [ 0,     -2,      0,       -zeta]])
    states = np.hstack((s0, np.zeros(2)))

    status_IC_way, eta_max_OPT = \
        find_eta_max_by_inspection_IC(1, s0[0:2], s0[2:4], np.zeros(2), Q)
    if status_IC_way < 0:
        opt_status, prob, eta_max_OPT, P_boundz =  \
            find_eta_max_opt(s0, Q, Abar, sdp_solver='MOSEK', sdp_verbose=True)
        print("eig(P_boundz)")
        eigP,_ = LA.eig(P_boundz)
        print(eigP)

    print("===================== Solution Summary ===============================")
    eta_max_ANL, time_star_ANL = find_eta_max_analytic(states, Q, 2, zeta, debug=True)

    print("[ANL, OPT(IC)] = [%.4f, %.4f] \topt_status = %s\n" \
            % (eta_max_ANL, eta_max_OPT, status_IC_way))
