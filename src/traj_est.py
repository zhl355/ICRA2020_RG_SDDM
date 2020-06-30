#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory estimation for double integrator dynamics. Compute the output peak
of using analytical solution.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
import numpy as np

def get_quadratic_info(Q, kp, kd, x0, xdot0):
    """ Given design parameters and initial condition, find out
    damping condition and associated quadratic equation of critial point
    of ||x(t)||_Q^2

    Input:
        design para [kp, kd]
        ic: p.d. matrix Q, x0, xdot0
    Output:
        1. damping conditions
            [-1: underdamped, 0: critcally dampled, 1: overdamped]
        2. beta_sq
        3. simplified quadratic coeffs p = [a b c]
            for a * u **2 + b * u + c = 0
    """
    damped_status = 0
    damped_factor = kd ** 2 - 4 * kp  # scalar that determine damped status

    # constants
    m1 = x0.T @ Q @ x0
    m2 = xdot0.T @ Q @ xdot0
    m12 = - xdot0.T @ Q @ x0
    lam = kd / 2.0
    beta_sq = lam ** 2 - kp

    if damped_factor >= 0:
        if damped_factor > 1e-10:
            # print('system is overdamped')
            damped_status = 1
            beta = np.sqrt(beta_sq)
            beta_sq = lam ** 2 - kp
            # cosh >= 1 using u = tanh(beta t) is better don't need
            # to worry about zero divider issue
            a = (kd ** 3 / (4 * beta_sq) - kd) * m1 \
                + 2*(1 - kd**2/(2 * beta_sq)) * m12 \
                + kd / beta_sq * m2
            b = (kd ** 2 / (2 * beta) - 2 * beta) * m1 - 2.0/beta * m2
            c = 2 * m12

        else:
            # print('system is critically-damped')
            a = np.sqrt(kp) * (kp * m1 - 2 * np.sqrt(kp) * m12 + m2)
            b = kp * m1 - m2
            c = m12

    else:
        # print('system is underdamped')
        damped_status = -1
        beta_bar = np.sqrt(-beta_sq)

        a = (kd ** 3 / (4 * beta_sq) - kd) * m1 \
            + 2*(1 - kd**2/(2 * beta_sq)) * m12 \
            + kd / beta_sq * m2
        a = -a
        b = (kd ** 2 / (2 * beta_bar) + 2 * beta_bar) * m1 \
            - 2.0/beta_bar * m2
        c = 2 * m12

    return int(damped_status), beta_sq, a, b, c


def quadratic_solver(a, b, c, eps=1e-10):
    """ solve quadratic equation a * u **2 + b * u + c = 0
    if no valid solution using dummy solution u = 0
    """
    u_sln = np.zeros(2)
    if np.abs(a) < eps:
        if np.abs(b) < eps:
            if np.abs(c) < eps:
                # infinite many solution this should not ever happens for
                # stable linear system with nonzero IC
                pass
            else:
                print("a == 0, b == 0 c != 0 no solution!")
        else:
            u_sln = np.array([-c/b, -c/b])
    else:
        """ normal case a=! 0"""
        # print('phi(t)= %.2f t^2 + %.2f t + %.2f' %(a,b,c))
        Delta_phi = b ** 2 - 4 * a * c
        # print('Delta = b^2 - 4ac = %.2f' %(Delta_phi))
        if Delta_phi >= 0:
            u1 = 1.0/(2*a) * (-b - np.sqrt(Delta_phi))
            u2 = 1.0/(2*a) * (-b + np.sqrt(Delta_phi))
            u_sln = np.array([u1, u2])

    return u_sln


def find_critial_time(damped_status, beta_sq, a, b, c):
    """ solve quadratic a * u **2 + b * u + c = 0 to find critical point
    of t.
    Input:
        1. damping conditions
            [-1: underdamped, 0: critcally dampled, 1: overdamped]
        2. simplified quadratic coeffs p = [a b c]
            for a * u **2 + b * u + c = 0
    Output:
        critical time instances
    """
    u_sln = quadratic_solver(a, b, c)

    if damped_status == 0:
        # critically damped
        t_sln = u_sln

    if damped_status == -1:
        # under damped case u = tan(beta_bar t) could have periodic solution
        # we just need find the smallest positive one
        beta_bar = np.sqrt(-beta_sq)
        t_sln = np.mod(np.arctan(u_sln), np.pi) / beta_bar

    if damped_status == 1:
        # over-dampled case u = tanh(beta t)
        # restrict u_sln = [-1, 1] outside does not matter
        u_sln[u_sln < -1 + 1e-6] = 0
        u_sln[u_sln > 1 - 1e-6] = 0
        t_sln = np.arctanh(u_sln) / np.sqrt(beta_sq)

    # print("find_critial_time tsln = [%.4f, %.4f]" %(t_sln[0], t_sln[1]))
    t_sln[t_sln < 0] = 0
    return t_sln


def find_eta_max_analytic(s0, Q, kp, kd, debug=False):
    """
    Given initial condition s0 = (x0, xdot0), current Pt and fixed damping
    find out the maiximum of eta_max = max (x.T @ P x) for all time
    """
    x0, xdot0 = s0[-2:] - s0[0:2], s0[2:4]
    damped_status, beta_sq, a, b, c = \
        get_quadratic_info(Q, kp, kd, x0, xdot0)

    lam = kd / 2.0
    beta_sq = lam ** 2 - kp
    tvec = np.zeros(3)
    eta_vec = np.zeros(3)
    tvec[1:3] = find_critial_time(damped_status, beta_sq, a, b, c)

    for ii, t in enumerate(tvec):
        # critically damped
        if damped_status == 0:
            c3 = 1 + lam * t
            c4 = t
        # underdamped
        if damped_status == -1:
            beta_bar = np.sqrt(-beta_sq)
            c3 = np.cos(beta_bar * t) + lam/beta_bar * np.sin(beta_bar * t)
            c4 = np.sin(beta_bar * t) / beta_bar
        # overdamped
        if damped_status == 1:
            beta = np.sqrt(beta_sq)
            c3 = np.cosh(beta * t) + lam/beta * np.sinh(beta * t)
            c4 = np.sinh(beta * t) / beta

        xt = np.exp(-lam * t) * (c3 * x0 + c4 * xdot0)
        eta_vec[ii] = xt.T @ Q @ xt

    eta_max_idx = np.argmax(eta_vec)
    eta_star = eta_vec[eta_max_idx]
    time_star = tvec[eta_max_idx]
    if debug is True:
        print('analytical eta_max = [%.4f] happens at time = [%.4f]' % (
            eta_star, time_star))
        # print("r = %.4f" %np.sqrt(eta_star))
    return eta_star, time_star
