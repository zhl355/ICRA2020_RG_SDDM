#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare trajectory bounds using different metrics. Visualize bounds across
time by different colors.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# personal
from my_utils import get_dir_mat, pressQ_to_exist, save_fig_to_folder
from traj_est import find_eta_max_analytic
from rgs_sparse_known import RbtGovSys
from lti_solver import RGS_LTISolver
from gov_ellipse import MyEllipse, get_geometry_ellipse
from gov_map import MyMap, OBSTACLES_1, NAV_PATH_1, BOUNDARY_PTS_1

# remove type3 fonts in figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# np float precision
np.set_printoptions(precision=4)

#%% Initialization
print('===========================Init Start==============================')
energy_metric = 'ball'
xg0 = np.array([0,  0])
c1, c2 = 1.0, 4.0
zeta = 1
#zeta = 2 * np.sqrt(2) # critical damped  2*np.sqrt(2)
design_parameter =  [c1, 1, zeta] # kv, kg, zeta under-damped case
rgs_map = MyMap(BOUNDARY_PTS_1, OBSTACLES_1, NAV_PATH_1)
xr0 = np.array([-2, 0])
xv0 = np.array([0, 2])
xvec0 = np.hstack((xr0,xv0, xg0))
Pd = np.diag([c1, c2])
PV0 = get_dir_mat(c1, c2, xr0 - xg0)
PT = np.eye(2)
eta_max0 = 0 # just a place holder
rgs = RbtGovSys(rgs_map, xvec0, eta_max0, energy_metric, \
                Pd, PV0, PT, design_parameter, dt=0.01)
print('Dynamics: xddot = -2 kv * x - zeta * xdot')
print('Design Parameters: [kg kv zeta] = [%.2f %.2f %.2f]'\
      % (rgs.kg, rgs.kv, rgs.zeta))
Pd_str = 'SDDM Pd_diag_%d_%d'  %(Pd[0][0], Pd[1][1])
print(Pd_str)
print('Set up LTI solver for 2nd order system...')
rgs_solver = RGS_LTISolver([rgs.kg, rgs.kv, rgs.zeta], Pd, PV0, dt=rgs.dt)

goal_reach_th = 0.1
print('Initial state: ', end='')
print('xr = [%6.2f, %6.2f], xv = [%6.2f, %6.2f], xg = [%6.2f, %6.2f]' \
      %(xvec0[0],xvec0[1],xvec0[2],xvec0[3],xvec0[4],xvec0[5]))
theta_v0 = np.rad2deg(np.arctan2(xv0[1],xv0[0]))
plt.ion()
print('===========================Init Finish==============================')
# %% finite time horizon trajectory
sim_tlen = 10 # simulation duration
box_bound_pts, da,db, da_pt, db_pt, xhist  = \
     rgs_solver.get_max_dev(xvec0, sim_tlen, xg0, debug_info= False)

xp_traj = xhist
# %% visualization of initial vel
# arrow
#stx, sty, dx, dy,_ ,_ = xvec0
#arrow_scale = 1
#dx, dy = arrow_scale * dx, arrow_scale * dy
#v_arrow = mp.Arrow(stx, sty, dx, dy, color = 'red', width = 0.5)
#ax.add_patch(v_arrow)

#%% check how potential energy looks like at different time stamp
color_mat = np.array([  [     0,         0,    1.0000],
                        [     0,    0.5000,    1.0000],
                        [     0,    1.0000,    1.0000],
                        [0.5000,    1.0000,    0.5000],
                        [1.0000,    1.0000,         0],
                        [1.0000,    0.5000,         0],
                        [1.0000,         0,         0],
                        [0.5000,         0,         0]])

cm_list = color_mat.tolist()

frame_vec = np.array([0, 36, 74, 111, 151, 204, 309, 449]) * int(0.01/rgs.dt)
ellipse_cm = LinearSegmentedColormap.from_list('ellipse_cm', cm_list, N=len(frame_vec))
Ns = len(frame_vec)
Nticks = len(frame_vec)
tick_loc = np.linspace(0, 1, Nticks+1) + 1/(2*Nticks)
tick_loc= tick_loc[0:-1]
tick_str = [str(round(item*rgs.dt,1)) for item in frame_vec]


#%% SDDM
print('\n\n-------------------- SDDM ----------------------------')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xhist[:,0],xhist[:,1], 'k', lw = 1.5) # trajectory of x
plt.gca().set_aspect('equal')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.grid()
for ii in range(Ns):
    frame_id = frame_vec[ii]
    states = xhist[frame_id]
    xp_t = states[0:2]
    alpha_t = xp_t -  xg0
    Qt = get_dir_mat(c1, c2, alpha_t)
    ev_t0 = alpha_t.T @ Qt @ alpha_t
    eta_max_ANL, time_star_ANL = find_eta_max_analytic(states, Qt, 2, 1)
    print('[Iter %3d] x0 = [%7.4f %7.4f] xdot0 = [%7.4f %7.4f] \teta_max = %.4f'\
      %(frame_id, states[0], states[1], states[2], states[3], eta_max_ANL))

    ellipse_ev_t = MyEllipse(Qt, xg0, form = 'e', E = eta_max_ANL)
    ellipse_ev_t._alpha = 0.7
    ellipse_ev_t = ellipse_ev_t.get_ellipse_patch_color(color_mat[ii])
    ax.add_patch(ellipse_ev_t)

    frame_id = frame_vec[ii]
    states = xp_traj[frame_id]
    xp_t = states[0:2]
    ax.scatter(xp_t[0], xp_t[1], s=40, lw = 2, fc=color_mat[ii], ec='k', zorder=20+ii)

child_ax, bar_kw = mpl.colorbar.make_axes(ax, location='right', \
                                          fraction=0.15, shrink=1.0, aspect=20)
cb1 = mpl.colorbar.ColorbarBase(child_ax, cmap=ellipse_cm, alpha=0.7)


cb1.set_label('Time (seconds)')
cb1.set_ticks(tick_loc)
cb1.set_ticklabels(tick_str)
ax.set_title('Trajectory bound SDDM', fontsize=14)
plt.show()

folder = '../sim_figs/summary'
save_fig_to_folder(fig, folder, 'traj_bounds_sddm')

#%% Euclidean Norm
print('\n\n-------------------- Euclidean Norm ----------------------------')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xhist[:,0],xhist[:,1], 'k', lw = 1.5) # trajectory of x
plt.gca().set_aspect('equal')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.grid()
for ii in range(Ns):
    frame_id = frame_vec[ii]
    states = xhist[frame_id]
    Qt = np.eye(2)
    eta_max_ANL, time_star_ANL = find_eta_max_analytic(states, Qt, 2, 1)
    print('[Iter %3d] x0 = [%7.4f %7.4f] xdot0 = [%7.4f %7.4f] \teta_max = %.4f'\
      %(frame_id, states[0], states[1], states[2], states[3], eta_max_ANL))

    ellipse_ev_t = MyEllipse(Qt, xg0, form = 'e', E = eta_max_ANL)
    ellipse_ev_t._alpha = 0.7
    ellipse_ev_t = ellipse_ev_t.get_ellipse_patch_color(color_mat[ii])
    ax.add_patch(ellipse_ev_t)

    frame_id = frame_vec[ii]
    states = xp_traj[frame_id]
    xp_t = states[0:2]
    ax.scatter(xp_t[0], xp_t[1], s=40, lw = 2, fc=color_mat[ii], ec='k', zorder=20+ii)

child_ax, bar_kw = mpl.colorbar.make_axes(ax, location='right', \
                                          fraction=0.15, shrink=1.0, aspect=20)
cb1 = mpl.colorbar.ColorbarBase(child_ax,
                                cmap=ellipse_cm,
                                alpha=0.7)
cb1.set_label('Time (seconds)')
cb1.set_ticks(tick_loc)
cb1.set_ticklabels(tick_str)
ax.set_title('Trajectory bound 2-norm', fontsize=14)
plt.show()

folder = '../sim_figs/summary'
save_fig_to_folder(fig, folder, 'traj_bounds_2norm')
#%% Trajectory bound comparison

Npt = len(xhist)
etaQ_path = np.zeros(Npt)
areaQ_path = np.zeros(Npt)

etaE_path = np.zeros(Npt)
areaE_path = np.zeros(Npt)

for ii in range(Npt):
    s0 = xhist[ii]
    v0 = s0[0:2] - s0[-2:]
    Qt = get_dir_mat(c1, c2, v0)
    ev_t0 = v0.T @ Qt @ v0
    etaQ, _ = find_eta_max_analytic(s0, Qt, 2, 1)
    etaE, _ = find_eta_max_analytic(s0, np.eye(2), 2, 1) # etaE = x^T I x

    _, ellipse_spec = get_geometry_ellipse(Qt, s0[-2:], form = 'e', E = etaQ)
    areaQ = ellipse_spec[3]
    etaQ_path[ii] = etaQ
    areaQ_path[ii] = areaQ

    etaE_path[ii] = etaE
    areaE_path[ii] = etaE * np.pi

pressQ_to_exist()
##%% eta and area two plots in one figure
#Nplot = 500
#tvec = np.linspace(0, sim_tlen, Npt)
#fig2, (ax21,ax22) = plt.subplots(2, 1, sharex=True)
#fig2.suptitle("Trajectory bound comparison", fontsize=14)
#ax21.plot(tvec[:Nplot], etaQ_path[:Nplot], 'b-',  lw = 3, label=r'$\max ||x(t)||_{Q}^2$' )
#ax21.plot(tvec[:Nplot], etaE_path[:Nplot], 'r-',  lw = 3, label=r'$\max ||x(t)||_2^2$')
#ax21.grid()
#ax21.legend()
#ax22.plot(tvec[:Nplot], areaQ_path[:Nplot], 'b--',  lw = 3, label=r'$\pi a b}$' )
#ax22.plot(tvec[:Nplot], areaE_path[:Nplot], 'r--',  lw = 3, label=r'$\pi r^2}$')
#ax22.set_xlabel('Time (seconds)', fontsize=12)
#ax22.grid()
#ax22.legend()
#plt.show()
#
##%% eta and area all in one
#fig2, ax2 = plt.subplots()
#ax2.plot(tvec[:Nplot], etaQ_path[:Nplot], 'b-',  lw = 2, label=r'$\max ||x(t)||_{Q}^2$' )
#ax2.plot(tvec[:Nplot], etaE_path[:Nplot], 'r-',  lw = 2,  label=r'$\max ||x(t)||_2^2$')
#ax2.plot(tvec[:Nplot], np.sqrt(areaQ_path[:Nplot]), 'b--',  lw = 2, label=r'$\sqrt{\pi a b}$' )
#ax2.plot(tvec[:Nplot], np.sqrt(areaE_path[:Nplot]), 'r--',  lw = 2, label=r'$\sqrt{\pi r^2}$')
#ax2.set_xlabel('Time (seconds)', fontsize=12)
#ax2.grid()
#ax2.legend()
#plt.show()

