#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Governor System Simulation
    Sparse box environment with circular obstacles or empty corridor
    Hand pick navigation path
    SDDM/Eculidean Metric

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import os
import pickle
import numpy as np
import matplotlib as mpl
from numpy.linalg import norm
# personal libraries
from gov_map import MyMap
from gov_map import OBSTACLES_2, NAV_PATH_2, BOUNDARY_PTS_2   # corridor env
from gov_map import NAV_PATH_14, OBSTACLES_1, BOUNDARY_PTS_1  # circular env
from rgs_sparse_known import RbtGovSys
from gov_log_viewer import GovLogViewer
from lti_solver import RGS_LTISolver
from traj_est import find_eta_max_analytic
from my_utils import tic, toc, pressQ_to_exist
from my_utils import getRotpsd_theta

# remove type3 fonts in figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# np float precision
np.set_printoptions(precision=4)

# %% Initialization
# timing
t1 = tic()
print('===========================Init Start==============================')
GOAL_REACH_TH = 1
# Metric Setup
c1, c2 = 1.0, 1.0 # Euclidean norm
#c1, c2 = 1.0, 4.0  # Quadratic norm eigenvalues

if np.abs(c2-c1) > 1e-3:
    energy_metric = 'ellipse'
else:
    energy_metric = 'ball'

corridor_sim = False
#corridor_sim = True

print('Energy metric is %s' % energy_metric)
if corridor_sim is False:
    print('Create general map for simualtion...')
    rgs_map = MyMap(BOUNDARY_PTS_1, OBSTACLES_1, NAV_PATH_14)
    xg0 = rgs_map.start_pt + \
        (NAV_PATH_14[1] - NAV_PATH_14[0]) / \
        norm((NAV_PATH_14[1] - NAV_PATH_14[0]))

    xvec0 = np.hstack((rgs_map.start_pt, np.zeros(2), xg0))
else:
    print('Create corridor map for simualtion...')
    rgs_map = MyMap(BOUNDARY_PTS_2, OBSTACLES_2, NAV_PATH_2)
    xvec0 = np.hstack((rgs_map.start_pt, np.zeros(
        2), rgs_map.start_pt + np.array([1, 0])))


print('Set up intial condition robot-governor system...')
Pd = np.diag([c1, c2])
xr0, xv0, xg0 = xvec0[0:2], xvec0[2:4], xvec0[4:6]  # initial states
dx0, dy0 = xg0 - xr0
theta_deg0 = np.rad2deg(np.arctan2(dy0, dx0))
PV0 = getRotpsd_theta(Pd, theta_deg0)               # initial metric
P_kinematic = np.eye(2)                             # kinematic matrix
Pd_str = 'Pd_diag_%d_%d' % (Pd[0][0], Pd[1][1])
print(Pd_str)
ev0 = (xg0 - xr0).T @ Pd @ (xg0 - xr0)

# controller desgin paras
kv, kg, zeta = c1, 1, 2 * np.sqrt(2)
design_parameter = [kv, kg, zeta]  # kv, kg, zeta
eta_max0,_ = find_eta_max_analytic(xvec0, PV0, 2*kv, zeta)

print('Design Parameters: [kv kg zeta] = [%.2f %.2f %.2f]' % (kv, kg, zeta))
print('Set up LTI solver for 2nd order system...')
# linear system solver
rgs_solver = RGS_LTISolver(design_parameter, Pd, PV0, dt=0.05)
# reference governor system
rgs = RbtGovSys(rgs_map, xvec0, eta_max0, energy_metric,
                Pd, PV0, P_kinematic, design_parameter,
                dt=0.05)

print(' xr = [%6.2f, %6.2f], xv = [%6.2f, %6.2f], xg = [%6.2f, %6.2f]'
      % (xvec0[0], xvec0[1], xvec0[2], xvec0[3], xvec0[4], xvec0[5]))
print('===========================Init Finish=============================')

# %% Main Loop
loop_cnt = 0
loop_err_cnt = 0
# while loop_cnt <= 3:
while loop_cnt <= 1000:
    time_now = rgs.dt * loop_cnt
    xvec = rgs.xvec
    dgg = norm(xvec[4:] - rgs.goal_pt)
    print('[ITER %3d | CLOCK %3d | %.2f sec] xr = [%6.2f, %6.2f], xg = [%6.2f, %6.2f] dgg = %.2f'
          % (loop_cnt, rgs.clock, time_now,  xvec[0], xvec[1],
             xvec[4], xvec[5],    dgg))
    # udpate system
    gov_status, xg_bar = rgs.update_gov()
    rgs.xvec, rgs.PV, rgs.eta_max \
        = rgs_solver.update(rgs.xvec, xg_bar)

    rgs.dvec_log.append(rgs.dvec)
    rgs.xvec_log.append(rgs.xvec)

    # retrive updated info
    xvec = rgs.xvec
    dvec = rgs.dvec
    deltaE = rgs.deltaE
    dgF, drg, drF = dvec

    if gov_status < 0:
        print('WARNING main loop error')

    # check if goal reached, if yes, print out result summary info
    if drg <= GOAL_REACH_TH and RbtGovSys.GOV_GOAL_REACHED_FLAG is True:
        print('========GOAL CONFIGURATION REACHED!===============')
        print('Time used: %.2f sec' % (rgs.dt * loop_cnt))
        print('Loop error cnt %d' % loop_err_cnt)
        dvec_log_array = np.array(rgs.dvec_log)
        dgF_min, _, drF_min = np.min(dvec_log_array, axis=0)
        drg_max = np.max(np.abs(dvec_log_array), axis=0)[1]
        deltaE_min = min(rgs.deltaE_log)
        print('deltaE_min is %.4f' % (deltaE_min))
        print('[dgF_min, drF_min, drg_max] = [%.2f, %.2f %.2f]'
              % (dgF_min, drF_min, drg_max))
        break
    loop_cnt = loop_cnt + 1

toc(t1, "Total simulation")
# %% saving simulation log
folder = '../log'
res_log = {'rgs': rgs, 'rgs_map': rgs_map, 'rgs_solver': rgs_solver}

if corridor_sim is True:
    log_filename = 'ec_' + energy_metric + Pd_str + '.pkl'
else:
    log_filename = 'sparse_' + energy_metric + Pd_str + '.pkl'

if not os.path.exists(folder):
        os.makedirs(folder)

logname_full = os.path.join(folder, log_filename)
print('Saving simulation result to %s' %logname_full)
with open(logname_full, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(res_log, f)

# %% Show the result
viewer = GovLogViewer(logname_full)
# viewer.play_animation(0.01, save_fig=True, fig_sample_rate=1)
viewer.show_trajectory()
# viewer.show_robot_gov_stat()
# viewer.compare_eta_max()
pressQ_to_exist()