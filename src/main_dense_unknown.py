#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Governor System Simulation
    Clutter known environment with dense aribitrary shape obstacles
    A* replanning
    SDDM
Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import pickle
import numpy as np
import matplotlib
from numpy.linalg import norm
# third party
import trimesh
# personal
from rgs_dense_unknown import RbtGovSys
from rgs_mapping import RgsEnv, LaserSim
from traj_est import find_eta_max_analytic
from my_utils import get_rotPSD_pts, tic, toc
from lti_solver import RGS_LTISolver

# avoid type 3 fonts in pdf type figure
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# float precision
np.set_printoptions(precision=4)

# %% Initialization
t1 = tic()
print('===========================Init Start==============================')
c1, c2 = 1.0, 4         # eigenvalue of directional matrix
# c1, c2 = 1.0, 1         # Euclidean norm
vox_res = 0.25          # voxel resolution
GOAL_REACH_TH = 0.1     # goal reach threshold

if c2/c1 > 1:
    energy_metric = 'ellipse'
else:
    energy_metric = 'ball'
print('Energy metric is %s' % energy_metric)
print('Init environment...')
print('Import mesh enviroment for simualtion...')
mesh_JZ = trimesh.load_mesh('../mesh/dense_env.stl')
mesh_path = '../mesh/dense_env.stl'
ls = LaserSim()
start_pt = np.array([1, 0.75])
goal_pt = np.array([18, 4])
xg0 = start_pt + np.array([0.01, 0])
# init env class
rgs_env = RgsEnv(mesh_JZ, start_pt, goal_pt, vox_res)
print('Set up intial condition robot-governor system...')
# init states
xr0 = start_pt
xv0 = np.zeros(2)
xvec0 = np.hstack((xr0, xv0, xg0))
nav_path0 = np.vstack((start_pt, goal_pt))
# init metrics
Pd = np.diag([c1, c2])
PV0 = get_rotPSD_pts(Pd, xr0, xg0)
P_kinematic = np.eye(2)  # kinematic matrix
Pd_str = 'Pd_diag_%d_%d' % (Pd[0][0], Pd[1][1])
print(Pd_str)
# controller design paras
kv, kg, zeta = c1, 1, 2 * np.sqrt(2)
design_parameter = [kv, kg, zeta]  # kv, kg, zeta
print('Design Parameters: [kv kg zeta] = [%.2f %.2f %.2f]' % (kv, kg, zeta))
print('Set up LTI solver for 2nd order system...')
rgs_solver = RGS_LTISolver(design_parameter, Pd, PV0, dt=0.05)

# RGS class init
eta_max0,_ = find_eta_max_analytic(xvec0, PV0, 2*kv, zeta)
rgs = RbtGovSys(xvec0, goal_pt, eta_max0, energy_metric,
                Pd, PV0, P_kinematic, design_parameter, nav_path0,
                dt=0.05, path_res=rgs_env._vox_res)

print(' xr = [%6.2f, %6.2f], xv = [%6.2f, %6.2f], xg = [%6.2f, %6.2f]'
      % (xvec0[0], xvec0[1], xvec0[2], xvec0[3], xvec0[4], xvec0[5]))
print('===========================Init Finish==============================')

# %% Main Loop
loop_cnt = 0
loop_err_cnt = 0
# while loop_cnt <= 3:
while loop_cnt <= 2000:
    time_now = rgs.dt * loop_cnt
    xvec = rgs.xvec
    dgg = norm(xvec[4:] - rgs.goal_pt)
    # state monitor
    print('[ITER %3d | CLOCK %d | %.2f sec] xr = [%6.2f, %6.2f], xv = [%6.2f, %6.2f], xg = [%6.2f, %6.2f] dgg = %.2f'
          % (loop_cnt, rgs.clock, time_now,  xvec[0], xvec[1], xvec[2], xvec[3], xvec[4], xvec[5],    dgg))

    # update map
    rbt_loc = xvec[0:2]
    gov_loc = xvec[4:6]
    _, nav_path_future = rgs_env.update_map2D(mesh_JZ, ls, rbt_loc, gov_loc)
    dvec = rgs_env.dist_vec_Qnorm(xvec, rgs.PV)
    # udpate RGS
    rgs.nav_path = nav_path_future
    gov_status, xg_bar = rgs.update_gov(dvec)
    # dmsg = 'Evec = [deltaE, e_plus, e_rgs, et, ev, delta] %s' %flist1D(rgs.Evec)
    # debug_print(1, dmsg)
    rgs.xvec, rgs.PV, rgs.eta_max = rgs_solver.update(rgs.xvec, xg_bar)
    rgs.dvec_log.append(dvec)
    rgs.xvec_log.append(rgs.xvec)

    # retrive updated info
    xvec = rgs.xvec
    deltaE = rgs.deltaE
    dgF, drg, drF = dvec

    if gov_status < 0:
        print('WARNING main loop error')
        _, ax21, ax22 = rgs_env.map_vis_viewer(loop_cnt)

    # check if goal reached, if yes, print out result summary info
    if drg <= GOAL_REACH_TH and RbtGovSys.GOV_GOAL_REACHED_FLAG is True:
        print('========GOAL CONFIGURATION REACHED!===============')
        print('Time used: %.2f sec' % (rgs.dt * loop_cnt))
        print('Loop error cnt %d' % loop_err_cnt)
        dvec_log_array = np.array(rgs.dvec_log)
        dgO_min, _, drO_min = np.min(dvec_log_array, axis=0)
        drg_max = np.max(np.abs(dvec_log_array), axis=0)[1]
        deltaE_min = min(rgs.deltaE_log)
        print('deltaE_min is %.4f' % (deltaE_min))
        print('[dgO_min, drO_min, drg_max] = [%.2f, %.2f %.2f]'
              % (dgO_min, drO_min, drg_max))
        break
    loop_cnt = loop_cnt + 1

toc(t1, "Dense env total time ")
# %% saving simulation log
print('Saving simulation result...')
res_log = {'rgs': rgs, 'rgs_solver': rgs_solver}
log_filename = '../log/dense_rgs_' + energy_metric + Pd_str + '.pkl'
with open(log_filename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(res_log, f)

print('Result saved!')
mesh_path = '../mesh/dense_env.stl'
res_log = {'rgs':rgs,
           'rgs_solver':rgs_solver,
           'mesh_path':mesh_path,
           'occgrid_log':rgs_env.occgrid_log,
           'lidar_pt_log':rgs_env.lidar_endpts_log,
           'rbt_loc_log':rgs_env.rbt_loc_log,
           'nav_path_log':rgs_env.path_log,
           'vox_res':vox_res}

log_filename2 = '../log/dense_full_' + energy_metric + '.pkl'
with open(log_filename2, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(res_log, f)

print('Full Result saved!')