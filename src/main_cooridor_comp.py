#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of corridor simulations using different methods.
SDDM vs. Eculdiean metric.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
import numpy as np
from numpy.linalg import norm

import matplotlib as mpl
import matplotlib.patches as mp
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
#from gov_vis_cdc import GovLogViewer
from gov_vis_icra_circular import GovLogViewer
from my_utils import save_fig_to_folder
# remove type3 fonts in figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#%% Retrieve logs
log1 = './log/log_corridor_ballPd_diag_1_1.pkl'
log2 = './log/log_corridor_ellipsePd_diag_1_4.pkl'

viewer1 = GovLogViewer(log1)
viewer2 = GovLogViewer(log2)
#%% Same frame comparsion
frame_id = 200
frame_info1 = viewer1.extract_frame(frame_id)
frame_info2 = viewer2.extract_frame(frame_id)

# get common info
start_pt    = viewer1.map.start_pt
goal_pt     = viewer1.map.goal_pt
ball_size, arrow_size, arrow_scale = 5, 2, 1
DV = 0.01 # small constant
dt = viewer1.dt
# plot navigation path
patch_nav_path1, verts_nav_path1 = viewer1.create_path_patch('red')
patch_nav_path2, verts_nav_path2 = viewer2.create_path_patch('red')

# plot vertes of path
xs1, ys1 = zip(*verts_nav_path1)
xs2, ys2 = zip(*verts_nav_path2)

plt.ion()
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.4, 0.9, 0.35])
ax2 = fig1.add_axes([0.1, 0.1, 0.9, 0.35])
ax1.plot(start_pt[0], start_pt[1], 'r*',markersize=12)
ax2.plot(start_pt[0], start_pt[1], 'r*',markersize=12)
ax1.plot(goal_pt[0], goal_pt[1], 'g*',markersize=12)
ax2.plot(goal_pt[0], goal_pt[1], 'g*',markersize=12)
viewer1.set_canvas(ax1)
viewer2.set_canvas(ax2)

ax1.add_patch(patch_nav_path1)
ax2.add_patch(patch_nav_path2)
ax1.plot(xs1, ys1, 'o-', lw=2, color='black', ms=5, label='nav path')
ax2.plot(xs2, ys2, 'o-', lw=2, color='black', ms=5, label='nav path')

# for ball
xvec, dvec, evec, lpg, rho = frame_info1
xr, xv, xg = xvec[0:2], xvec[2:4], xvec[4:6]
dgF, drg, drF = dvec[0],dvec[1],dvec[2]
st =   start_pt
stx, sty = xg

local_energy_circle = mp.Circle(st, rho, fc='y',alpha=0.5)
gov_dgF_circle = mp.Circle(st, ball_size , fc='tab:grey', alpha=0.3)
local_energy_circle.center = xg
local_energy_circle.radius = rho
gov_dgF_circle.center = xg
gov_dgF_circle.radius = dgF


gov_pos1 = mp.Circle(xg, ball_size, fc='blue')
robot_pos1 = mp.Circle(xr, ball_size , fc='green')
lpg_pos1 = mp.Circle(lpg, ball_size/2 , fc='red')
# add common patches
ax1.add_patch(gov_pos1)
ax1.add_patch(robot_pos1)
ax1.add_patch(lpg_pos1)
ax1.add_patch(local_energy_circle)
ax1.add_patch(gov_dgF_circle)

# for ellipse
xvec, dvec, evec, lpg, rho, LEE_specs, GDE_specs = frame_info2
xr, xv, xg = xvec[0:2], xvec[2:4], xvec[4:6]
dgF, drg, drF = dvec[0],dvec[1],dvec[2]
st =   start_pt
stx, sty = xg

width, height, angle = LEE_specs
# local energy ellipse
LEE = Ellipse((xg[0], xg[1]), width, height,\
        angle, color='y', alpha=0.4)

width, height, angle = GDE_specs
# governor dgF ellipse
GDE = Ellipse((xg[0], xg[1]), width, height,\
        angle, color='grey', alpha=0.4)

gov_pos2 = mp.Circle(xg, ball_size, fc='blue')
robot_pos2 = mp.Circle(xr, ball_size , fc='green')
lpg_pos2 = mp.Circle(lpg, ball_size/2 , fc='red')
# add common patches
ax2.add_patch(gov_pos2)
ax2.add_patch(robot_pos2)
ax2.add_patch(lpg_pos2)

ax2.add_patch(LEE)
ax2.add_patch(GDE)

ax1.text(25, 25, 'Time: 10.00 sec', fontsize=12)
ax2.text(25, 25, 'Time: 10.00 sec', fontsize=12)

xl,xh, yl, yh = viewer1.map.boundary
offset_dd = 5.5
xrange = [xl-offset_dd+1, xh+offset_dd-2]
yrange = [yl-offset_dd+2, yh+offset_dd-0.5]

ax1.set_xlim(*xrange)
ax1.set_ylim(*yrange)

xrange = [xl-offset_dd+1, xh+offset_dd-2]
yrange = [yl-offset_dd+2, yh+offset_dd-0.5]
ax2.set_xlim(*xrange)
ax2.set_ylim(*yrange)

save_fig_to_folder(fig1, './sim_figs/summary', 'corridor_snapshot')

#%%  compare robot position converge time to goal
dist2G_ball    =  norm(goal_pt - viewer1.xvec_log[:,0:2], axis = 1)
dist2G_ellipse =  norm(goal_pt - viewer2.xvec_log[:,0:2], axis = 1)

len1 = len(dist2G_ball)
len2 = len(dist2G_ellipse)

plt.ion()
fig2, ax = plt.subplots()
time_vec1 = np.linspace(0, len1 * dt, len1 )
time_vec2 = np.linspace(0, len2 * dt, len2 )
ax.plot(time_vec1, dist2G_ball, color = 'b', lw = 2, label='Controller 1')
ax.plot(time_vec2, dist2G_ellipse, color = 'r', lw = 2, label='Controller 2')
plt.xlabel('Time (second)', fontsize = 12)
plt.ylabel('Euclidean Distance (m)', fontsize = 12)
ax.legend()
ax.grid()
#fig2.savefig('corridor_r2G_comp.pdf', dpi=300, bbox_inches='tight')
save_fig_to_folder(fig2, './sim_figs/summary', 'corridor_r2G_comp')