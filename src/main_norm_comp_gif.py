#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate GIF for SDDM comparison in presentation slides.
Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# built-in
import numpy as np
import matplotlib
import matplotlib.patches as mp
from matplotlib import pyplot as plt
# personal
from my_utils import getRotCCW, save_fig_to_folder
from my_utils import pressQ_to_exist
from gov_ellipse import MyEllipse
from opt_solver import dist_pt2circle_arr_Pnorm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def set_canvas(this_ax, arrow_vec, circle_spec, ellipse_patch,
               tstr, xstar):
    """set canvas for plots"""
    this_ax.plot(0,0,'o', color='black')
    this_ax.plot(xstar[0], xstar[1], 'bs', markersize=5)

    this_ax.add_patch(ellipse_patch)
    this_circle = mp.Circle(circle_spec, rc, fc='gray')
    this_ax.add_patch(this_circle)

    # add arrow
    arrow_scale = 1
    arrow_size = 0.2
    arrow_vec_normalized = arrow_vec / np.linalg.norm(arrow_vec)
    dx, dy = arrow_vec_normalized * arrow_scale
    v1_arrow = mp.Arrow(xc[0], xc[1], dx, dy, \
                        color = 'magenta', width= arrow_size)

    this_ax.add_patch(v1_arrow)
    x1, y1 = circle_spec[0:2]
    # add text
    this_ax.text(-3.5, 1.8, tstr,  fontsize=16)
    this_ax.grid()
    this_ax.axis([-4, 6, -2.5, 2.5])
    this_ax.set_aspect('equal')
    return this_ax

#%% Module Test
if __name__ == '__main__':
    """ Create and test some ellipse examples"""
    a = 1
    b = 4
    xc = np.array([0,0])
    Pe1 = np.array([[ a , 0 ],[ 0,  b]])
    angle_vec = np.linspace(-90,90,10)
    video_frame_fd = 'sim_figs/norm_demo2D'
    for item in enumerate(angle_vec):
        frame_idx, angle = item
        print('frame_idx = %d, angle = %.2f' %(frame_idx, angle))
        Rccw_mat = getRotCCW(angle)
        Pe_rot = Rccw_mat @ Pe1 @ Rccw_mat.T
        rbt_vel = Rccw_mat @ np.array([1, 0])
        Pe_ball = np.array([[ 1 , 0 ],[ 0,  1]])

        # craete ellipse in energy form x'Px <= e
        my_ellipse1 = MyEllipse(Pe_ball, xc,  'e', E=1)
        my_ellipse2 = MyEllipse(Pe_rot,  xc,  'e', E=1)
        my_ellipse1._color = 'cyan'
        my_ellipse2._color = 'cyan'

        # Create circles obtables and find shortest distance to them.
        rc = 1.0
        x1, y1 = 4, 0
        circle_arr = np.array([[x1,y1,rc]])
        # compute distance
        dstar1, xstar1, dist_list1 = \
        dist_pt2circle_arr_Pnorm(xc, circle_arr, Pe_ball, extra_info = True)

        dstar2, xstar2, dist_list2 = \
        dist_pt2circle_arr_Pnorm(xc, circle_arr, Pe_rot, extra_info = True)

        dist2obs1 = dist_list1[0]
        dist2obs2 = dist_list2[0]

        ellipse_patch1 = my_ellipse1.get_ellipse_patch()
        ellipse_patch2 = my_ellipse2.get_ellipse_patch()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        tstr1 = 'Edist = %.3f' %(dstar1)
        tstr2 = 'Qdist = %.3f' %(dstar2)
        set_canvas(ax1, rbt_vel, circle_arr[0], ellipse_patch1, tstr1, xstar1)
        set_canvas(ax2, rbt_vel, circle_arr[0], ellipse_patch2, tstr2, xstar2)
        plt.tight_layout()

        frame_name = 'frame_' + str(10000+frame_idx)
        folder = '../sim_figs/norm_comp'
        save_fig_to_folder(fig, folder, frame_name)
    print('Result saved in %s'  %folder)
    pressQ_to_exist()

