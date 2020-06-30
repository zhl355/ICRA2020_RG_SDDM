#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison norms in simple parallel wall env.
SDDM vs. Eculdiean metric.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""

import numpy as np
import matplotlib
import matplotlib.patches as mp
from matplotlib import pyplot as plt

from my_utils import getRotCCW, save_fig_to_folder
from gov_ellipse import MyEllipse
from opt_solver import dist_pt2seglist_Pnorm

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def vis_norm(wall_star, unit_dist_patch,
               tstr, figname, xc=np.zeros(2)):

    fig, ax = plt.subplots()
    # robot center
    ax.plot(0,0,'o', color='black')
    # wall
    ax.plot([-3,1], [-2,2], lw=5, color='brown')
    ax.plot([-1,3], [-2,2], lw=5, color='brown')

    # closet pt in Q-norm
    ax.plot(wall_star[0],wall_star[1],'s', color='blue')
    ax.add_patch(unit_dist_patch)

    # velocity arrow indicator
    arrow_scale = 0.5
    dx, dy = np.array([1,1]) * arrow_scale
    vel_arrow = mp.Arrow(xc[0], xc[1], dx, dy, \
                        color = 'green', width= 0.2)
    ax.add_patch(vel_arrow)
    ax.set_title(tstr, fontsize=20)
    ax.grid()
    ax.axis('equal')
    ax.axis([-2, 2, -2, 2])

#    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    folder = './sim_figs/summary'
    save_fig_to_folder(fig, folder, figname)
    return fig, ax

# %% Module Test
if __name__ == '__main__':
    """ Create and test some ellipse examples"""
    a = 1
    b = 4
    rbt_energy = 1

    xc = np.array([0,0])
    Pe1 = np.array([[ a , 0 ],[ 0,  b]])
    Rccw_45 = getRotCCW(45)
    Pe1_rot45 = Rccw_45 @ Pe1 @ Rccw_45.T

    Pe2_ball = np.array([[ 1 , 0 ],[ 0,  1]])
    # craete ellipse in energy form x'Px <= e

    my_ball = MyEllipse(Pe2_ball,  xc,  'e', E=rbt_energy)
    my_ellipse0 = MyEllipse(Pe1      , xc,  'e', E=rbt_energy)
    my_ellipse45  = MyEllipse(Pe1_rot45, xc,  'e', E=rbt_energy)

    my_ball._color = 'cyan'
    my_ellipse0._color = 'cyan'
    my_ellipse45._color = 'cyan'

    ball_patch = my_ball.get_ellipse_patch()
    ellipse_patch0 = my_ellipse0.get_ellipse_patch()
    ellipse_patch45 = my_ellipse45.get_ellipse_patch()

    """
    Create wall and compute distance to a wall
    using dist_2segment_list.
    """

    line_seg1 = [np.array([-3.0,-2.0]),np.array([1.0,2.0])]
    line_seg2 = [np.array([-1.0,-2.0]),np.array([3.0,2.0])]
    wall_ls = [line_seg1,line_seg2]

    dist2wall_ball,  wall_star_ball  = dist_pt2seglist_Pnorm(xc, wall_ls, Pe2_ball)
    dist2wall_rot0,  wall_star_rot0  = dist_pt2seglist_Pnorm(xc, wall_ls, Pe1)
    dist2wall_rot45, wall_star_rot45 = dist_pt2seglist_Pnorm(xc, wall_ls, Pe1_rot45)

    tstr_ball = 'Q-Dist. to wall %.2f' %(dist2wall_ball)
    tstr_rot0 = 'Q-Dist. to wall %.2f' %(dist2wall_rot0)
    tstr_rot45 = 'Q-Dist. to wall %.2f' %(dist2wall_rot45)

    vis_norm(wall_star_ball,  ball_patch, tstr_ball, 'norm_comp_ball.png')
    vis_norm(wall_star_rot0,  ellipse_patch0, tstr_rot0, 'norm_comp_rot0.png')
    vis_norm(wall_star_rot45, ellipse_patch45, tstr_rot45, 'norm_comp_rot45.png')
