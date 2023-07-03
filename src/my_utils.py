#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of common used utility functions

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
import os
import numpy as np
from numpy.linalg import norm
import matplotlib as mplt
import matplotlib.patches as mp
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
import decimal as dcm
import time

mplt.rcParams['pdf.fonttype'] = 42
mplt.rcParams['ps.fonttype'] = 42
#=======================================================================
# Constants
#=======================================================================
MY_EPS = 1e-6

def pressQ_to_exist():
    """ press key `Q` or `q` to exit
    """
    while True:
        key_got = input('Press [Q] to quit, Press [R] to replay \n')
        if key_got == 'q' or key_got == 'Q':
            print('Received %c, Program terminated'  %key_got)
            break
    else:
        pass
    return 0

def save_fig_to_folder(fig, folder, fname, dpi=300, ftype_ext='.png'):
    """ Save figure to specified location (create folder if it does not exist)
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    figname_full = os.path.join(folder, fname + ftype_ext)
    fig.savefig(figname_full, dpi=dpi, bbox_inches='tight')
    return 0

#https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    """ plot truncated colormap
    """
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def draw_path_2D(this_ax, traj, path_clr='cyan'):
    ''' h = draw_path_2D(h,traj)
        traj = num_traj x num_pts x num_dim
    '''
    if traj.ndim < 3:
        traj = traj[None, ...]

    if type(this_ax) is LineCollection:
        this_ax.set_verts(traj)
    else:
#        colors = [mcolors.to_rgba(c)
#                  for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        this_handle = this_ax.add_collection(LineCollection(traj,
                                            colors=path_clr,
                                            linewidth=2))
    return this_handle

# counting time
def tic():
    return time.time()

def toc(tstart, nm=""):
    print('%s elapsed: %.6f sec.\n' % (nm, (time.time() - tstart)))

def check_lyap(Abar, P):
    """check solution of lyapunov equation solution
    """
    Qbar = -(Abar.T @ P + P @ Abar)
    eigQbar, _ = np.linalg.eig(Qbar)
    print('Qbar = -(Abar.T @ P + P @ Abar), dots = Abar s')
    print('\neigQbar = %s' % eigQbar)

def debug_print(n, msg):
    """Debug printing for showing different level of debug information.
    """
    if n >= 0:
        tab_list = ['  '] * n
        dmsg = ''.join(tab_list) + 'DEBUG ' + msg
        print(dmsg)
    else:
        pass

def listmin(mylist):
    """ Find minimum and correspondint index given a list,
    if they are multiple min value elements in the list, take the 1st one.
    """
    dist_arr = np.array(mylist, dtype=float)
    min_idx_arr = np.argmin(dist_arr)
    if min_idx_arr.size > 1:
        min_idx = min_idx_arr[0]
    else:
        min_idx = min_idx_arr

    min_value = mylist[min_idx]
    return min_idx, min_value

def getRotCCW(theta_deg):
    """Get 2D conter-clockwise rotation matrix of theta deg.
    """
    q11 = np.cos(np.deg2rad(theta_deg))
    q12 = -np.sin(np.deg2rad(theta_deg))
    q21 = -q12
    q22 = q11
    Rmat = np.array([[q11, q12], [q21, q22]])
    return Rmat

def getRotpsd_theta(psdmat, theta):
    """ Get 2D rotated psd matrix, theta is measured in CCW
    """

    Rccw = getRotCCW(theta)
    return Rccw @ psdmat @ Rccw.T

def getRotpsd(psdmat, theta):
    """ Get 2D rotated psd matrix, theta is measured in CCW
    """

    Rccw = getRotCCW(theta)
    return Rccw @ psdmat @ Rccw.T

def getPdir(c1, c2, v):
    """ Get 2D rotated psd matrix
    """

    if norm(v) <= 1e-6:
        Pdir = np.diag((c1, c1))
    else:
        Pdir = c2 * np.eye(2) + (c1 - c2) * np.outer(v, v) / (norm(v) ** 2)

    return Pdir

def wall_corners2_ls(corners, show_debug=False):
    """Turn all wall corners to line segments.
    """
    ls_list = []
    for i in range(len(corners)-1):
        ls_list.append([corners[i], corners[i+1]])
    ls_list.append([corners[-1], corners[0]])
    if show_debug is True:
        print('wall line segments list')
        print(ls_list)
    return ls_list

def path2_ls(path, show_debug=False):
    """Turn a navigation path to a ordered collection of line segments.
    """
    ls_list = []
    end_idx = len(path) - 1
    for i in range(end_idx):
        ls_list.append([path[i], path[i+1]])

    if show_debug is True:
        print('wall line segments list')
        print(ls_list)
    return ls_list

def create_path_patch(path_array, lc='red'):
    """Create nav_path patch using navigation global waypoints.
    """
    verts = path_array
    path_len = len(path_array)
    codes = [Path.MOVETO] + [Path.LINETO] * (path_len - 1)
    path = Path(verts, codes)
    patch_path = mp.PathPatch(path, facecolor='none', ec=lc, lw=2)
    return patch_path, verts

def force2D_arr(input_array):
    """ Given a array (1D/2D) force it to be 2D so that there won't be
    unexpect behavior when loop the array
    """
    output_array = np.reshape(input_array, (-1, input_array.shape[-1]))
    return output_array

def Pnorm_square(P, x):
    """ compute quadractic norm square for vector "x" induced by psd matrix P
        y = x.T @ P @ x
    """
    return x.T @ P @ x

def Pnorm_len(P, x):
    """ compute quadractic norm for vector "x" induced by psd matrix P
        y = sqrt(x.T @ P @ x)
    """
    return np.sqrt(x.T @ P @ x)

def flist1D(input_list):
    """ control the precision of list
    """
    output_list = [float(dcm.Decimal("%.4f" % e)) for e in input_list]
    return output_list

def tolist_struct(input_list):
    """ Given a possible list with elements >=1, force it to be list
    in case when input_list only has one elements the type of input list
    will change leading unexpect behavior when loop the array
    """
    if type(input_list) != list:
        output_list = [input_list]
    else:
        output_list = input_list
    return output_list

def circle_objlist2arr(circle_obj_list):
    """Transform circle list to cirlce array
    each row of the array is a cirlce [x,y,r]
    """
    circle_obj_list = tolist_struct(circle_obj_list)
    n = len(circle_obj_list)
    circle_arr = np.zeros((n, 3))
    for ii in range(n):
        circle_obj = circle_obj_list[ii]
        [x, y], r = circle_obj.center, circle_obj.radius
        circle_arr[ii] = np.array([x, y, r])

    return circle_arr

def create_circle_patch(my_map, my_color='tab:grey', my_alpha=0.8):
    """
    Given a list of same objects: (circles, ploygons, ...), create a
    patch collection object with certain color and trasparent alpha.
    """
    circle_array = my_map.obstacles
    if len(circle_array) == 0:
        return []
    clist = []
    # PREVENT BUG: array size degeneration always turn it to 2D
    circle_array = np.reshape(circle_array, (-1, circle_array.shape[-1]))
    for item in circle_array:
        x, y, r = item
        circle = mp.Circle((x, y), r)
        clist.append(circle)
    patch_circles = PatchCollection(clist, color=my_color, alpha=my_alpha)
    return patch_circles

def set_canvas(boundary_pts, this_ax):
    """ Set canvas of ax using map boundary pts.
    """
    xl, xh, yl, yh = boundary_pts
    xrange = [xl, xh]
    yrange = [yl, yh]
    this_ax.set_xlim(*xrange)
    this_ax.set_ylim(*yrange)
    this_ax.grid()
    this_ax.set_aspect('equal')

def set_env(my_map, title_str=[], show_navpath=True):
    #        show_gov=True, show_robot=True, save_fig=True
    plt.ion()
    fig, ax = plt.subplots()
    start_pt = my_map.start_pt
    goal_pt = my_map.goal_pt
    path_array = my_map.nav_path
    boundary_pts = my_map.boundary
    # create circle patch to draw
    if len(my_map.obstacles) > 0:
        patch_circles = create_circle_patch(my_map)
        ax.add_collection(patch_circles)

    if show_navpath is True:
        # plot start and goal
        ax.plot(start_pt[0], start_pt[1], 'r*', markersize=12)
        ax.plot(goal_pt[0], goal_pt[1], 'g*', markersize=12)

        # plot navigation path
        patch_nav_path, verts_nav_path \
            = create_path_patch(path_array)

        ax.add_patch(patch_nav_path)
        # plot vertes of path
        xs, ys = zip(*verts_nav_path)
        ax.plot(xs, ys, 'o-', lw=2, color='black', ms=5, label='nav path')

    set_canvas(boundary_pts, ax)
    ax.set_title(title_str)
    ax.legend()

    return fig, ax

def get_rotR_from_two_pts(xp, yp):
    """ compute directional matrix given two pts
    """
    debug_level = -1

    eta = yp - xp
    eta1, eta2 = eta
    theta_deg = np.rad2deg(np.arctan2(eta2, eta1))

    dmsg1 = 'vector eta is %s theta_deg is %.2f ' % (eta, theta_deg)
    debug_print(debug_level, dmsg1)

    R1 = getRotCCW(theta_deg)
    dmsg2 = '\nRotation matrix from angle form is %s' % R1
    debug_print(debug_level, dmsg2)

    if norm(eta) <= 1e-6:
        R2 = np.eye(2)
    else:
        R2 = 1.0/norm(eta) * np.array([[eta1, -eta2], [eta2, eta1]])

    dmsg3 = '\nMatrix from vector form %s' % R2
    debug_print(debug_level, dmsg3)

    dmsg4 = '\nnorm(R1-R2) is %.2f\n' % norm(R1-R2)
    debug_print(debug_level, dmsg4)

    return R2, theta_deg

def get_rotPSD_pts(psd_mat, s, t):
    """ Given a psd matirx, and two points. Rotate the psd CCW, angle
    determined by vector s --> t start from xp points to yp
    """
    eta = t - s
    eta1, eta2 = eta

    if norm(eta) <= 1e-6:
        Rccw = np.eye(2)
        print('WARNING, angle is not well defined! \
              given points dist(xp, yp) = %e' % (norm(eta)))
    else:
        Rccw = 1.0/norm(eta) * np.array([[eta1, -eta2], [eta2, eta1]])

    return Rccw @ psd_mat @ Rccw.T

def get_dir_mat(c1, c2, x, eps=1e-8):
    """ compute directional matrix according to vector preference v
    """
    v = x.copy()
    n = len(v)
    if norm(v) <= eps:
        Q = c1 * np.eye(n)
    else:
        v = v/np.linalg.norm(v)
        Q = c2 * np.eye(n) + (c1 - c2) * np.outer(v, v.T)

    return Q

def wrap_angle_360(angle):
    #    https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
    # reduce the angle
    angle = np.array(angle % 360)

    # force it to be the positive remainder, so that 0 <= angle < 360
    angle = (angle + 360.0) % 360
    return angle

def wrap_angle_mp180(angle):
    #    https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
    angle = wrap_angle_360(angle)
    # force into the minimum absolute value residue class,
    # so that -180 < angle <= 180
    if angle > 180:
        angle -= 360
    return angle

def wrap_angle_pmp(angle_vec):
    """
    Normalize angle in radian to [-pi, pi)
    angle_vec: angle despcription in radian
    """
    angle_vec = (angle_vec + np.pi) % (2 * np.pi) - np.pi
    return angle_vec

def restrict_range(num, lower, upper):
    """
    restrict number to range [lower, uppper)
    """
    if lower >= upper:
        raise ValueError(
            "Invalid lower and upper limits: (%s, %s)" % (lower, upper))

    # regular case
    dd = upper - lower
    if num >= lower and num < upper:
        res = num
    elif num == upper:
        res = lower

    else:
        if num < lower:
            res = upper - np.array(lower - num) % dd
        if num > upper:
            res = lower + np.array(num - lower) % dd
    return res

def dist_angle(sourceA, targetA):
    #    https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    """
    compute distance between angles output diff in [-180,180)
    """
    diff = targetA - sourceA
    diff = (diff + 180) % 360 - 180
    return diff
