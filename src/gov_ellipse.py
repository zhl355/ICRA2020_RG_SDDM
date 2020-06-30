#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ellipse Class for RGS.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import numpy as np
from numpy import linalg as LA
import matplotlib.patches as mp
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
# personal
from my_utils import listmin, getRotCCW, wall_corners2_ls, path2_ls, debug_print
from my_utils import create_path_patch, force2D_arr, circle_objlist2arr
from my_utils import draw_path_2D
from opt_solver import dist_ellipse_circle , dist_ellipse_line_segment
from opt_solver import dist_ellipse_pt_Pnorm, proj_ellipse_interseg_Pnorm

class EllipseError(Exception):
    """ User Defined Exceptions.
    """
    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "EllipseError exception: {0}".format(self.msg)
        else:
            return "EllipseError exception"


class MyEllipse:
    """
    Given a 2*2 Postive Definite matirx
    Construct the corresponding ellipse with different forms
        # (s) standard form: Ellipse = {x | (x-xc)^T Ps^{-1} (x-xc) <= 1}.
        # (e) energy   form: Ellipse = {x | (x-xc)^T Pe (x-xc) <= E}.
    """
    _ellipse_cnt = 0
    _color_list = ['m','red', 'blue']
    _alpha = 0.4
    _draw_limit = 10
    _ellipse_title_list = []
    ellipse_list = []

    def __init__(self, PV, xc, form = 's', E = 1, eps = 1e-6, clr = 'multi'):
        """
        Initalize ellipse object given matirx (Ps/Pe) and center xc.
        And other attributes of the ellipse.

        Get_ellipse_geometrics compute ellipse object attributes
        width, height, angle given matirx P and center xc.
        and set other attributes as initialized in advance.
        """
        # turn engery form to standard form
        if form == 'e':
            Ps = E * LA.inv(PV)
            self.PV = PV
        else:
            Ps = PV

        V, eig, _ = LA.svd(Ps)
        v1 = V[:, 0]
        # ellipse in standard form
        # get semi-axis length of the ellipse
        semi_len = np.sqrt(eig)
        MyEllipse._draw_limit = max(10, np.max(semi_len))
        # store all attributes
        self.center = xc
        self.width =  2*semi_len[0]
        self.height = 2*semi_len[1]

        self._ellipse_idx = MyEllipse._ellipse_cnt
        color_idx = self._ellipse_idx % len(MyEllipse._color_list)
        self._color = MyEllipse._color_list[color_idx]
        self._alpha = MyEllipse._alpha
        self.clr = clr
        self._Ps = Ps
        self._eig = eig
        self._v1 = v1
        self.eps = eps
        self.energy = E

        # add new ellipse in the class list and increase
        MyEllipse._ellipse_cnt = MyEllipse._ellipse_cnt + 1
        # get_ellipse_theta
        """ Get the orientation angle between x-axis positive and
        eigenvector v1, correspoinding to largest eigenvalue of P inverse
        eps: numerical zeros threshold (default eps = 1e-6).
        """
        dx, dy = v1
        if np.abs(dx) <= eps:
            if dx == 0:
                dx = eps
                # print('WARNING NUMERICAL ISSUE: attempt to divide by 0')
                print('dy/dx is %s / %s' % (dy, dx ))
            else:
                dx = np.sign(dx) * eps
        self.angle =  np.arctan(dy/dx)

    def get_ellipse_patch(self):
        """ get ellipse patch given attributes
        """
        # get bulti-in patch ellipse object
        ecx, ecy = self.center  # get center of ellipse
        width, height, angle = self.width, self.height, np.rad2deg(self.angle)

        if self.clr == 'mono':
            ellipse_color =  'grey'
        else:
            ellipse_color = self._color
        ellipse_patch = \
            Ellipse((ecx, ecy), width, height, angle, \
            color=ellipse_color, alpha=self._alpha)
        # add this patch to class ellipse list
        MyEllipse.ellipse_list.append(ellipse_patch)
        MyEllipse._ellipse_title_list.append(str(ellipse_patch))
        return ellipse_patch

    def get_ellipse_patch_color(self, ell_fc):
        """ get ellipse patches given attributes and face color profile
        """
        # get bulti-in patch ellipse object
        ecx, ecy = self.center  # get center of ellipse
        width, height = self.width, self.height
        angle = np.rad2deg(self.angle)

        ellipse_patch = \
            Ellipse((ecx, ecy), width, height, angle, \
            fc=ell_fc, ec='k', alpha=self._alpha)
        # add this patch to class ellipse list
        MyEllipse.ellipse_list.append(ellipse_patch)
        MyEllipse._ellipse_title_list.append(str(ellipse_patch))
        return ellipse_patch

    def dist_2segment_list(self, segment_list, \
                           norm_mat=np.eye(2), show_debug = False):
        """ Compute distance between a ellipse and a collection of line
        segments. For example to a wall contain the ellipse.
        """
        xstar_list = []
        zstar_list = []
        optval_list = []

        xc, Ps_inv = self.center, LA.inv(self._Ps)
        if type(segment_list[0]) == list:
            # many segments in the list
            for seg in segment_list:
                # compute distance to each line segment
                # print('check line segment %s' % (segment))
                dval, xval, zval \
                    = dist_ellipse_line_segment(seg, Ps_inv, xc, norm_mat)

                xstar_list.append(xval)
                zstar_list.append(zval)
                optval_list.append(dval)

            # find the minimum distance and corresponding optimal pts
            dist_min_idx , dstar = listmin(optval_list)
            xstar = xstar_list[dist_min_idx]
            zstar = zstar_list[dist_min_idx]
            if show_debug == True:
                print('xstar_list, zstar_list, dstar_list')
                print(xstar_list)
                print(zstar_list)
                print(optval_list)
            else:
                pass
        else:
            # only one segment in the list
            segment = segment_list
            dstar, xstar, zstar \
                = dist_ellipse_line_segment(segment, Ps_inv, xc, norm_mat)
            if show_debug == True:
                print('dstar is %.2f from xstar %s to zstar %s'\
                      %(dstar,xstar,zstar))

        return dstar, xstar, zstar

    def proj_2nav_path(self, nav_path, show_debug=False):
        """ Compute projected goal on navigation path. Backward checking
        each line segment (sA--sB) until found one with zero distance. Then
        find closest pt to sB lies in ellipse.

        # Output
            status  : status of this algorithm (0 succeeded, -1 failed)
            xstar   : projected pt
            sB_idx  : index of pt of segement (sA--sB) in navagation path
                      the ellipse is intersecting with
        """
        status, xstar, sB_idx = -1, [], -1 # intialized as failed status
        inter_seg  = []
        checked_segnum = 0
        segment_list = path2_ls(nav_path)
        xc, Ps_inv = self.center, LA.inv(self._Ps)

        # check if the nav_path end point already in ellipse
        nav_end = nav_path[-1]
        tmp = (xc - nav_end).T @ Ps_inv @ (xc - nav_end)
        if tmp < 1:
            print('nav path end %s is in ellipse tmp = %.4f !' %(nav_end, tmp))
            status = 0
            xstar = nav_end
            return status, xstar, sB_idx

        Nseg = len(segment_list)
        short_cut_succ = False
        Nseg_short = min([10, Nseg])

        nav_short_end = nav_path[Nseg_short]
        tmp = (xc - nav_short_end).T @ Ps_inv @ (xc - nav_short_end)
        if tmp < 1:
            print('nav_short end %s is in ellipse tmp = %.4f !' %(nav_short_end, tmp))
            print('Cannot do short cut alg')
        else:
            """ try to backward find intersects the ellipse by partial path"""
            # print('\tTry short on nav_path[0:%d] Nseg = %d' %(Nseg_short, Nseg))
            for ii in range(Nseg_short-1, -1, -1):
                segment = segment_list[ii]
                dstar, xstar, zstar \
                    = dist_ellipse_line_segment(segment, Ps_inv, xc, show_debug=False)
                # found segment intersect with ellipse and break loop
                checked_segnum += 1
                if dstar <= self.eps:
                    inter_seg = segment
                    short_cut_succ = True
                    break
                else:
                   pass

        if short_cut_succ is False and Nseg >= Nseg_short:
            checked_segnum = 0 # clear checkseg num
            """ try to backward find all of nav_path segment intersects the ellipse"""
            print('Use standard Way ')
            for ii in range(Nseg-1, Nseg_short-1, -1):
                segment = segment_list[ii]
                dstar, xstar, zstar \
                    = dist_ellipse_line_segment(segment, Ps_inv, xc, show_debug=False)
                # found segment intersect with ellipse and break loop
                checked_segnum += 1
                if dstar <= self.eps:
                    inter_seg = segment
                    # MyEllipse.debug_plot(self, nav_path[0:Nseg_short], xstar)
                    break
                else:
                    pass

        # cannot found segment intersect with ellipse
        if dstar > 2 * self.eps:
            print('Nav path closest distance to ellipse is %.6f' % (dstar))
            print('WARNING EllipseError Cannot Find Projected Goal')
            # raise EllipseError('Cannot Find Projected Goal')
            print ('Cannot Find Projected Goal checked %d segs' %checked_segnum)
            # print('segment list, Ps_inv, xc')
            # print(segment_list)
            print('ellipse Ps_inv')
            print(Ps_inv)
            print('ellipse center')
            print(xc)
            print('ellipse energy %.4f'  %(self.energy))
            print('Function output of dist_ellipse_line_segment')
            print('dstar, xstar, zstar')
            print(dstar,xstar,zstar)
            MyEllipse.debug_plot(self, nav_path, None)

        else:
            status = 0
            if short_cut_succ is False:
                sB_idx = Nseg - checked_segnum + 1
            else:
                sB_idx = Nseg_short - checked_segnum + 1

            if show_debug == True:
                print('\nFind line segment intersects with ellipse!')
                print('sB_idx is %d' %(sB_idx))
                print('the intersecitng line segment [%d] (0-based) is %s: '\
                      % (sB_idx-1, inter_seg))

            """
            After we can find the intersecting segment
            find the minimum distance ||x - sB||_2
            where x inside ellipse and lies in the segment.
            # you can possibly skip this step by changing rho to small value
            # in function dist_ellipse_line_segment, but will face numerical
            # problem sometimes
            """
            # projection on navigation path
            opt_status, dstar, xstar\
                 = proj_ellipse_interseg_Pnorm(inter_seg, Ps_inv, xc, show_debug)
            if opt_status < 0:
                print('SOLVER GLITCH, use backup')
                status = 0
            else:
                pass
        return status, xstar, sB_idx



    def dist_2circle_list(self, circle_obj_list, \
                          norm_mat=np.eye(2), show_debug = False):
        """ Compute distance between a ellipse and a collection of circles.
        circle_obj_list = [circle_obj1, circle_obj2,...]
        """
        circle_arr = circle_objlist2arr(circle_obj_list)
        return MyEllipse.dist_2circle_arr(self,circle_arr,norm_mat,show_debug)

    def dist_2circle_arr(self, circle_arr, \
                          norm_mat=np.eye(2), show_debug = False, extra_output= False):
        """ Compute distance between a ellipse and a collection of circles.
        circle_arr = [circle_vec1, circle_vec2,...] 2D array
        circle_vec = [xc1,xc2,r] 1D array
        """
        xstar_list = []
        zstar_list = []
        optval_list = []

        xc, Ps_inv = self.center, LA.inv(self._Ps)
        # incase there is only one obstacle in the array
        circle_arr = force2D_arr(circle_arr)
        for circle_vec in circle_arr:
            dval, xval, zval \
                = dist_ellipse_circle(circle_vec, Ps_inv, xc, norm_mat)
            xstar_list.append(xval)
            zstar_list.append(zval)
            optval_list.append(dval)

        # find the minimum distance and corresponding optimal pts
        dist_min_idx , dstar = listmin(optval_list)
        xstar = xstar_list[dist_min_idx]
        zstar = zstar_list[dist_min_idx]
        if show_debug == True:
            print('xstar_list, zstar_list, dstar_list')
            print(xstar_list)
            print(zstar_list)
            print(optval_list)
        else:
            pass
        if extra_output == False:
            return dstar, xstar, zstar
        else:
            return dstar, xstar, zstar, optval_list

    def dist_inside_pt2boundary(self, pt):
        """
        Compute distance between given pt to the boundary of the ellipse
        """
        b_sq = (self.height/2) ** 2
        a_sq = (self.width/2) ** 2
        x0, y0 = pt

        h_x0_sq = b_sq - b_sq/a_sq * x0 ** 2
        w_y0_sq = a_sq - a_sq/b_sq * y0 ** 2

        if h_x0_sq < -1e-16 or w_y0_sq < 1e-16:
            print('NUMERICAL ERROR ')
            print('h_x0_sq : %e, w_w_y0_sq: %e' %(h_x0_sq, w_y0_sq))

        h_x0 = np.sqrt(b_sq - b_sq/a_sq * x0 ** 2) # height at x0
        w_y0 = np.sqrt(a_sq - a_sq/b_sq * y0 ** 2) # width at y0
        dh = h_x0 - abs(y0)
        dw = w_y0 - abs(x0)
        dstar = min(dh, dw)
        dmsg  = 'dist_inside_pt2boundary'
        dmsg += 'h_x0 = %.2f w_y0 = %.2f dh= %.2f dw = %.2f' % (h_x0, w_y0, dh, dw)
        debug_print(-1,dmsg)

        return dstar

    @classmethod
    def show_all_ellipse(cls, show_titles = True):
        """ Display all the ellipses
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # add each ellipse and assemble a long title string full_str
        full_tstr = ''
        for ii in range(cls._ellipse_cnt):
            this_ellipse = cls.ellipse_list[ii]
            this_tstr = cls._ellipse_title_list[ii]
            full_tstr = full_tstr + this_tstr + '\n'
            ax.add_patch(this_ellipse)

        print(full_tstr)
        _draw_limit = MyEllipse._draw_limit
        # set ax
        ax.set_xlim([-_draw_limit, _draw_limit])
        ax.set_ylim([-_draw_limit, _draw_limit])
        ax.grid()
        plt.gca().set_aspect('equal')
        if show_titles == True:
            ax.set_title(full_tstr)
        plt.show()
        return fig, ax

    @classmethod
    def reset_class_list(cls):
        cls._ellipse_cnt = 0
        cls._ellipse_title_list = []
        cls.name_list = []
        cls.ellipse_list = []


    def debug_plot(self, test_path_arr, xg_bar):
        """ plot proj onto nav path"""
        ellipse_patch1 = MyEllipse.get_ellipse_patch(self)
        print('ellipse centered at ')
        ecx, ecy = ellipse_patch1.center
        print(ellipse_patch1.center)

        _, ax = plt.subplots()
        ax.add_patch(ellipse_patch1)
        ax.plot(ecx, ecy, 's', color='cyan')
            # plot the test navigation path
        patch_nav_path, verts_nav_path = create_path_patch(test_path_arr, 'black')
        ax.add_patch(patch_nav_path)
        xs, ys = zip(*verts_nav_path)
        ax.plot(xs, ys, '8-', lw=2, color='green', ms=5, label='nav path')
        # plot the projected goal
        if xg_bar is not None:
            ax.plot(xg_bar[0],xg_bar[1],'*', color='yellow')
        # canvas setting
        plt.gca().set_aspect('equal')

        xmin = min([self.center[0] - 2, 0])
        xmax = max([test_path_arr[-1][0] + 2, 10])
        print('xmax is %.2f' %xmax)
        ymin = min([self.center[1] - 2, 0])
        ymax = max([self.center[1] + 2, 2])

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.grid()
        plt.show()


def get_geometry_ellipse(PV, xc, form = 's', E = 1, eps=1e-3, angle_unit='deg'):
    """
    Given a 2*2 Postive Definite matirx
    Construct the corresponding ellipse with different forms
        # (s) standard form: Ellipse = {x | (x-xc)^T Ps^{-1} (x-xc) <= 1}.
        # (e) energy   form: Ellipse = {x | (x-xc)^T Pe (x-xc) <= E}.
    """

    # turn engery form to standard form
    if form == 'e':
        Ps = E * LA.inv(PV)
        PV = PV
    else:
        Ps = PV

    V, eig, _ = LA.svd(Ps)
    v1 = V[:, 0]


    # ellipse in standard form

    # get semi-axis length of the ellipse
    semi_len = np.sqrt(eig)
    # store all attributes
    center = xc
    width =  2*semi_len[0]
    height = 2*semi_len[1]

    # add new ellipse in the class list and increase
    # get_ellipse_theta
    """ Get the orientation angle between x-axis positive and
    eigenvector v1, correspoinding to largest eigenvalue of P inverse
    eps: numerical zeros threshold (default eps = 1e-6).
    """
    dx, dy = v1
    if np.abs(dx) <= eps:
        if dx == 0:
            dx = eps
            # print('WARNING NUMERICAL ISSUE: attempt to divide by 0')
            print('dy/dx is %s / %s' % (dy, dx ))
        else:
            dx = np.sign(dx) * eps

    angle = np.arctan(dy/dx)
    if angle_unit == 'deg':
        angle =  np.rad2deg(angle)

    area = np.pi * semi_len[0] *semi_len[1]
    geo_spec = [width, height, angle]
    extra_spec = [Ps, center, v1, area]

    return geo_spec, extra_spec


#%% Module Test
if __name__ == '__main__':
    """ Create and test some ellipse examples"""

#    NAV_PATH2 =np.array([[ 7.5 ,  4.  ],
#                         [ 7.75,  4.  ],
#                         [ 8.  ,  4.  ],
#                         [ 8.25,  4.  ],
#                         [ 8.5 ,  3.75],
#                         [ 8.75,  3.5 ],
#                         [ 9.  ,  3.25],
#                         [ 9.25,  3.25],
#                         [ 9.5 ,  3.25],
#                         [ 9.75,  3.25],
#                         [10.  ,  3.  ],
#                         [10.25,  3.  ],
#                         [10.5 ,  3.  ],
#                         [10.75,  3.  ],
#                         [11.  ,  3.  ],
#                         [11.25,  3.25],
#                         [11.5 ,  3.5 ],
#                         [11.75,  3.75],
#                         [11.75,  4.  ],
#                         [12.  ,  4.25],
#                         [12.25,  4.5 ],
#                         [12.5 ,  4.5 ],
#                         [12.75,  4.5 ],
#                         [13.  ,  4.5 ],
#                         [13.25,  4.5 ],
#                         [13.5 ,  4.5 ],
#                         [13.75,  4.5 ],
#                         [14.  ,  4.5 ],
#                         [14.25,  4.5 ],
#                         [14.5 ,  4.5 ],
#                         [14.75,  4.5 ],
#                         [15.  ,  4.5 ],
#                         [15.25,  4.5 ],
#                         [15.5 ,  4.5 ],
#                         [15.75,  4.5 ],
#                         [16.  ,  4.5 ],
#                         [16.25,  4.5 ],
#                         [16.5 ,  4.5 ],
#                         [16.75,  4.5 ],
#                         [17.  ,  4.5 ],
#                         [17.25,  4.25],
#                         [17.5 ,  4.25],
#                         [17.75,  4.  ],
#                         [18.  ,  4.  ]])
#    NAV_PATH2 = NAV_PATH2[0:10]

    NAV_PATH2 = np.array([[10, 10], [20, 10]])
    xg = NAV_PATH2[0]
    PV0 = np.diag([1,4])
    # my_ellipse1 = MyEllipse(PV0, xc, 'e', E=0.1436)

    deltaE = 4
    PV = np.array([[1, 0],[0, 4]])
    LEE_debug = MyEllipse(PV, xg, 'e', deltaE)  # LEE

    ellipse_patch1 = LEE_debug.get_ellipse_patch()
    status, xg_bar, sB_idx = LEE_debug.proj_2nav_path(NAV_PATH2, show_debug=True)
