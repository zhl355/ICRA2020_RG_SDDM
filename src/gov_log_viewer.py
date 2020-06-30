#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization class for robot-governor simulation log

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import os
import time
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mp
from matplotlib.path import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
# third party
import trimesh
# personal
from my_utils import debug_print, flist1D
from my_utils import save_fig_to_folder, pressQ_to_exist
from rgs_mapping import RgsEnv
# remove type3 fonts in figure
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# np float precision
np.set_printoptions(precision=4)


class LogError(Exception):
    """ User Defined Exception for log file error.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "LogError exception: {0}".format(self.msg)
        else:
            return "LogError exception"


class GovLogViewer:
    """
    Class for governor result visualization.
    """

    def __init__(self, log_fn='res_logball.pkl'):
        """
        Init the view using saved log file
        """
        self._filename = log_fn
        # unpack log file
        if os.path.isfile(self._filename):
            with open(self._filename, 'rb') as f:
                res_log = pickle.load(f)
        else:
            err_msg = str(self._filename) + ' does not exists!'
            raise LogError(err_msg)


        # change configuration according to log filename
        if  self._filename == '../log/ec_ballPd_diag_1_1.pkl' \
            or self._filename == '../log/ec_ellipsePd_diag_1_4.pkl':
            self._frame_annotation= False
        else:
            self._frame_annotation = True

        # unpack more info for dense env setup
        if self._filename == '../log/dense_full_ellipse.pkl':
            print('Unpacking res_log ...')
            self.rgs = res_log['rgs']
            self.rgs_solver = res_log['rgs_solver']
            self.mesh_path = res_log['mesh_path']
            self.occgrid_log = res_log['occgrid_log']
            self.lidar_pt_log = res_log['lidar_pt_log']
            self.rbt_loc_log = res_log['rbt_loc_log']
            self.nav_path_log = res_log['nav_path_log']
            print('Unpack res_log succeed.')
        else:
            self.map = res_log['rgs_map']
            self.rgs = res_log['rgs']
            self.rgs_solver = res_log['rgs_solver']
        # history log container
        self.xvec_log = np.array(self.rgs.xvec_log)
        self.dvec_log = np.array(self.rgs.dvec_log)
        self.deltaE_log = np.array(self.rgs.deltaE_log)
        self.le_rho_log = np.array(self.rgs.le_rho_log)
        self.lpg_log = np.array(self.rgs.lpg_log)
        self.nav_adv_idx_log = self.rgs.nav_adv_idx_log
        self.Evec_log = np.array(self.rgs.Evec_log)
        self.eta_max_log_ANL = self.rgs_solver.eta_max_log_ANL
        self.eta_max_log_OPT = self.rgs_solver.eta_max_log_OPT

        self.start_pt = self.rgs.start_pt
        self.goal_pt = self.rgs.goal_pt
        # other parameters
        self.kv = self.rgs.kv
        self.dt = self.rgs.dt
        self.energy_metric = self.rgs.energy_metric
        if self.energy_metric == 'ellipse':
            self.LEE_specs_log = np.array(self.rgs.LEE_specs_log)
            self.GDE_specs_log = np.array(self.rgs.GDE_specs_log)
            theta_vec = self.LEE_specs_log[:, 2]
            xv_log = self.xvec_log[1:, 2:4]
            xv_dir_vec = np.rad2deg(np.arctan2(xv_log[:, 1], xv_log[:, 0]))
            N1 = len(xv_dir_vec)
            N2 = len(theta_vec)
            N = min(N1, N2)
            self.angle_diff = xv_dir_vec[0:N] - theta_vec[0:N]

    def create_circle_patch(self, my_color='tab:grey', my_alpha=0.8):
        """
        Given a list of same objects: (circles, ploygons, ...), create a
        patch collection object with certain color and trasparent alpha.
        """
        circle_array = self.map.obstacles
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

    def set_canvas(self, this_ax, axis_equal=True):
        """ Set canvas of ax using map boundary pts.
        """
        wt = 4  # wall thickness
        xl, xh, yl, yh = self.map.boundary
        xrange = [xl-wt, xh+wt]
        yrange = [yl-wt, yh+wt]
        this_ax.set_xlim(*xrange)
        this_ax.set_ylim(*yrange)
        this_ax.grid()
        if axis_equal is True:
            this_ax.set_aspect('equal')
        else:
            pass
        # plot walls
        wall_corners = self.rgs.map.corners + \
            np.array([[-wt, -wt], [-wt, wt], [wt, wt], [wt, -wt]])/2
        wall_plg = mp.Polygon(wall_corners)

        patch_wall_plg = PatchCollection([wall_plg], lw=wt,
                                         edgecolor='brown', facecolor='none')
        this_ax.add_collection(patch_wall_plg)

    def create_path_patch(self, lc):
        """Create nav_path patch using navigation global waypoints.
        """
        path_array = self.map.nav_path
        verts = path_array
        path_len = len(path_array)
        codes = [Path.MOVETO] + [Path.LINETO] * (path_len - 1)
        path = Path(verts, codes)
        patch_path = mp.PathPatch(path, facecolor='none', ec=lc, lw=2)
        return patch_path, verts

    def compare_eta_max(self, save_fig=True, fig_folder='../sim_figs/summary'):
        """ Compare eta_max between ANL way and OPT way
        """

        eta_max_ANL_vec = np.array(self.eta_max_log_ANL)
        eta_max_OPT_vec = np.array(self.eta_max_log_OPT)
        N = len(eta_max_ANL_vec)  # pt number of log
        time_vec = np.linspace(0, N * self.dt, N)
        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(time_vec, eta_max_ANL_vec, 'r-', lw=2, label='ANL')
        ax.plot(time_vec, eta_max_OPT_vec, 'b-', lw=2, label='OPT')
        ax.grid()
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_aspect(0.005)
        if save_fig is True:
            save_fig_to_folder(fig, fig_folder, 'eta_max_comparison')
        return fig, ax

    def show_trajectory(self, fig_folder='../sim_figs/summary'):
        """ Display trajectories of final result
        """

        plt.ion()
        fig, ax = plt.subplots()
        start_pt = self.map.start_pt
        goal_pt = self.map.goal_pt
        xvec_path = self.xvec_log

        # create circle patch to draw
        if len(self.map.obstacles) > 0:
            patch_circles = GovLogViewer.create_circle_patch(self)
            ax.add_collection(patch_circles)

        # plot start and goal
        ax.plot(start_pt[0], start_pt[1], 'r*', markersize=12)
        ax.plot(goal_pt[0], goal_pt[1], 'g*', markersize=12)

        # plot navigation path
        patch_nav_path, verts_nav_path \
            = GovLogViewer.create_path_patch(self, 'black')
        ax.add_patch(patch_nav_path)

        # plot vertes of path
        xs, ys = zip(*verts_nav_path)
        ax.plot(xs, ys, 'o--', lw=2, color='black', ms=5, label='nav path')

        # retrieve governor and robot trajectories and plot them
        xg_path = xvec_path[:, -2:]
        xv_path = xvec_path[:, 2:4]
        xr_path = xvec_path[:, 0:2]

        xg_coord, yg_coord = xg_path[:, 0], xg_path[:, 1]
        ax.plot(xg_coord, yg_coord, 'o--', lw=2, color='blue', ms=2, label='governor path')

        xr_coord, yr_coord = xr_path[:, 0], xr_path[:, 1]
        ax.plot(xr_coord, yr_coord, '*:', lw=2, color='green', ms=2, label='robot path')

        # add normalized arrow indicating speed
        N = len(xg_path)

        for tidx in range(1, N, 3):
            stx, sty = xr_path[tidx]
            xv = xv_path[tidx]
            dx, dy = -xv[1], xv[0]
            vp_arrow = mp.Arrow(stx, sty, dx, dy, color='m', width=0.1)
            ax.add_patch(vp_arrow)

        GovLogViewer.set_canvas(self, ax)
        ax.legend(loc='best')

        finish_time = (len(xg_path) - 2) * self.dt
        text_msg = 'Finish Time: %.2f sec' % (finish_time)
        status_text1 = ax.text(160, 20, text_msg, fontsize=12)
        status_text1.set_text(text_msg)

        figname = 'rgs_traj_%s' % (self.energy_metric)
        # figname_wpath = os.path.join(fig_folder, figname)
        # plt.savefig(figname_wpath, dpi=300, bbox_inches='tight')
        save_fig_to_folder(fig, fig_folder, figname)
        plt.show()
        return fig, ax

    def cpt_pct_list(self, alist, tlist):
        """Compute percentage of  item_a/ item_t from alist, tlist.
        """
        plist = [x/y*100.0 for x, y in zip(alist, tlist)]
        return plist

    def show_robot_gov_stat(self, fig_folder='../sim_figs/summary'):
        """ Show robot governor states from log file.
        """
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)

        dvec_log = self.dvec_log
        Evec_log = self.Evec_log
        deltaE_path = self.deltaE_log

        dgF_path = dvec_log[:, 0]
        drg_path = dvec_log[:, 1]
        drF_path = dvec_log[:, 2]
        eplus_path = Evec_log[:, 1]

        N = len(deltaE_path)  # pt number of log
        time_vec = np.linspace(0, N * self.dt, N)

        # Robot and governor distance to Free space
        plt.ion()
        fig, axarr = plt.subplots(2, sharex=True)
        ax1, ax2 = axarr[0], axarr[1]

        if self.energy_metric == 'ball':
            ax1.plot(time_vec, drF_path, color='red', label=r'$d_2(x, \mathcal{O})$')
            ax1.plot(time_vec, dgF_path, color='green',label=r'$d_2(g, \mathcal{O})$')
            ax1.plot(time_vec, drg_path, color='blue', label=r'$d_2(g, x)$')
            ax2.plot(time_vec, eplus_path, label=r'$d_2^2(g, \mathcal{O})$')
            ax2.plot(time_vec, deltaE_path, label=r'$\Delta\, E_1(\mathbf{x})$')
        else:
            ax1.plot(time_vec, drF_path, color='red', label=r'$d_Q(x, \mathcal{O})$')
            ax1.plot(time_vec, dgF_path, color='green', label=r'$d_Q(g, \mathcal{O})$')
            ax1.plot(time_vec, drg_path, color='blue', label=r'$d_Q(g, x)$')
            ax2.plot(time_vec, eplus_path, label=r'$d_P^2(g, \mathcal{O})$')
            ax2.plot(time_vec, deltaE_path, label=r'$\Delta\, E_2(\mathbf{x})$')

        ax1.grid()
        ax2.grid()
        ax1.legend(loc='upper right', fontsize='large')
        ax2.legend(loc='upper right', fontsize='large')
        ax2.set_xlabel('Time (seconds)', fontsize=14)

        figname = 'rgs_dist_' + self.energy_metric
        # plt.savefig(figname, dpi=300, bbox_inches='tight')
        save_fig_to_folder(fig, fig_folder, figname)
        plt.show()

        return 0

    def play_animation(self, dt_play=0.05, save_fig=False,
                       fig_sample_rate=1,
                       fig_folder='../sim_figs/ellipse_opt/'):

        """ Display an animation to show how RGS runs.
        """

        plt.ion()
        fig, ax = plt.subplots()
        energy_metric = self.energy_metric
        start_pt = self.map.start_pt
        goal_pt = self.map.goal_pt
        xvec_path = self.xvec_log
        dvec_log = self.dvec_log
        xg_bar_path = self.lpg_log
        rho_path = self.le_rho_log
        evec_log = self.Evec_log
        static_obs = self.map.obstacles.copy()

        if energy_metric == 'ellipse':
            LEE_specs_path = self.LEE_specs_log
            GDE_specs_path = self.GDE_specs_log

        # plot start and goal
        ax.plot(start_pt[0], start_pt[1], 'r*', markersize=12)
        ax.plot(goal_pt[0], goal_pt[1], 'g*', markersize=12)

        # plot navigation path
        patch_nav_path, verts_nav_path = GovLogViewer.create_path_patch(self, 'red')
        ax.add_patch(patch_nav_path)

        # plot vertes of path
        xs, ys = zip(*verts_nav_path)
        ax.plot(xs, ys, 'o-', lw=2, color='black', ms=5, label='nav path')

        # get back governor and robot trajectories and plot them
        xg_path = xvec_path[:, -2:]
        xr_path = xvec_path[:, 0:2]
        xv_path = xvec_path[:, 2:4]

        GovLogViewer.set_canvas(self, ax)
        ax.set_title('Robot-Governor System Simulation')

        dgF_path = dvec_log[:, 0]
        st = start_pt
        stx, sty = start_pt
        ball_size, arrow_size, arrow_scale = 5, 2, 1
        DV = 0.01  # small constant

        # create animation circles/ellipse
        gov_pos = mp.Circle(st, ball_size, fc='blue')
        robot_pos = mp.Circle(st, ball_size, fc='green')
        lpg_pos = mp.Circle(st, ball_size/2, fc='red')

        # create circle patch to draw
        if len(self.map.obstacles) > 0:
            patch_circles = GovLogViewer.create_circle_patch(self)
            ax.add_collection(patch_circles)

        # initialize patcheds local safe zone and governor free space zone
        if energy_metric == 'ball':
            local_energy_circle = mp.Circle(st, rho_path[0], fc='y', alpha=0.5)
            gov_dgF_circle = mp.Circle(st, ball_size, fc='tab:grey', alpha=0.3)
            if len(static_obs) > 0:
                print('Debug init obstalce one location %s' %
                      flist1D(static_obs[0][0:2]))
            ax.add_patch(local_energy_circle)
            ax.add_patch(gov_dgF_circle)

        else:
            # local energy ellipse
            LEE = Ellipse((stx, sty), ball_size, ball_size, 0, color='y', alpha=0.4)
            # governor dgO ellipse
            GDE = Ellipse((stx, sty), ball_size, ball_size, 0, color='grey', alpha=0.4)
            ax.add_patch(LEE)
            ax.add_patch(GDE)

        # create arrows
        # v1: robot->gov, move it parallely centered at gov
        # v2: gov->lpg
        # v3: robot vel, move it parallely centered at gov

        v1_arrow = mp.Arrow(stx, sty, DV, DV, color='green',width=arrow_size)
        v2_arrow = mp.Arrow(stx, sty, DV, DV, color='blue', width=arrow_size)
        v3_arrow = mp.Arrow(stx, sty, DV, DV, color='cyan', width=arrow_size)

        # add common patches
        ax.add_patch(gov_pos)
        ax.add_patch(robot_pos)
        ax.add_patch(lpg_pos)
        ax.add_patch(v1_arrow)
        ax.add_patch(v2_arrow)
        ax.add_patch(v3_arrow)

        # add annotation of time
        # status_text1 = ax.text(0.03, 0.70, '', transform=ax.transAxes)
        status_text1 = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=14)
        # animation loop
        tidx = 0  # animation counter
        while tidx < len(xr_path) - 1:
            #        while tidx < 30:
            xvec = xvec_path[tidx]
            xv = xvec[2:4]
            dgO = dgF_path[tidx]
            evec = evec_log[tidx]
            xr, xg, xg_bar = xr_path[tidx], xg_path[tidx], xg_bar_path[tidx]
            rho = rho_path[tidx]
            if tidx >= 0:
                time_now = tidx * 0.05
                dmsg = '\t\t[deltaE, e_plus, e_rgs, ep, ev, ez] is %s' \
                    % flist1D(evec)
                debug_level = -1
                debug_print(debug_level, dmsg)
            # update ball patches
            if energy_metric == 'ball':
                local_energy_circle.center = xg
                local_energy_circle.radius = rho
                gov_dgF_circle.center = xg
                gov_dgF_circle.radius = dgO
            else:
                # update ellipse center, height, width, angle
                width, height, angle = LEE_specs_path[tidx]
                LEE.center = xg
                LEE.height = height
                LEE.width = width
                LEE.angle = angle

                width, height, angle = GDE_specs_path[tidx]
                GDE.center = xg
                GDE.height = height
                GDE.width = width
                GDE.angle = angle

            # update common circle center
            gov_pos.center = xg
            robot_pos.center = xr
            lpg_pos.center = xg_bar

            # update arrows
            ax.patches.remove(v1_arrow)
            v1_dx, v1_dy = (xg - xr) * arrow_scale
            v1_arrow = mp.Arrow(xg[0], xg[1], v1_dx, v1_dy, color='green', width=arrow_size)
            ax.add_patch(v1_arrow)

            ax.patches.remove(v2_arrow)
            v2_dx, v2_dy = (xg_bar - xg) * arrow_scale
            v2_arrow = mp.Arrow(xg[0], xg[1], v2_dx, v2_dy, color='blue', width=arrow_size)
            ax.add_patch(v2_arrow)

            ax.patches.remove(v3_arrow)
            v4_dx, v4_dy = xv
            v3_arrow = mp.Arrow(xr[0], xr[1], v4_dx, v4_dy,color='cyan', width=arrow_size)
            ax.add_patch(v3_arrow)

            # add perpendicular velocity profile
            if tidx % 3 == 0:
                stx, sty = xr_path[tidx]
                xv = xv_path[tidx]
                dx, dy = -xv[1], xv[0]
                vp_arrow = mp.Arrow(stx, sty, -dx, -dy, color='m', width=0.1)
                ax.add_patch(vp_arrow)

            # update text
            time_stamp = time_now
            status_text1.set_text("Iter %d | time %.2f sec " % (tidx, time_stamp))

            # save figure
            if (save_fig is True) and (tidx % fig_sample_rate == 0):
                fig.canvas.flush_events()
                save_fig_to_folder(fig, fig_folder, str(1000+tidx))
            else:
                fig.canvas.flush_events()
                time.sleep(dt_play)
            tidx = tidx + 1
        print("Video finished")
        return 0

    def extract_frame(self, frame_id, save_fig=True, sim_name='sparse_'):
        """ Extract one frame from the simulation video
        """
        if save_fig is True:
            fig_folder = '.'
            print('save figure to <%s> ' % (fig_folder))

        plt.ion()
        fig, ax = plt.subplots()
        plt.tight_layout()
        energy_metric = self.energy_metric
        start_pt = self.map.start_pt
        goal_pt = self.map.goal_pt

        # create circle patch to draw
        if len(self.map.obstacles) > 0:
            patch_circles = GovLogViewer.create_circle_patch(self)
            ax.add_collection(patch_circles)

        # plot start and goal
        ax.plot(start_pt[0], start_pt[1], 'r*', markersize=12)
        ax.plot(goal_pt[0], goal_pt[1], 'g*', markersize=12)

        # plot navigation path
        patch_nav_path, verts_nav_path \
            = GovLogViewer.create_path_patch(self, 'red')
        ax.add_patch(patch_nav_path)
        # plot vertes of path
        xs, ys = zip(*verts_nav_path)
        ax.plot(xs, ys, 'o-', lw=2, color='black', ms=5, label='nav path')

        GovLogViewer.set_canvas(self, ax)
        # get back information of particluar frame
        xvec = self.xvec_log[frame_id]
        dvec = self.dvec_log[frame_id]
        evec = self.Evec_log[frame_id]

        lpg = self.lpg_log[frame_id]
        rho = self.le_rho_log[frame_id]

        xr, _, xg = xvec[0:2], xvec[2:4], xvec[4:6]
        dgF = dvec[0]

        st = start_pt
        stx, sty = xg
        ball_size, _, _ = 5, 2, 1

        time_now = frame_id * 0.05
        print('Frame %d info:' % frame_id)
        print('[ITER %3d | %.2f sec] xvec = %s' % (frame_id, time_now, flist1D(xvec)))
        print('[dgF, drg, drF] = [%5.2f, %5.2f, %5.2f]' % (dvec[0], dvec[1], dvec[2]))

        print('[deltaE, e_plus, e_rgs, et, ev] = [%.2f, %.2f, %.2f, %.2f, %.2f]\n'
              % (evec[0], evec[1], evec[2], evec[3], evec[4]))

        gov_pos = mp.Circle(xg, ball_size, fc='blue')
        robot_pos = mp.Circle(xr, ball_size, fc='green')
        lpg_pos = mp.Circle(lpg, ball_size/2, fc='red')
        # add common patches
        ax.add_patch(gov_pos)
        ax.add_patch(robot_pos)
        ax.add_patch(lpg_pos)

        if energy_metric == 'ellipse':
            LEE_specs = self.LEE_specs_log[frame_id]
            GDE_specs = self.GDE_specs_log[frame_id]

            width, height, angle = LEE_specs
            # local energy ellipse
            LEE = Ellipse((xg[0], xg[1]), width, height,
                          angle, color='y', alpha=0.4)

            width, height, angle = GDE_specs
            # governor dgF ellipse
            GDE = Ellipse((xg[0], xg[1]), width, height, angle, color='grey', alpha=0.4)
            ax.add_patch(LEE)
            ax.add_patch(GDE)
            frame_info = [xvec, dvec, evec, lpg, rho, LEE_specs, GDE_specs]
        else:
            local_energy_circle = mp.Circle(st, rho, fc='y', alpha=0.5)
            gov_dgF_circle = mp.Circle(st, ball_size, fc='tab:grey', alpha=0.3)
            local_energy_circle.center = xg
            local_energy_circle.radius = rho
            gov_dgF_circle.center = xg
            gov_dgF_circle.radius = dgF

            ax.add_patch(local_energy_circle)
            ax.add_patch(gov_dgF_circle)
            frame_info = [xvec, dvec, evec, lpg, rho]

        if self._frame_annotation:
            text_msg = 'Iter %d | time %.2f sec' % (frame_id, time_now)
            status_text1 = ax.text(10, 210, text_msg, fontsize=16)
            status_text1.set_text(text_msg)

            text_msg2 = 'dist(g, O) =  %.2f' % (dgF)
            status_text1 = ax.text(10, 190, text_msg2, fontsize=16)
            status_text1.set_text(text_msg2)

        ax.tick_params(axis='both', which='both', length=0)

        if save_fig is True:
            frame_name = sim_name + energy_metric + str(frame_id)
            save_fig_to_folder(fig, '../sim_figs/summary', frame_name)
            figname_wpath = os.path.join(fig_folder, frame_name)
            fig.savefig(figname_wpath, dpi=300, bbox_inches='tight')

        return frame_info

    def extract_frame_new(self, frame_id):
        """extract info at certan frame dt = 0.05 sec """
        xvec = self.xvec_log[frame_id]  # (6,)
        # tiny red ball
        lpg = self.lpg_log[frame_id] #(2,)
        # blue solid line
        nav_path = np.array(self.nav_path_log[frame_id]) # 2D array (Num_pts * 2)
        # black grid
        occgrid = self.occgrid_log[frame_id] # 2D array bool type
        # red pts
        lidar_pts = self.lidar_pt_log[frame_id] # 2D array (Num_pts * 2)

        # 3D flat ellise
        LEE_specs = self.LEE_specs_log[frame_id] #list [width, height, angle]
        GDE_specs = self.GDE_specs_log[frame_id] #list [width, height, angle]

        xr_log_mat = self.xvec_log[0:frame_id+1,0:2]
        return xvec, lpg, nav_path, lidar_pts, occgrid, LEE_specs, GDE_specs, xr_log_mat

    def save_frames_dense(self, folder='sim_figs/dense_env2D'):
        """ save frame figures for dense env simulation top view.
        """
        mesh_dense = trimesh.load_mesh(self.mesh_path)
        rgs_env = RgsEnv(mesh_dense, self.start_pt, self.goal_pt)
        fig1, ax1 = rgs_env.show_mesh(mesh_dense)
        print('uploading log data to rgs_env...')
        rgs_env.occgrid_log = self.occgrid_log
        rgs_env.lidar_endpts_log = self.lidar_pt_log
        rgs_env.rbt_loc_log = self.rbt_loc_log
        rgs_env.path_log = self.nav_path_log
        print('rgs_env init finished!')
        Nframe = len(self.occgrid_log)
        for fidx in range(Nframe):
            time_now = fidx * self.dt
            text_msg = 'Iter %03d | Time  %5.2f sec' % (fidx, time_now)
            print(text_msg)

            fig2, ax21, ax22, new_lidar_pts, old_occgrid, \
            new_grid, imap2D, nav_path_arr, rbt_loc \
                = rgs_env.map_vis_viewer(fidx,
                                         rbt_vis=False,
                                         lidar_vis_mode=0)
            xvec = self.xvec_log[fidx]
            xr, _, xg = xvec[0:2], xvec[2:4], xvec[4:6]

            lpg = self.lpg_log[fidx]
            LEE_specs = self.LEE_specs_log[fidx]
            GDE_specs = self.GDE_specs_log[fidx]
            width, height, angle = LEE_specs

            # local energy ellipse
            LEE = Ellipse((xg[0], xg[1]), width, height, angle, color='y')
            width, height, angle = GDE_specs

            # governor dgF ellipse
            GDE = Ellipse((xg[0], xg[1]), width, height, angle, color='grey', alpha=0.3)

            ball_size = 0.16
            rbt_pos = mp.Circle(xr, ball_size, fc='green')
            gov_pos = mp.Circle(xg, ball_size, fc='blue')
            lpg_pos = mp.Circle(lpg, ball_size/2, fc='red')

            ax21.add_patch(GDE)
            ax21.add_patch(LEE)

            ax21.add_patch(rbt_pos)
            ax21.add_patch(gov_pos)
            ax21.add_patch(lpg_pos)

            status_text1 = ax21.text(13.5, 4.7, text_msg, fontsize=12)
            status_text1.set_text(text_msg)
            frame_name = 'LMOG_' + str(10000+fidx)
            save_fig_to_folder(fig2, folder, frame_name, dpi=300)

#%% uncomment corresonding lines to view result you want to see
# The figures can be directly viewed when Spyder IDE is used
if __name__ == '__main__':
    # log_circular = '../log/sparse_ballPd_diag_1_1.pkl'
    log_circular = '../log/sparse_ellipsePd_diag_1_4.pkl'
    print('input file is %s' %(log_circular))
    viewer = GovLogViewer(log_circular)
    viewer.show_trajectory()
    viewer.compare_eta_max()
    # viewer.show_robot_gov_stat()
    # speed depends on hardware.
    # viewer.play_animation(save_fig=True, fig_sample_rate=10)
    print('Result generated.')
#    pressQ_to_exist()
# %% create folder if it does not exist



