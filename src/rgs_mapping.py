#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perception and Mapping Class for RGS.

Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""

# python built in package
import numpy as np
from numpy.linalg import norm
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# third-party
import trimesh
# personal
import mylib.astar_py as pyAstar
import mylib.gridutils_py as pyGridUtils
from my_utils import tic, toc, debug_print

mplt.rcParams['pdf.fonttype'] = 42
mplt.rcParams['ps.fonttype'] = 42


class LaserSimError(Exception):
    """ User Defined Exceptions for Laser Simulation.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "LaserSimError exception: {0}".format(self.msg)
        else:
            return "LaserSimError exception"


class AstarPlanningError(Exception):
    """ User Defined Exceptions.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "AstarPlanningError exception: {0}".format(self.msg)
        else:
            return "AstarPlanningError exception"


class UpdateErr(Exception):
    """ User Defined Exception for env update.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "UpdateErr exception: {0}".format(self.msg)
        else:
            return "UpdateErr exception"


class LaserSim:
    """ Class of LIDAR Simulaiton. 3D -> 2D
    """

    def __init__(self, min_hang=-np.pi, max_hang=np.pi,
                 min_vang=-np.pi/12, max_vang=np.pi/12,
                 hcount=180, vcount=3,
                 min_rng=0.05, max_rng=30.0, noise_sd=0.0):
        """ Init LIDAR Specs, and cooridinate tranformation constant matrix
        for all elevation and azimuth.
        """
        # self.angle_limits \
        #            = np.array([min_hang, max_hang, min_vang, max_vang])
        # self.angle_counts = np.array([hcount,vcount])
        azimuth = min_hang + \
            np.arange(hcount)*(max_hang - min_hang)/(hcount)
        elevation = min_vang + \
            np.arange(vcount)*(max_vang - min_vang)/(vcount)
        ce = np.cos(elevation[:, None])
        se = np.tile(np.sin(elevation), (hcount, 1)).T
        # spherical to cartensian
        # x = r cos(elevation) cos(azimuth);
        # y = r cos(elevation) sin(azimuth)
        # z = r sin(elevation)
        self._vcount = vcount
        self._hcount = hcount
        self._azimuth = azimuth
        self._elevation = elevation
        self._min_rng = min_rng
        self._max_rng = max_rng

        self._frame = np.stack(
            [ce * np.cos(azimuth), ce * np.sin(azimuth), se], axis=2)
        self.range_limits = np.array([min_rng, max_rng])
        self._noise_sd = noise_sd
        self.dist_vec = []  # dvec = np.array([dgF, drg, drF])

    def get_range_scan3D(self, R, p, mesh):
        """Get 3D scan points rho of all rays, given LIDAR pose(R, p) and
        enviroment (mesh). Intersections between rays and mesh environment
        are obtained using trimesh built-in methods.
        """
        # rho = np.zeros(np.prod(self.angle_counts))
        # R is (3,3) frame_aug = (hcount,vcount,3,1)
        # broadcasting rule is R @ last two dimension (3,1)
        # res is (hcount * vcount, 3) each row is a laser scan point
        ray_directions = (R @ self._frame[..., None]).reshape((-1, 3))
        ray_origins = np.tile(p, (ray_directions.shape[0], 1))
        # these may be used to obtain color information too!
        # using mesh sometimes can be 0-volumne error
        # lidar scan can penetrate some place causing lidar range err!!!
        locations, index_ray, _ \
            = mesh.ray.intersects_location(ray_origins=ray_origins,
                                           ray_directions=ray_directions,
                                           multiple_hits=False)
        rho = np.inf*np.ones(ray_directions.shape[0])
        rho[index_ray] = np.sqrt(
            np.sum((locations - ray_origins[index_ray, :])**2, axis=1))
        rho = rho.reshape((self._frame.shape[0], self._frame.shape[1]))
        rho = rho[::-1, :]  # reverse row index for lower left origin
        return rho

    def get_range_2D(self, R, p, mesh):
        """ Extract 2D plane (z=0) by extracting 3D data
        """
        p3D = np.hstack((p, 0))
        rho_3D = LaserSim.get_range_scan3D(self, R, p3D, mesh)
        rho_zero = rho_3D[int(self._vcount/2)]
        return rho_zero

    def get_lidar_endpts(self, meshbounds, rho, p, dim=2, show_fig=False, eps=0.25):
        """ Get lidar scan end points coordinates in 2D/3D
        """
        if dim == 2:
            idx_LMR = rho < self._max_rng  # index less than maximum range
            # polar to cartesian 2D
            x = np.cos(self._azimuth[idx_LMR]) * rho[idx_LMR] + p[0]
            y = np.sin(self._azimuth[idx_LMR]) * rho[idx_LMR] + p[1]

            lidar_endpts = np.vstack((x, y)).T  # Num_pts * dim
            xmin, ymin = np.min(lidar_endpts, axis=0)
            xmax, ymax = np.max(lidar_endpts, axis=0)
            xL, xH = meshbounds[:, 0]
            yL, yH = meshbounds[:, 1]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 0] >= xL - eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 0] <= xH + eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 1] >= yL - eps]
            lidar_endpts = lidar_endpts[lidar_endpts[:, 1] <= yH + eps]

            if show_fig is True:
                fig, ax = plt.subplots()
                ax.plot(p[0], p[1], 'gs')
                ax.plot(x, y, 'r*')
                ax.set_aspect('equal')
                ax.grid()
                return lidar_endpts, fig, ax
            else:
                return lidar_endpts, None, None

        else:
            raise LaserSimError("Haven't Implemented 3D Yet!")


class RgsEnv:
    """ Class of environment for Robot-Governro System simulation, taken
    mesh file as enviroment, and transfer it to occgrid. Map updates from
    LIDAR sensing data.
    """

    def __init__(self, mesh, start, goal, vox_res=0.25,
                 dim=2, inflation_radius=0.25):
        """ Init Class
        Input
        1. mesh: a 3D mesh (.stl)
        2. start
        3. goal
        4. vox_res: vox discretization size
        5. dim = 2 # planning dimension
        """
        self.start = start
        self.goal = goal
        self._vox_res = vox_res  # same res for all dimension
        self._dim = dim
        self._mesh_origin = mesh.bounds[0][0:dim]
        self._meshbounds = mesh.bounds

        self._xlimit = mesh.bounds[1, 0] - mesh.bounds[0, 0]
        self._ylimit = mesh.bounds[1, 1] - mesh.bounds[0, 1]
        self._zlimit = mesh.bounds[1, 2] - mesh.bounds[0, 2]
        self._grid_xlimit = self._xlimit / vox_res
        self._grid_ylimit = self._ylimit / vox_res
        self._grid_zlimit = self._zlimit / vox_res

        self._inflation_radius = np.ones(dim) * inflation_radius
        self.rbt_loc = start
        self.map_tidx = 0

        # voxelize the mesh
        t1 = tic()
        vox = mesh.voxelized(pitch=vox_res)
        toc(t1, "Voxelization")

        self._voxshape = vox.shape
        self._voxscale = vox.scale
        # occpuancy grid in 2D
        self.occgrid2D = np.zeros(self._voxshape[0:dim], dtype=bool)
        self.rbt_loc_cell = RgsEnv.meter2cell(self, self.rbt_loc)
        self.start_loc_cell = RgsEnv.meter2cell(self, start)
        self.goal_loc_cell = RgsEnv.meter2cell(self, goal)
        self.planning_start_cell = None
        self.planning_start_cell_bk = self.start_loc_cell

        # create log containers
        self.lidar_endpts = None
        self.lidar_endpts_log = []
        self.occgrid_log = []
        self.imap2D_log = None
        self.rbt_loc_log = []

        self.path = []
        self.path_log = []
        self.imap2D = []
        self.nav_path_world = None
        self.goal_reached_flag = False
        print('rgs env init finished')

    def reset_log(self):
        """ Reset object to empty log
        """
        self.rbt_loc = self.start
        self.map_tidx = 0
        self.occgrid2D = np.zeros(self._voxshape[0:self._dim], dtype=bool)
        self.rbt_loc_cell = RgsEnv.meter2cell(self, self.rbt_loc)
        self.start_loc_cell = RgsEnv.meter2cell(self, start)
        self.goal_loc_cell = RgsEnv.meter2cell(self, goal)
        self.planning_start_cell = None

        self.lidar_endpts = None
        self.lidar_endpts_log = []
        self.occgrid_log = []
        self.imap2D_log = None
        self.rbt_loc_log = []

        # nav_path = RgsEnv.update_map2D(ls, start)
        self.path = []
        self.path_log = []
        self.imap2D = self.occgrid2D
        self.nav_path_world = None
        self.goal_reached_flag = False
        print('rgs reset')

    @staticmethod
    def axis_equal_3D(this_ax):
        """ set aspect ratio to equal for 3D
        https://github.com/WISDEM/CommonSE/blob/master/src/commonse/axisEqual3D.py
        """
        print('set aspect ratio to equal for 3D')
        extents = np.array([getattr(this_ax, 'get_{}lim'.format(dim))()
                            for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(this_ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    @staticmethod
    def show_mesh(mesh, azim_angle=-89, elev_angle=75):
        """ Show ground truth environment mesh
        """
        fig_mesh = plt.figure(figsize=plt.figaspect(0.4))
        ax_mesh = fig_mesh.add_subplot(1, 1, 1, projection='3d')
        ax_mesh.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces]))
        # for better visualization
        ax_mesh.view_init(azim=azim_angle, elev=elev_angle)
        RgsEnv.axis_equal_3D(ax_mesh)
        ax_mesh.set_xlim(mesh.bounds[0, 0], mesh.bounds[1, 0])  # xlimit
        ax_mesh.set_ylim(mesh.bounds[0, 1], mesh.bounds[1, 1])  # ylimit
        ax_mesh.set_zlim(mesh.bounds[0, 2], mesh.bounds[1, 2])  # zlimit
        plt.title('mesh visualization')
        plt.show()
        return fig_mesh, ax_mesh

    def meter2cell(self, loc):
        """ Convert environment from meter to cell
        Input:
            loc:     nd vector containing loc coordinates in world frame (in meters)
            grid_min: nd vector containing the lower left corner of the grid (in meters)\n"
            grid_res: nd vector containing the grid resolution (in meters)\n"
        Output:
            loc_cell: nd vector containg index of cell grid index goes
            from lower left to upper right
        """
        grid_min = self._mesh_origin
        grid_res = self._vox_res
        diff = (loc - grid_min)/grid_res
        loc_cell = diff.astype(int)
        # print('loc at %s, grid_min %s, grid_res %s' \
        #             %(flist1D(loc), flist1D(grid_min), flist1D(grid_res)))
        return loc_cell

    def cell2meter(self, cell_loc):
        """
        Input:
            cell_loc: nd vector containing the cell index (from lower left to upper right)
            mesh_origin: nd vector containing the cell origin (in meters)"
            grid_res: nd vector containing the grid resolution (in meters)\n"
        Output:
            loc: nd vector containing loc coordinates in world frame (in meters)
            from lower left to upper right
        """
        grid_res = self._vox_res
        loc = cell_loc * grid_res + self._mesh_origin

        # print('loc at %s, grid_min %s, grid_res %s' \
        #             %(flist1D(loc), flist1D(grid_min), flist1D(grid_res)))
        return loc

    def replanning(self, imap, planning_start_cell):
        """ Replanning to update navigation Path_future
        Planninng: Astar on updated inflated occupancy grid
        Here planning is fast for simplicity we do planning every iteration
        If planning computation is high, one should considering better
        strategy
        """
        self.planning_start_cell = planning_start_cell
        path = self.path
        if norm(planning_start_cell-self.goal_loc_cell) < self._vox_res:
            print('Goal Grid Reached! Planning Mission Completed!')
            self.goal_reached_flag = True
            new_path = path
        # when goal is not reached
        else:
            pcost, new_path = pyAstar.planOn2DGrid(imap,
                                                   planning_start_cell,
                                                   self.goal_loc_cell)
            if np.isfinite(pcost):
                # dmsg = 'Replanning... path_cost = %d' % pcost
                # debug_print(-1, dmsg)
                pass
            # handle exception
            else:
                _, ax3 = plt.subplots()
                RgsEnv.set_canvas_grid(self, ax3)
                ax3.matshow(imap.T, origin='lower')
                ax3.plot(rbt_loc_cell[0], rbt_loc_cell[1], 'r*', markersize=4)
                raise AstarPlanningError('No path found')

        return new_path

    def set_canvas_world(self, ax_world):
        """set canvas for world map"""
        ax_world.set_aspect('equal')
        ax_world.set_xlim(0, self._xlimit)
        ax_world.set_ylim(0, self._ylimit)
        start = self.start
        goal = self.goal
        ax_world.plot(start[0], start[1], 'r*', markersize=12)
        ax_world.plot(goal[0], goal[1], 'g*', markersize=12)
        plt.setp(ax_world.get_yticklabels(), visible=False)
        plt.setp(ax_world.get_xticklabels(), visible=False)
        ax_world.tick_params(axis='both', which='both', length=0)

    def set_canvas_grid(self, ax_plan):
        """set canvas for inflated occgrid map"""
        ax_plan.set_aspect('equal')
        ax_plan.set_xlim(0, self._grid_xlimit)
        ax_plan.set_ylim(0, self._grid_ylimit)
        goal_loc_cell = self.goal_loc_cell
        start_loc_cell = self.start_loc_cell
        ax_plan.plot(start_loc_cell[0], start_loc_cell[1], 'r*', markersize=12)
        ax_plan.plot(goal_loc_cell[0], goal_loc_cell[1], 'g*', markersize=12)
        plt.setp(ax_plan.get_xticklabels(), visible=False)
        plt.setp(ax_plan.get_yticklabels(), visible=False)
        ax_plan.tick_params(axis='both', which='both', length=0)

    def map_vis_init(self):
        """ Init plots for map visualization
        """
        fig2 = plt.figure(figsize=(10, 5))
        gs = GridSpec(2, 1)
        gs.update(hspace=0.05)
        ax21 = plt.subplot(gs[0])
        ax22 = plt.subplot(gs[1])
        return fig2, ax21, ax22

    @staticmethod
    def create_path_patch(path_array, lc):
        """Create nav_path patch using navigation global waypoints.
        path format Num_pts * dim
        """
        verts = path_array
        path_len = len(path_array)
        codes = [Path.MOVETO] + [Path.LINETO] * (path_len - 1)
        path = Path(verts, codes)
        patch_path = mp.PathPatch(path, facecolor='none', ec=lc, lw=1)
        return patch_path, verts

    @staticmethod
    def draw_path_2D(this_ax, traj, path_clr='cyan', lw=1):
        ''' h = draw_path_2D(h,traj)
            traj = num_traj x num_pts x num_dim
        '''
        if traj.ndim < 3:
            traj = traj[None, ...]

        if type(this_ax) is LineCollection:
            this_ax.set_verts(traj)
        else:
            this_handle = this_ax.add_collection(LineCollection(traj, colors=path_clr, linewidth=lw))
        return this_handle

    def map_vis_update(self, this_fig, ax_world, ax_plan):
        """ Refresh map with new data
        """
        lidar_endpts = self.lidar_endpts_log[-1]
        self.draw_path_2D(ax_plan, np.array(rgs_env.path))
        ax_world.plot(lidar_endpts[:, 0], lidar_endpts[:, 1],
                      'r*', markersize=0.1)
        ax_plan.matshow(rgs_env.occgrid2D.T, origin='lower')

    def dist_vec_Qnorm(self, xvec, Q):
        """
        Compute distances of :
            # dgF: governor to the cloest obstacle
            # drg: robot to governor
            # drF: robot to to the closet obstacle

        in Quadratic norm determined by P.D. matrix Q
        Input
            pt_arr: num_pts * dim (in meter)
            xg: governor loc (in meter)
            xr: robot loc (in meter)
            Q: pd matrix induced quadratic norm
        """
        xr, xg = xvec[0:2], xvec[4:6]
        pt_arr = self.lidar_endpts
        XgF = pt_arr - xg
        XrF = pt_arr - xr
        dgF_vec = np.sqrt(np.sum(XgF @ Q * XgF, axis=1))
        drF_vec = np.sqrt(np.sum(XrF @ Q * XrF, axis=1))
        dgF = np.min(dgF_vec)

        drF = np.min(drF_vec)
        drg = np.sqrt((xr - xg).T @ Q @ (xr - xg))
        dvec = np.array([dgF, drg, drF])
        return dvec

    def get_imap(self, grid):
        """ Get inflated map
        """
        map2D = grid
        imap2D = pyGridUtils.inflateMap(map2D.flatten(), 
                                        grid.shape, 
                                        self._voxscale[:self._dim], 
                                        True, 
                                        self._inflation_radius)
        imap2D = np.reshape(imap2D, (map2D.shape[0], map2D.shape[1]))
        return imap2D

    def get_new_obs_cell(self, R,  p, lidar_endpts):

        # find obstacle cell
        obs_cell_loc = RgsEnv.meter2cell(self, lidar_endpts)
        new_intersects_loc = np.zeros(self._voxshape[0:self._dim], dtype=bool)

        try:
            new_intersects_loc[(obs_cell_loc[:, 0], obs_cell_loc[:, 1])] = True
        except IndexError:
            print('obs_cell_loc shape %s' % (obs_cell_loc.shape))
            print('new_intersects_loc shape %s' % (new_intersects_loc.shape))

        return new_intersects_loc

    def check_pt_free(self, check_pt, imap2D, occgrid2D):
        """ Check whether check_pt is in free cell of inflated map.
        If cell loc is occupied, return fail (-1), cell_loc
        else return succ(0), cell_loc
        """
        check_res = -100
        cell_loc = RgsEnv.meter2cell(self, check_pt)
        if self.map_tidx > 0 and imap2D[cell_loc[0], cell_loc[1]] > 0:
            # print('WARNING! check_pt is on obstacle of imap')

            if occgrid2D[cell_loc[0], cell_loc[1]] > 0:
                print('Warning! check_pt is on obstacle of occgrid')
                print('check_pt [%.2f, %.2f] at cell [%d, %d]'
                      % (check_pt[0], check_pt[1],
                         cell_loc[0], cell_loc[1]))
                check_res = -2
            else:
                check_res = -1
        else:
            check_res = 0

        return check_res, cell_loc

    def update_map2D(self, mesh, ls, p, planning_start,
                     view_mode=False, lidar_hist=None,
                     occgrid_hist=None, info_level=-1):
        """
        Update world and occgrid map for new robot position (meter) using
        LIDAR scan at new pose replanning using planning_start (meter)
        """
        R = np.eye(3)
        nav_path_world = None
        old_grid = self.occgrid2D

        if view_mode is False:
            old_imap2D = self.imap2D
            rbt_status, rbt_cell_loc = RgsEnv.check_pt_free(
                self, p, old_imap2D, old_grid)
            if rbt_status < -1:
                # some numerical problem might happens
                print('WARNING rbt loc on old grid!')
            else:
                pass

            debug_print(info_level, 'get new lidar scan...')
            rho_2D = ls.get_range_2D(R, p, mesh)
            lidar_endpts, _, _ = ls.get_lidar_endpts(
                self._meshbounds, rho_2D, p, 2)
        else:
            debug_print(-1, 'Use lidar hist ...')
            lidar_endpts = lidar_hist

        new_intersects_loc = RgsEnv.get_new_obs_cell(self, R, p, lidar_endpts)
        debug_print(info_level, 'update occgrid...')
        # occgrid OR new intersects
        if view_mode is False:
            new_grid = np.bitwise_or(old_grid, new_intersects_loc)
            new_imap2D = RgsEnv.get_imap(self, new_grid)
            debug_print(info_level, 'update navigation path')
            ps_status, planning_start_cell \
                = RgsEnv.check_pt_free(self, planning_start, new_imap2D, new_grid)

            if ps_status < -1:
                print('WARNING gov cell loc on new_grid!')
            else:
                pass

            new_path = RgsEnv.replanning(self, new_imap2D, planning_start_cell)
            new_path_world = RgsEnv.cell2meter(self, np.array(new_path))
            # To prevent discretization error add governor position at
            # beginning of the path
            init_diff = new_path_world[0] - planning_start

            if norm(init_diff) > 1e-2:
                # print('dist(pS, xg) =  %.4f' %(norm(init_diff)))
                nav_path_world = np.vstack((planning_start, new_path_world))

            else:
                nav_path_world = new_path_world

            ns = nav_path_world[0]
            ns_status, ns_cell \
                = RgsEnv.check_pt_free(self, ns, new_imap2D, new_grid)
            if ns_status < -1:
                print('WARNING ns loc (%.2f, %.2f) [%d, %d] on new grid!'
                      % (ns[0], ns[1], ns_cell[0], ns_cell[1]))

        else:
            new_grid = np.bitwise_or(occgrid_hist, new_intersects_loc)
            imap2D = RgsEnv.get_imap(self, new_grid)
        # log data
        if view_mode is False:
            self.lidar_endpts = lidar_endpts
            self.lidar_endpts_log.append(lidar_endpts)
            self.occgrid2D = new_grid
            self.occgrid_log.append(new_grid)

            self.path = new_path
            self.path_log.append(new_path)
            self.imap2D = new_imap2D
            self.map_tidx += 1
            self.rbt_loc = p
            self.rbt_loc_cell = RgsEnv.meter2cell(self, p)
            self.rbt_loc_log.append(p)
            self.nav_path_world = nav_path_world
            return new_path, nav_path_world
        else:
            return imap2D, new_grid

    def map_vis_viewer(self, tidx, rbt_vis=True, info_level=-1,
                       lidar_vis_mode=0):
        """Retrive and plot map at certain time index"""
        # print('Visualize map at tidx = %d' %(tidx))
        fig2, ax21, ax22 = RgsEnv.map_vis_init(self)
        rbt_size = self._vox_res

        rbt_loc_log = self.rbt_loc_log
        rbt_loc = rbt_loc_log[tidx]
        # rbt_loc_cell = RgsEnv.meter2cell(self, rbt_loc)
        plannning_start_cell = self.path_log[tidx][0]
        # add robot vis
        if rbt_vis is True:
            rbt_patch21 = mp.Circle(rbt_loc,      rbt_size, fc='green')
            ax21.add_patch(rbt_patch21)

        ps_patch22 = mp.Circle(plannning_start_cell, rbt_size*2, fc='blue')
        ax22.add_patch(ps_patch22)

        # obtain information up to tidx
        if tidx > 0:
            old_lidar_pts_log = self.lidar_endpts_log[0:tidx-1]
            old_occgrid = self.occgrid_log[tidx-1]
        else:
            old_lidar_pts_log = []
            old_occgrid = np.zeros(self.occgrid_log[-1].shape,  dtype=bool)

        new_lidar_pts = self.lidar_endpts_log[tidx]

        # plot lidar pts old and new
        for lidar_pts in old_lidar_pts_log:
            ax21.plot(lidar_pts[:, 0], lidar_pts[:, 1], 'ks', markersize=0.2)

        # view lidar end pts as point cloud
        if lidar_vis_mode == 0:
            ax21.plot(new_lidar_pts[:, 0],
                      new_lidar_pts[:, 1], 'rs', markersize=2)
        # draw line collections
        else:
            Num_lines = lidar_pts.shape[0]
            start_pts = np.tile(rbt_loc, (Num_lines, 1))
            # print('lidar ray number %d' %(Num_lines))
            """
            Create list of line segments from x and y coordinates, in the correct
            format for LineCollection:
            an array of the form   numlines x (points per line) x 2 (x and y) array
            """
            tmp_mat = np.hstack((start_pts, lidar_pts))
            seg_lines = np.reshape(tmp_mat, (Num_lines, 2, 2))
            lines_cL = LineCollection(seg_lines, colors='r', linewidths=0.5)
            ax21.add_collection(lines_cL)

        # plot occgrid up to tidx
        # ax22.matshow(self.occgrid_log[tidx].T, origin='lower')
        imap2D, new_grid = RgsEnv.update_map2D(self, None, None, None, None,
                                               view_mode=True,
                                               lidar_hist=new_lidar_pts,
                                               occgrid_hist=old_occgrid)

        cmap_custom = ListedColormap(['w', 'grey', 'dimgray'])
        ax22.matshow(imap2D.T, origin='lower', cmap=cmap_custom)

        # plot path
        rbt_loc_arr = np.array(rbt_loc_log[:tidx+1])
        xr_coord = rbt_loc_arr[:, 0]
        yr_coord = rbt_loc_arr[:, 1]
        ax21.plot(xr_coord, yr_coord, '*:', lw=2, color='green',
                  ms=2, label='rbt  path')

        nav_path_arr = np.array(self.path_log[tidx])
        nav_path_arr_loc = RgsEnv.cell2meter(self, nav_path_arr)
        # RgsEnv.draw_path_2D(ax21, nav_path_arr_loc)

        # plot remaining navigation path
        patch_nav_path, verts_nav_path \
            = RgsEnv.create_path_patch(nav_path_arr_loc, 'blue')
        ax21.add_patch(patch_nav_path)
        # plot vertes of path
        xs, ys = zip(*verts_nav_path)
        # ax21.plot(xs, ys, 'o--', lw=2, color='cyan', ms=2, label='nav path')
        ax21.plot(xs, ys, lw=1, color='blue', label='nav path')
        RgsEnv.draw_path_2D(ax22, nav_path_arr, path_clr='blue')

        RgsEnv.set_canvas_world(self, ax21)
        ax21.legend(loc='lower right')
        RgsEnv.set_canvas_grid(self, ax22)
        return fig2, ax21, ax22, new_lidar_pts, old_occgrid, new_grid, imap2D, nav_path_arr, rbt_loc


# %% Modlue Test for 2D Top-view visualization
if __name__ == '__main__':
    mesh_JZ = trimesh.load_mesh('../mesh/dense_env.stl')
    ls = LaserSim()
    start = np.array([1, 0.75])
    goal = np.array([18, 4])
    vox_res = 0.25
    rgs_env = RgsEnv(mesh_JZ, start, goal, vox_res)
    # show env mesh
    fig1, ax1 = rgs_env.show_mesh(mesh_JZ)
    # update map using LIDAR scan
    plannning_start = start
    nav_path, _ = rgs_env.update_map2D(mesh_JZ, ls, start, plannning_start)
    rgs_env.map_vis_viewer(0, rbt_vis=True)
    xvec_fake = np.hstack((start, start, start))
    dist_vec0 = rgs_env.dist_vec_Qnorm(xvec_fake, np.eye(2))


# %% Move on grid test without dynamics (one time a jump along navigation path)
    loop_cnt = 2000
    rgs_env.reset_log()
    while loop_cnt > 0 and len(nav_path) > 1 and rgs_env.goal_reached_flag is False:
        rbt_loc = rgs_env.rbt_loc
        rbt_loc_cell = rgs_env.rbt_loc_cell
        rbt_loc_new = rgs_env.cell2meter(np.array(nav_path[1]))

        print('Iter [%3d] |[%d, %d] (%.2f %.2f)--> [%d, %d] (%.2f, %.2f)'
              % (rgs_env.map_tidx,
                 rbt_loc_cell[0], rbt_loc_cell[1], rbt_loc[0],      rbt_loc[1],
                 nav_path[1][0],  nav_path[1][1],  rbt_loc_new[0],  rbt_loc_new[1]))

        plannning_start = rbt_loc_new
        nav_path, _ = rgs_env.update_map2D(
            mesh_JZ, ls, rbt_loc_new, plannning_start)
        loop_cnt -= 1


# %% check intermediate result at specified time index
    rgs_env.map_vis_viewer(20, rbt_vis=True)
    rgs_env.map_vis_viewer(40, rbt_vis=True)
    rgs_env.map_vis_viewer(rgs_env.map_tidx-1, rbt_vis=True)
    fig1, ax1 = rgs_env.show_mesh(mesh_JZ, 90, 75)

# %%
    rgs_env.map_vis_viewer(20, rbt_vis=True, lidar_vis_mode=1)
    rgs_env.map_vis_viewer(40, rbt_vis=True, lidar_vis_mode=1)
    rgs_env.map_vis_viewer(rgs_env.map_tidx-1, rbt_vis=True, lidar_vis_mode=1)
    fig1, ax1 = rgs_env.show_mesh(mesh_JZ, 90, 75)
