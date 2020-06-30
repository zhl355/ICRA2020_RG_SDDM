#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Governor System Class for sparse known env.
Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import numpy as np
from numpy.linalg import norm
from numpy import linalg as LA
# personal
from gov_ellipse import MyEllipse
from my_utils import debug_print, Pnorm_len, flist1D
import opt_solver as opt
# float precision
np.set_printoptions(precision=4)


class GovError(Exception):
    """ User Defined Exceptions for RG.
    """

    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''

    def __str__(self):
        if self.msg:
            return "GovError exception: {0}".format(self.msg)
        else:
            return "GovError exception"


class RbtGovSys:
    """
    A class for robot-governor system (RGS) update in simulation.
    It consists of three big parts
        1. RGS
            # xvec = [xr, xv, xg]
                - xr is the location of the robot
                - xg is the postion of governor
                - xv is the speed of the robot

            # Controller Design Paramters kv, kg, zeta
                - xr_dot = xv
                - xv_dot = -2 kv (xr-xg) - zeta xv
                - xg_dot = - kg (xg - xg_bar)

            # Numerical computation varibles:
                - eps: numerical machine eps
                - dt : discretization time interval

            # dvec = [dgO, drg, drO]
                - dgO/dgF: governor to the cloest obstacle
                - drg: robot to governor
                - drO/drF: robot to to the closet obstacle

    """
    # Running status table
    GOV_HALT = 0
    GOV_NORMAL = 1
    GOV_GOAL_REACHED = 100
    GOV_GOAL_REACHED_FLAG = False
    GOV_NO_LOCAL_GOAL = -1
    ERROR_NEGATIVE_deltaE = -2

    def __init__(self, rgs_map, xvec0, eta_max0, energy_metric,
                 Pd, PV0, P_kinematic, design_paras,
                 dt=0.05, eps=1e-6):
        """ Init RGS system.
        """

        self.map = rgs_map
        self.xvec = xvec0
        self.eta_max = eta_max0
        self.energy_metric = energy_metric

        self.Pd = Pd
        self.PV = PV0                   # potential energy norm
        self.P_kinematic = P_kinematic  # kinematic

        self.design_paras = design_paras
        self.kv, self.kg, self.zeta = design_paras
        self.clock = 0

        self.dt = dt
        self.eps = eps

        self.start_pt = self.map.start_pt
        self.goal_pt = self.map.goal_pt
        self.gov_status = RbtGovSys.GOV_HALT

        # eigenvalue of P_energy
        _, eig_PV, _ = LA.svd(self.PV)
        self.start_pt = xvec0[0:2]
        # initialize distance vector
        dvec0, _, _, _ = RbtGovSys.dist_vec(self)
        if sum(dvec0 >= 0) < 3:
            dist_err_str = 'Init Error' + str(flist1D(dvec0))
            raise GovError(dist_err_str)


        self.gov_status = RbtGovSys.GOV_NORMAL
        # history log container
        self.xvec_log = [xvec0]
        self.dvec_log = []
        self.deltaE_log = []
        self.lpg_log = []
        self.le_rho_log = []
        self.nav_adv_idx_log = []
        self.Evec_log = []
        self.LEE_specs_log = []
        self.GDE_specs_log = []


    def dist_pt2wall(self, pt):
        """ Find distance from pt to all wall
        """
        debug_level = -100
        if self.energy_metric == 'ball':
            # find closet point in Obs Space in 2-norm
            xl, xh, yl, yh = self.map.boundary
            x, y = pt
            dist_walls = np.abs([x-xl, x - xh, y - yl, y - yh])
            mindist_idx = np.argmin(dist_walls)
            dist_wall = dist_walls[mindist_idx]
            pstar_candidates = np.array([[xl, y], [xh, y], [x, yl], [x, yh]])
            # get back the closet pt on the wall
            pstar = pstar_candidates[mindist_idx]
        else:
            # find closet point in Obs Space in Quadratic-norm
            wall_ls = self.map.wall_ls
            norm_mat = self.PV  # NEL should be rotated
            dist_wall, pstar \
                = opt.dist_pt2seglist_Pnorm(pt, wall_ls, norm_mat)

        debug_print(debug_level, 'dist_pt2wall')
        dmsg = 'Get pstar %s and dist_wall %.2f from pt %s' \
            % (pstar, dist_wall, pt)
        debug_print(debug_level, dmsg)
        return dist_wall, pstar

    def dist_pt2obs(self, pt, add_noise=True):
        """ Find distance from pt to all obstacles
        for here just to the a collection of circle obstacles
        """
        debug_level = -1
        dist2obs_vec = []
        obs = self.map.obstacles.copy()

        if len(obs) > 0:
            if self.energy_metric == 'ball':
                dist2obs_vec = norm(obs[:, :2] - pt, axis=1) - obs[:, -1]
                mindist_idx = np.argmin(dist2obs_vec)
                pstar = obs[mindist_idx, :2]
                dist_obs = dist2obs_vec[mindist_idx]

                dmsg1 = 'DEBUG dist_pt2obs %s ' % (dist2obs_vec)
                dmsg2 = 'get pstar %s and dist_obs = %.2f from pt %s' \
                    % (pstar, dist_obs, pt)
                debug_print(debug_level, dmsg1)
                debug_print(debug_level, dmsg2)

            else:
                norm_mat = self.PV  # NEL should be rotated
                dist_obs, pstar \
                    = opt.dist_pt2circle_arr_Pnorm(pt, obs, norm_mat)
        else:
            dist_obs = np.inf
            pstar = []

            dmsg1 = 'DEBUG dist_pt2obs %s ' % (dist2obs_vec)
            dmsg2 = 'get pstar %s and dist_obs = %.2f from pt %s' \
                % (pstar, dist_obs, pt)
            debug_print(debug_level, dmsg1)
            debug_print(debug_level, dmsg2)
        return dist_obs, pstar

    def dist_pt2F(self, pt, msg):
        """
        Find distance from pt to free space
        for here we simply takv the min of the dist 2 env and obs.
        """
        debug_level = -100
        dmsg = 'dist_pt2F' + msg + str(pt)
        debug_print(debug_level, dmsg)
        dist_F = 0
        d2wall, pstar_wall = RbtGovSys.dist_pt2wall(self, pt)
        d2obs,  pstar_obs = RbtGovSys.dist_pt2obs(self, pt)

        if d2wall < 0:
            print('d2wall is %.2f from pt %s' % (d2wall, pt))
            raise GovError('Crush to wall')
        elif d2obs < 0:
            print('d2obs is %.2f from pt %s' % (d2obs, pt))
            print('Obs location')
            print(self.map.obstacles)
            raise GovError('Crush to obstacles')
        else:
            if d2obs <= d2wall:
                dist_F = d2obs
                pstar_F = pstar_obs
                msg = '!!! cloest pt at Xobs dist_F %.2f' % dist_F
                debug_print(debug_level, msg)
            else:
                dist_F = d2wall
                pstar_F = pstar_wall
                msg = '!!! cloest pt at wall dist_F %.2f' % dist_F
                debug_print(debug_level, msg)

        return dist_F, pstar_F

    def dist_vec(self):
        """
        Compute distances of :
            # dgO: governor to the cloest obstacle
            # drg: robot to governor
            # drO: robot to to the closet obstacle
        """
        debug_level = -100
        xr, xg = self.xvec[0:2], self.xvec[-2:]
        msg_xg = 'wrt <xg> '
        msg_xr = 'wrt <xr> '

        dgO, pt_gstar_O = RbtGovSys.dist_pt2F(self, xg, msg_xg)
        drO, pt_rstar_O = RbtGovSys.dist_pt2F(self, xr, msg_xr)

        if self.energy_metric == 'ball':
            drg = norm(xr - xg)
        else:
            drg = Pnorm_len(self.PV, xr-xg)
        dvec = np.array([dgO, drg, drO])
        self.dvec = dvec
        dmsg = 'dist_vec: [dgO, drg, drO]=[%.4f %.4f %.4f]' % (dgO, drg, drO)
        debug_print(debug_level, dmsg)
        return dvec, dgO, drg, drO

    def cpt_deltaE_ellipse(self):
        """
        Compute energy that can be safely added to the system.
        Energy mesured in ellipse metrics.
        """
        debug_level = -1
        PV, PT, xvec = self.PV, self.P_kinematic, self.xvec
        xr, xv, xg = xvec[0:2], xvec[2:4], xvec[4:6]
        s1 = xr - xg
        s2 = xv
        dvec, dgO, drg, drO = RbtGovSys.dist_vec(self)
        # compute energy composition
#        e_plus = 0.5 * (dgO**2)
#        ev = 0.5 * (s1.T @ PV @ s1)  # potential energy
#        et = 0.5 * (s2.T @ PT @ s2)  # kinematic energy

        e_plus = dgO**2
        ev = (s1.T @ PV @ s1)        # potential energy
        et = 0.5 * (s2.T @ PT @ s2)  # kinematic energy
        e_rgs = et + ev

        deltaE = e_plus - self.eta_max
        Evec = [deltaE, e_plus, e_rgs, et, ev]
        self.deltaE = deltaE
        self.Evec = Evec
        dmsg = 'cpt_deltaE_ellipse Evec \
                = [deltaE, e_plus, e_rgs, et, ev] is %s' % flist1D(Evec)
        debug_print(debug_level, dmsg)
        return Evec

    def cpt_deltaE_ball(self):
        """
        Compute energy that can be safely added to the system.
        Energy mesured in ball sense.
        """
        debug_level = -100
        debug_print(debug_level, 'cpt_deltaE_ball start')
        kv = self.kv
        xv = self.xvec[2:4]
        dvec, dgO, drg, drO = RbtGovSys.dist_vec(self)
        et = 0.5 * norm(xv)**2      # kinematic energy
        ev = kv * drg ** 2          # potential energy
        e_plus = kv * dgO ** 2
        e_rgs = et + ev
        deltaE = e_plus - e_rgs
        Evec = [deltaE, e_plus, e_rgs, et, ev]
        self.deltaE = deltaE
        self.Evec = Evec
        return Evec

    def find_proj_goal_ellipse(self, th=1):
        """
        Find local projected goal "xg_bar" using current state.

        Given current governor position, distance to free space
        and navigtaion path return the furthest possible pt
        along the Path (alpha*) in free space.

        # Key attributes from self
            xg      : center of ball  dim: 1 * D (governor position)
            deltaE  : from self, extra energy in ellipse-sense
                      PT can be safely added to the system.

        # Output
            STATUS  : status of this algorithm (GOAL_REACHED, NORMAL, FAIL...)
            xg_bar  : projected goal given current xg,
            nav_adv_idx : index of navagation path the system is heading to
            LEE_specs: local energy ellipse specifications (W, H, angle)
            GDE_specs: governor energy ellipse specifications (W, H, angle)

        """
        xg = self.xvec[-2:]
        goal_pt = self.goal_pt
        xg_bar = goal_pt
        nav_adv_idx = -1
        # governor reached the goal
        LEE_specs = []
        status = RbtGovSys.GOV_GOAL_REACHED
        """
        If not reached, using navigation path projection on ellipse
        as local projected goal.
        """
        deltaE, e_plus = self.Evec[0], self.Evec[1]
        if deltaE < 0:
            raise GovError('find_proj_goal_ellipse logic error')
        local_energy_ellipse = MyEllipse(self.PV, xg, 'e', deltaE)  # LEE
        dmsg = 'local_energy_ellipse energy form PV:' \
            + str(self.PV) + 'xg: ' + str(xg) +'deltaE:' + str(deltaE)
        debug_print(-1, dmsg)
        # not here we need to notice that GDE's energy
        # is 1/2 ||xg - pstar||_P^2
#        gov_dgF_ellipse = MyEllipse(self.PV, xg, 'e', 2 * e_plus) # GDE
        gov_dgF_ellipse = MyEllipse(self.PV, xg, 'e', e_plus)      # GDE

        width = local_energy_ellipse.width
        height = local_energy_ellipse.height
        angle_deg = np.rad2deg(local_energy_ellipse.angle)
        LEE_specs = [width, height, angle_deg]

        width = gov_dgF_ellipse.width
        height = gov_dgF_ellipse.height
        angle_deg = np.rad2deg(gov_dgF_ellipse.angle)
        GDE_specs = [width, height, angle_deg]

        if norm(xg - goal_pt) >= th:
            status, xg_bar, nav_adv_idx \
                = local_energy_ellipse.proj_2nav_path(self.map.nav_path)
            if status < RbtGovSys.GOV_HALT:
                raise GovError('CANNOT FIND LOCAL GOAL \
                               ALL SEGS ARE TOO FAR AWAY')
                xg_bar = self.lpg_log[-1]
                nav_adv_idx = self.nav_adv_idx_log[-1]
                LEE_specs = self.LEE_specs_log[-1]
                GDE_specs = self.GDE_specs_log[-1]
                status = RbtGovSys.GOV_NO_LOCAL_GOAL
            else:
                status = RbtGovSys.GOV_NORMAL

        return status, xg_bar, nav_adv_idx, LEE_specs, GDE_specs

    def find_proj_goal_ball(self, th=1):
        """
        Find local projected goal "xg_bar" using current state.

        Given current governor position, distance to free space
        and navigtaion path return the furthest possible pt
        along the Path (alpha*) in free space.

        # Key attributes from self
            xg      : center of ball  dim: 1 * D (governor position)
            deltaE  : from self, extra energy in ellipse-sense
                      that can be safely added to the system.
        # Output
            STATUS  : status of this algorithm (GOAL_REACHED, NORMAL, FAIL...)
            xg_bar  : local projected goal LPG
            nav_adv_idx : index of navagation path the system is heading to
        """
        xg = self.xvec[-2:]
        path = self.map.nav_path
        goal_pt = self.goal_pt
        r = np.sqrt(self.deltaE/self.kv)
        xg_bar = np.zeros(2)            # initialize goal pt
        path_len = len(path)            # check remained path length
        nav_adv_idx = -1

        # governor reached the goal
        if norm(xg - goal_pt) < th:
            return RbtGovSys.GOV_GOAL_REACHED, goal_pt, nav_adv_idx

        # PREVENT BUG WHEN gg.OBS DEGENERATE always turn it to 2D
        path = np.reshape(path, (-1, path.shape[-1]))

        # Alg. Navigation path with at least two vertices
        # ----------------------------------------------

        # iterative from path backward, extract two points iteratively
        # get segment AB

        # assume no repeat pts now
        # Compute the closest point on (A,B) segment that is in safe zone

        # u = X - A = AX
        # v = B - A = AB
        # up = u projection on v
        # up_perb = u - up
        # up = u'v / |v|^2 * v
        # w1  =  u'v / |v|^2
        # w1_hat = normalize(w1) to [0,1]
        # w2 = sqrt(r^2 - d^2) / |v|
        # ----------------------------------------------
        for path_idx in range(path_len):
            if path_idx == (path_len-1):
                #                raise GovError('CANNOT FIND LOCAL GOAL,\
                #                               ALL SEGS ARE TOO FAR AWAY')
                print('Warning CANNOT FIND LOCAL GOAL,\
                               ALL SEGS ARE TOO FAR AWAY')
                return RbtGovSys.GOV_HALT, xg, self.nav_adv_idx_log[-1]

            B = path[-path_idx-1]
            A = path[-path_idx-2]

            nav_adv_idx = -path_idx-1

            dAB = norm(A-B)  # segment length ptA from ptB
            u = xg - A
            v = B - A
            # print('path_idx is %d ----- A %s B %s' %(path_idx,A,B))
            # print('DEBUG segment A %s, B %s]' %(A,B))
            # print('DEBUG finding local goal at xg=[%s] \
            # rbar = %.4f with segment AB [%s,%s]' %(xg,r, A,B))

            w1 = np.inner(u, v) / dAB**2
            w1hat = max(min(w1, 1), 0)  # normalized to [0,1]
            dist2segAB = norm(u - w1hat*v)

            # by assumption this distance is always less than r for some AB
            # (i.e. local goal always exist)
            # print('DEBUG r = [%.2f] and dist2segAB is %.2f' %(r,dist2segAB))

            if dist2segAB > r:
                # print('segment AB is too far from x move to next segment')
                # print('dist2segAB is [%.2f] > [r = %2.f]' %(dist2segAB,r))
                continue
            else:
                # distance from x to line AB
                dist2lineAB = norm(u - w1 * v)
                # print('DEBUG dist to line AB is %.2f' %(dist2lineAB))
                w2 = np.sqrt(r**2 - dist2lineAB**2)/dAB  # ratio of |v|
                w = w1 + w2
                w = max(min(w, 1), 0)
                xg_bar = (1 - w) * A + w * B

                return RbtGovSys.GOV_NORMAL, xg_bar, nav_adv_idx

    def update_gov(self):
        """
        given current status of governor  find local projected goal xg_bar
        return the status of the main function and xg_bar

        # Input(from self)
            gov_status  : status of this govnor

        # Output(update self)
            gov_status          : new status of gov
            xg_bar              : local projected goal
            rB                  : local energy ball radius in ball way
            deltaE              : deltaE of the rgs
        """
        # get lastest gov_status, nav_adv_idx, xg_bar

        gov_status = self.gov_status
        if len(self.nav_adv_idx_log) > 0:
            nav_adv_idx = self.nav_adv_idx_log[-1]
#            xg_bar = self.lpg_log[-1]
        else:
            nav_adv_idx = 0
#            xg_bar = self.xvec[4:]

        rho = 0   # rho remains zero unless been changed in normal cases
        # case governor encounter error
        if gov_status < RbtGovSys.GOV_HALT:
            err_msg = 'gov_status %d' % gov_status
            raise GovError(err_msg)
        elif gov_status == RbtGovSys.GOV_HALT:
            print('Governor Halting')
        else:
            pass
        # case governor reached the goal
        if gov_status == RbtGovSys.GOV_GOAL_REACHED \
                and RbtGovSys.GOV_GOAL_REACHED_FLAG is False:
            RbtGovSys.GOV_GOAL_REACHED_FLAG = True
            print('----------- Governor Reached Goal -----------')
            gov_status = RbtGovSys.GOV_GOAL_REACHED

        """   Halt or normal running cases.
        """
        energy_metric = self.energy_metric

        if energy_metric == 'ball':
            Evec = RbtGovSys.cpt_deltaE_ball(self)
        else:
            Evec = RbtGovSys.cpt_deltaE_ellipse(self)

        deltaE, e_plus, e_rgs, et, ev = Evec
        self.deltaE = deltaE
        # halt case due to simulation discretization and numerical issue
        if deltaE <= 0:
            xvec = self.xvec
            xg = xvec[-2:]
            print('\nWARNING deltaE < 0: deltaE = %.4f' % (deltaE))
            print('[e_plus = %.4f e_rgs = %.4f, et = %.4f ev = %.4f]'
                  % (e_plus, e_rgs, et, ev))
            print('Current states xvec')
            print(xvec)
            gov_status = RbtGovSys.GOV_HALT
            xg_bar = xg

        # normal case find new xg_bar based on current configuration
            if energy_metric == 'ellipse':

                local_energy_ellipse = MyEllipse(self.PV, xg, 'e', 1e-3)  # LEE
                gov_dgF_ellipse = MyEllipse(self.PV, xg, 'e', e_plus)

                width = local_energy_ellipse.width
                height = local_energy_ellipse.height
                angle_deg = np.rad2deg(local_energy_ellipse.angle)
                LEE_specs = [width, height, angle_deg]

                width = gov_dgF_ellipse.width
                height = gov_dgF_ellipse.height
                angle_deg = np.rad2deg(gov_dgF_ellipse.angle)
                GDE_specs = [width, height, angle_deg]

                self.LEE_specs_log.append(LEE_specs)
                self.GDE_specs_log.append(GDE_specs)

        else:
            if energy_metric == 'ball':
                rB = np.sqrt(deltaE/self.kv)
                # find new xg_bar given current configuation and nav path
                gov_status, xg_bar, nav_adv_idx \
                    = RbtGovSys.find_proj_goal_ball(self)
                rho = rB
            else:
                # find new xg_bar given current configuation and nav path
                gov_status, xg_bar, nav_adv_idx, LEE_specs, GDE_specs \
                    = RbtGovSys.find_proj_goal_ellipse(self)
                xg = self.xvec[-2:]
                rho = norm(xg_bar - xg)
                self.LEE_specs_log.append(LEE_specs)
                self.GDE_specs_log.append(GDE_specs)

        # for all error free cases update relevant log and gov_status
        self.gov_status = gov_status
        self.deltaE_log.append(deltaE)
        self.le_rho_log.append(rho)
        self.lpg_log.append(xg_bar)
        self.nav_adv_idx_log.append(nav_adv_idx)
        self.Evec_log.append(Evec)
        """ UPDATE CLOCK
        """
        self.clock = self.clock + 1
        return gov_status, xg_bar
