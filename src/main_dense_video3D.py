#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creating and Displaying 3D visualization for dense env.
Author: zhichao li at UCSD ERL
Date: 06/26/2020
BSD 3-Clause License
https://github.com/zhl355/ICRA2020_RG_SDDM
"""
# python built in package
import os
import numpy as np
import matplotlib
# third party
import trimesh
import vtkplotter as vplt
# personals
from gov_log_viewer import GovLogViewer
from rgs_mapping import RgsEnv
from my_utils import tic, toc

# fonts and precision
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.set_printoptions(precision=4)

RBT_HEIGHT = -0.5
shadow_zval = -0.5
shadow_zshift = shadow_zval - RBT_HEIGHT
rbt_scaling = 2

def add_zdim(mat, z=RBT_HEIGHT, offset_z=0):
    """add thrid dimension to mat 2D array (Npts, 2) """
    Npts = mat.shape[0]
    zshift = z + offset_z
    zdim_vec = (np.ones(Npts) * zshift).reshape(Npts,1)
    mat3D = np.hstack((mat, zdim_vec))
    return mat3D

def expand_elements_dim(xvec, lpg, nav_path, occgrid, lidar_pts, xr_log_mat):
    """expand 2D elements to 3D"""
    xvec3D = add_zdim(xvec.reshape((3,2)), RBT_HEIGHT, shadow_zshift)
    xr3D, xv3D, xg3D = xvec3D[0], xvec3D[1], xvec3D[2]
    xv3D[-1] = 0
    lpg3D = np.hstack((lpg, RBT_HEIGHT+shadow_zshift))
    # turn nav_path from grid 2 meter
    nav_path_loc = rgs_env.cell2meter(nav_path)
    nav_path3D = add_zdim(nav_path_loc, RBT_HEIGHT, shadow_zshift)
    nav_path3D.tolist()
    lidar_pts3D = add_zdim(lidar_pts)
    rbt3D = xr3D - np.array([0, 0, shadow_zshift]) # rbt is at upper level
    lidar_pts3D = add_zdim(lidar_pts)
    rbt3D_trail = add_zdim(xr_log_mat, RBT_HEIGHT, shadow_zshift)

    return rbt3D, xr3D, xv3D, xg3D, lpg3D, nav_path3D, lidar_pts3D, rbt3D_trail

def get_theta_from_vel(xv2D):
    """ compute angle theta (rbt orientation) between X-axis and
    velocity vector """
    eta1, eta2 = xv2D
    theta_deg = np.rad2deg(np.arctan2(eta2, eta1))
    return theta_deg

def ellipse2Dto3D(center2D,
                  ellipse_2D_spec,
                  zaxis_semilen=0.1,
                  posz=RBT_HEIGHT+shadow_zshift,
                  ec='gray',
                  ea=0.5):
    """ convert ellipse 2D to fake 3D supported by vtkplotter
    """
    width, height, angle = ellipse_2D_spec
    vx = np.array([width, 0, 0])
    vy = np.array([0, height, 0])
    vz = np.array([0, 0, zaxis_semilen])
    center3D = np.hstack((center2D, posz))
    ellipse3D_actor = vplt.Ellipsoid(center3D,
                                     axis1 = vx, axis2 = vy, axis3 = vz,
                                     c=ec,
                                     alpha=ea)
    ellipse3D_actor.rotateZ(angle)

    return ellipse3D_actor

# %% 3D visualization using vtkplotter
if __name__ == '__main__':
    t1 = tic()
    """ Read and anlyaze log result
    """
    video_frame_fd = '../sim_figs/dense_env3D'
    log_name = '../log/dense_full_ellipse.pkl'
    viewer = GovLogViewer(log_name)
    start_pt, goal_pt = viewer.start_pt, viewer.goal_pt
    start3D = np.hstack((start_pt, RBT_HEIGHT))
    goal3D = np.hstack((goal_pt, RBT_HEIGHT))
    mesh_dense = trimesh.load_mesh(viewer.mesh_path)
    rgs_env = RgsEnv(mesh_dense, start_pt, goal_pt)
    print('uploading log data to rgs_env...')
    rgs_env.occgrid_log = viewer.occgrid_log
    rgs_env.lidar_endpts_log = viewer.lidar_pt_log
    rgs_env.rbt_loc_log = viewer.rbt_loc_log
    rgs_env.path_log = viewer.nav_path_log
    print('rgs_env init finished!')
    frame_id = 0
    frame_idx_end =  len(rgs_env.occgrid_log)
    # 0.05s per frame
    # 941 frames in total

    xvec, lpg, nav_path, lidar_pts, occgrid, LEE_specs, GDE_specs, xr_log_mat \
        = viewer.extract_frame_new(frame_id)

    xv2D, xg2D = xvec[2:4], xvec[4:6]
    rbt3D, xr3D, xv3D, xg3D, lpg3D, nav_path3D, lidar_pts3D, rbt3D_trail = \
        expand_elements_dim(xvec, lpg, nav_path, occgrid, lidar_pts, xr_log_mat)

    vp = vplt.Plotter(bg='white', axes={'xyGrid':True,
                                        'zxGrid2':True,
                                        'showTicks':True})
    # static
    print('add environment...')
    wall = vplt.load('../mesh/dense_walls.stl').c("grey").alpha(0.2).addShadow(z=shadow_zval)
    obs = vplt.load("../mesh/dense_obs.stl").c("grey").addShadow(z=shadow_zval)
    start_actor = vplt.load("../mesh/star.stl").c("red").pos(start3D)
    goal_actor = vplt.load("../mesh/star.stl").c("green").pos(goal3D)
    print('Init upper level...')

    # time and status display
    text_actor = vplt.Text('Time %4.2f' %(frame_id * viewer.dt), c='black')
    # robot
    theta_last = get_theta_from_vel(xv2D)
    robot_actor = vplt.load("../mesh/robocar_meter.stl").c("green").addShadow(z=RBT_HEIGHT-0.2)
    robot_actor.pos(rbt3D).scale(rbt_scaling)
    robot_actor.shadow.scale(rbt_scaling)
    robot_actor.rotateZ(get_theta_from_vel(xv2D))
    # velocity arrow
    vel_actor = vplt.Arrow(rbt3D, rbt3D + xv3D, c='magenta').scale(2)
    lidar_pts_actor = vplt.Points(lidar_pts3D, r = 4, c='red')
    # trail of robot
    rbt3D_trail_actor = vplt.Line(rbt3D_trail, c='green', lw=5)

    print('Init lower level...')
    # robot velocity vector
    xr3D_actor = vplt.Point(xr3D, r=8, c='green')
    gov_actor = vplt.Point(xg3D, r=8, c='blue')
    lpg_actor = vplt.Point(lpg3D, r=6, c='red')
    nav_path_actor = vplt.Line(nav_path3D, c='blue', lw=3, dotted=True)
    # ellipse GDE and LEE

# %% make video
    vp = vplt.Plotter(bg='white', axes={'xyGrid':True,
                                    'zxGrid2':True,
                                    'showTicks':True})

    print('Start playing video')
    for fidx in range(frame_idx_end):
        time_now = fidx * viewer.dt
        print('Frame %4d | time %.2f' %(fidx, time_now))
        # udpate actors
        text_actor = vplt.Text('Time %4.2f sec' %(time_now), c='black')

        xvec, lpg, nav_path, lidar_pts, occgrid, LEE_specs, GDE_specs, xr_log_mat \
            = viewer.extract_frame_new(fidx)

        xv2D, xg2D = xvec[2:4], xvec[4:6]
        rbt3D, xr3D, xv3D, xg3D, lpg3D, nav_path3D, lidar_pts3D, rbt3D_trail = \
            expand_elements_dim(xvec, lpg, nav_path, occgrid, lidar_pts, xr_log_mat)

        theta_new = get_theta_from_vel(xv2D)
        d_theta = theta_new - theta_last
        robot_actor.pos(rbt3D).rotateZ(d_theta)
        theta_last = theta_new
        gov_actor.pos(xg3D)
        lpg_actor.pos(lpg3D)

        # redraw
        vel_actor = vplt.Arrow(rbt3D, rbt3D + xv3D, c='magenta').scale(2)
        lidar_pts_actor = vplt.Points(lidar_pts3D, r = 6, c='red')
        rbt3D_trail_actor = vplt.Line(rbt3D_trail, c='green', lw=5)
        GDE_actor = ellipse2Dto3D(xg2D, GDE_specs, ec='gray')
        LEE_actor = ellipse2Dto3D(xg2D, LEE_specs, ec='yellow', ea=0.5)
        nav_path_actor = vplt.Line(nav_path3D, c='blue', lw=3, dotted=True)

        vplt.show(wall, obs, text_actor, start_actor, goal_actor,
                  robot_actor, vel_actor, lidar_pts_actor, rbt3D_trail_actor,
                  gov_actor, lpg_actor, nav_path_actor,
                  GDE_actor, LEE_actor,
                  interactive=0,
                  # camera setting may varies on different computer/monitors
                  camera={'pos':[-0.114, -21.332, 35.687],
                          'focalPoint':[9.611, 2.363, 0.07],
                          'viewup':[0.267, 0.767, 0.583],
                          'distance':43.871,
                          'clippingRange':[33.465, 57.074]})

        if not os.path.exists(video_frame_fd):
            os.makedirs(video_frame_fd)

        frame_name = 'frame_' + str(10000+fidx)+'.png'
        fig2_name_wpath = os.path.join(video_frame_fd, frame_name)
        vplt.screenshot(fig2_name_wpath)
        vp.remove(text_actor)
    print('Video finished!')
    vplt.closeWindow()
    toc(t1, "creating 3D video")
