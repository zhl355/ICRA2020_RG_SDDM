#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:35:31 2018
BSD 3-Clause License
@author: l3-cz
"""
import numpy as np
from my_utils import wall_corners2_ls
# Environment setup
X_LOW, X_HIGH, Y_LOW, Y_HIGH = 0, 320, 0, 240
BOUNDARY_PTS_1 = [X_LOW, X_HIGH, Y_LOW, Y_HIGH]

# Obstacles: a collection of circles [x,y,r]
OBSTACLES_1 = np.array([    [ 30,  80,  20],
                            [ 80, 160,  20],
                            [100,  50,  15],
                            [135,  95,  15],
                            [180, 170,  20],
                            [235,  85,  35],
                            [260, 175,  20]], dtype=np.float64)

# Navigation Path, a collection of way pts, 1st elements is start
# the last one is the goal
NAV_PATH_1 = np.array([ [ 20,  20],
                        [ 90, 130],
                        [135, 130],
                        [175, 130],
                        [220, 130],
                        [260, 165],
                        [300, 220]  ])

NAV_PATH_12 = np.array([ [ 20,  20],
                        [ 90, 130],
                        [135, 130] ])

# for P choice simulation
NAV_PATH_13 = np.array([ [ 20,  20],
                        [ 90, 130],
                        [110, 130] ])


NAV_PATH_14 = np.array([[ 40,  20],
                        [100, 130],
                        [220, 130],
                        [220, 220],
                        [300, 220] ])


# Another Enviroment Set Up
BOUNDARY_PTS_2 = [0, 400, 0, 40]

# Obstacles: a collection of circles [x,y,r]
OBSTACLES_2 = []

# Navigation Path, a collection of way pts, 1st elements is start
# the last one is the goal

NAV_PATH_2 = np.array([ [ 20,  20],
                        [ 380,  20] ])


# dense enviroment
BOUNDARY_PTS_3 = [0, 200, 0, 200]
OBSTACLES_3 = np.array([    [ 50,  50,  20],
                            [100,  50,  20],
                            [150,  50,  20],
                            [ 50,  100,  20],
                            [100,  100,  20],
                            [150,  100,  20],
                            [ 50,  150,  20],
                            [100,  150,  20],
                            [150,  150,  20]])

NAV_PATH_3 = np.array([[25, 25],
                       [75, 25],
                       [75, 75],
                       [125, 75],
                       [125, 125],
                       [175, 125],
                       [175, 175]])





class MyMap:
    """
    Initial map using global variables, later add loading external map feature
    """

    def __init__(self, bd_pts, obstacles, nav_path):
        """ Load one map from global variables.
        """
        xl, xh, yl, yh = bd_pts
        self.boundary = bd_pts
        self.obstacles = obstacles
        self.nav_path = nav_path
        self.start_pt = nav_path[0]
        self.goal_pt  = nav_path[-1]
        self.corners = np.array([[xl, yl], [xl, yh], [xh,yh], [xh, yl]])
        self.wall_ls = wall_corners2_ls(self.corners)

if __name__ == '__main__':
    print('Create the map for simualtion...')
    rgs_map = MyMap(BOUNDARY_PTS_1, OBSTACLES_1, NAV_PATH_1)