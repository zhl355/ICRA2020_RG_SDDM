import numpy as np  
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../lib")
import astar_py as pyAstar
import gridutils_py as pyGridUtils

def testPyStar():
  cmap = np.zeros((5,5),dtype=bool)
  cmap[2,2] = True
  start = np.array([0,0])
  goal = np.array([4,4])
  pcost, path = pyAstar.planOn2DGrid(cmap, start, goal)
  print(pcost)
  print(path)

def testBresenham():
  start = np.array([-0.6743,-1.2151,3.1086])
  end = np.array([-6.3607,-3.0567,-7.2978])
  gmin = np.array([-4.5,-4.5,-4.5])
  gres = np.array([1.0,1.0,1.0])
  cells = pyGridUtils.bresenham(start,end,gmin,gres)
  print(cells)
  
def testInflateMap():
  map_size = np.array([11,11])
  map_res = np.array([0.2,0.2])
  cmap = np.zeros((map_size[0],map_size[1]),dtype=np.int64)
  cmap[6,6] = 10
  cmap[0,10] = 10  
  inflation_radius = np.array([0.5,0.5])
  
  imap = pyGridUtils.inflateMap(cmap.flatten(),cmap.shape,map_res,True,inflation_radius)
  imap = np.reshape(imap,(cmap.shape[0],cmap.shape[1]))
  print(cmap)
  print('\n')
  print(imap)
  
if __name__ == '__main__':
  testPyStar()
  testBresenham()
  testInflateMap()

