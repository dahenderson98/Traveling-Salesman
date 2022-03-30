#!/usr/bin/python3


from PyQt6.QtCore import QLineF, QPointF


import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from queue import PriorityQueue
from TSPSolver import *
from TSPClasses import *

# Press the green button in the gutter to run the script.
def newPoints(n, seed):
    SCALE = 1
    data_range = {'x': [-1.5 * SCALE, 1.5 * SCALE], 'y': [-SCALE, SCALE]}
    random.seed(seed)

    ptlist = []
    RANGE = data_range
    xr = data_range['x']
    yr = data_range['y']
    npoints = n
    while len(ptlist) < npoints:
        x = random.uniform(0.0, 1.0)
        y = random.uniform(0.0, 1.0)
        if True:
            xval = xr[0] + (xr[1] - xr[0]) * x
            yval = yr[0] + (yr[1] - yr[0]) * y
            ptlist.append(QPointF(xval, yval))
    return ptlist


if __name__ == '__main__':

    TIME_ALLOWANCE = 10000
    SEED = 1    #random.randint(0,1001)
    DIFF = "Normal"
    N = 4

    solver = TSPSolver()
    points = newPoints(N, SEED)
    scenario = Scenario(points,DIFF,SEED)
    solver.setupWithScenario(scenario)
    results = solver.branchAndBound(TIME_ALLOWANCE)
    print("BnBResults: ",results)


