#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy( self,time_allowance=60.0 ):
		start_time = time.time()
		cities = self._scenario.getCities()
		ncities = len(cities)
		solutionDict = {}
		results = {}
		bssf = float('inf')

		# Build network connectivity graph
		graph = np.zeros((ncities,ncities))
		for i in range(ncities):
			for j in range(ncities):
				if i == j:
					graph[i,j] = float('inf')
				else:
					graph[i,j] = cities[i].costTo(cities[j])

		# Try to find greedy solution from each city until a complete solution is found
		for start in range(ncities):
			# Only iterate if still within time constraint
			if time.time() - start_time > time_allowance:
				break
			# Create a temporary copy of graph
			_graph = graph.copy()
			currentSolution = []

			# Add cities to currentSlution until all cities are added

			currentSolution.append(cities[start])
			_graph[:, start] = float('inf')
			current = start
			solutionCost = 0
			for count in range(ncities-1):
				# Find outbound edge w/ lowest cost, add destination to temp solution
				minCost = float('inf')
				minIndex = -1
				for i in range(ncities):
					if _graph[current,i] < minCost:
						minCost = _graph[current,i]
						minIndex = i

				# Break if no outbound edges available, find path starting from next city
				if minCost == float('inf'):
					break

				# Set path from all cities toward shortest destination to infinite
				solutionCost += minCost
				_graph[:, minIndex] = float('inf')
				currentSolution.append(cities[minIndex])
				current = minIndex

				if len(currentSolution) == ncities:
					# solutionCost += cities[current].costTo(cities[start])
					# currentSolution.append(cities[start])
					solutionDict[solutionCost] = TSPSolution(currentSolution)
					if solutionCost < bssf:
						bssf = solutionCost
		end_time = time.time()

		results['cost'] = bssf
		results['time'] = end_time-start_time
		results['count'] = len(solutionDict)
		results['soln'] = solutionDict[bssf]
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):
		bssf = self.greedy(time_allowance)
		pass



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	def fancy( self,time_allowance=60.0 ):
		pass
