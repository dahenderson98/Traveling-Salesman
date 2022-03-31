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
from queue import PriorityQueue
import copy



class TSPSolver:
	def __init__( self, gui_view=None):
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
		graph = np.zeros((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				if i == j:
					graph[i, j] = float('inf')
				else:
					graph[i, j] = cities[i].costTo(cities[j])

		# Try to find greedy solution from each city until a complete solution is found
		for start in range(ncities):
			# Only iterate if still within time constraint
			if time.time() - start_time > time_allowance:
				break
			# Create a new copy of graph
			_graph = np.copy(graph)
			currentSolution = []

			# Add cities to currentSolution until all cities are added

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
		# Initialize variables to hold values to be returned
		solutionDict = {}
		results = {}
		maxQueueSize = 0
		totalStates = 0
		prunedStates = 0
		bound = 0

		# Load list of Cities
		cities = self._scenario.getCities()
		ncities = len(cities)

		# Mark algorithm's ending time
		start_time = time.time()

		# Set BSSF to minimum cost of 3 random tours, and add the shortest path of the 3 to solution dictionary
		randomResults1 = self.defaultRandomTour(time_allowance)
		randomResults2 = self.defaultRandomTour(time_allowance)
		randomResults3 = self.defaultRandomTour(time_allowance)
		if randomResults1['cost'] <= randomResults2['cost'] and randomResults1['cost'] <= randomResults3['cost']:
			bssf = randomResults1['cost']
			solutionDict[bssf] = randomResults1['soln']
		elif randomResults2['cost'] < randomResults2['cost'] and randomResults1['cost'] <= randomResults3['cost']:
			bssf = randomResults2['cost']
			solutionDict[bssf] = randomResults2['soln']
		else:
			bssf = randomResults3['cost']
			solutionDict[bssf] = randomResults3['soln']

		# Using greedy for baseline BSSF
		# greedyResults = self.greedy(time_allowance)
		# bssf = greedyResults['cost']
		# solutionDict[bssf] = greedyResults['soln']

		# Hard-coding baseline BSSF
		# bssf = float('inf')
		# solutionDict[bssf] = None

		# Build network connectivity graph
		graph = np.zeros((ncities, ncities))
		for i in range(ncities):
			for j in range(ncities):
				if i == j:
					graph[i, j] = float('inf')
				else:
					graph[i, j] = cities[i].costTo(cities[j])

		# Reduce graph by rows, adding minCosts to bound
		for i in range(ncities):
			minCost = np.min(graph[i,:])
			graph[i,:] -= minCost
			bound += minCost

		# Reduce graph by columns, adding minCosts to bound
		for j in range(ncities):
			minCost = np.min(graph[:,j])
			graph[:,j] -= minCost
			bound += minCost

		# Create a new search State including the initial reduced cost matrix and bound associated
		# * Each path will start using this rootState search State
		rootState = State(bound=bound, graph=np.copy(graph), citiesIndex=-55555, pathSoFar=None, depth=1)

		# Initialize priority queue to keep track of search States (partial paths)
		pq = PriorityQueue()

		# Find best solutions from each starting city
		for start in range(ncities):
			# Iterate while within time constraint
			currentTime = time.time()
			if currentTime - start_time > time_allowance:
				break

			# Add rootState to the priority queue
			# * The priority queue weighs State priority based on bound / (2 * depth), prioritizing deep States and low-bounded States
			pq.put((rootState.bound/2, rootState))
			totalStates += 1
			if pq.qsize() > maxQueueSize:
				maxQueueSize = pq.qsize()

			while not pq.empty():
				# Iterate while within time constraint
				if time.time() - start_time > time_allowance:
					while not pq.empty():
						if pq.get()[1].bound >= bssf:
							prunedStates += 1
					break

				# Get/remove highest-priority (lowest-bound) state from the PQ -> (key, State)
				current = copy.deepcopy(pq.get()[1])

				# Throw away state if its bound is not less than BSSF, increment prunedStates
				if current.bound < bssf:
					# If partial path is shorter than ncities, expand state
					if len(current.pathSoFar) < ncities:
						# If current State has rootState's initial citiesIndex, set current citiesIndex to the starting city's index
						if current.citiesIndex == -55555:
							currentCityIndex = start
						else:
							currentCityIndex = current.citiesIndex

						# Add all reachable child cities to PQ as States (skip cities that were already visited)
						for destination in range(ncities):
							if np.array(current.graph)[currentCityIndex, destination] != float('inf'):
								newBound = current.bound + np.array(current.graph)[currentCityIndex, destination]
								_graph = np.copy(np.array(current.graph))
								_graph[currentCityIndex, :] = float('inf')
								_graph[:, destination] = float('inf')

								# Reduce graph by rows if needed, adding minCosts to bound
								# Skip originating city row, as all distances are now inf there
								reductionCost = 0
								for i in range(ncities):
									if i == currentCityIndex:
										continue
									minCost = np.min(_graph[i, :])
									if minCost != float('inf'):
										_graph[i, :] -= minCost
										reductionCost += minCost

								# Reduce graph by columns if needed, adding minCosts to bound
								# Skip originating city column, as all distances are now inf there
								for j in range(ncities):
									if j == destination:
										continue
									minCost = np.min(_graph[:, j])
									if minCost != float('inf'):
										_graph[:, j] -= minCost
										reductionCost += minCost
								# Add cost of reduction to bound of new State (partial path)
								newBound += reductionCost

								# Add start city to partial path if path is empty, inherit current.pathSoFar otherwise
								# Append destination city to partial path
								pathSoFar = []
								if len(current.pathSoFar) == 0:
									pathSoFar.append(cities[start])
								else:
									pathSoFar = current.pathSoFar.copy()
								pathSoFar.append(cities[destination])

								# Increment depth by 1 for new State
								newDepth = current.depth+1

								# Add a new State to PQ representing a possible branch to traverse
								newState = State(bound=newBound, graph=_graph, citiesIndex=destination, pathSoFar=pathSoFar.copy(), depth=newDepth)
								pq.put((newBound / (current.depth * 2), newState))
								totalStates += 1
								if pq.qsize() > maxQueueSize:
									maxQueueSize = pq.qsize()
					# Else, if size of partial path is  equal to ncities; add cost of returning to start and store as good solution
					# * Only store as a good solution if final bound is less than current BSSF
					elif len(current.pathSoFar) == ncities:
						newBound = current.bound + current.pathSoFar[-1].costTo(cities[start])
						if newBound < bssf:
							bssf = newBound
							solutionDict[bssf] = TSPSolution(current.pathSoFar)
					# Else, path is too long (invalid), prune it -> this shouldn't ever happen
				else:
					prunedStates += 1

		# Mark algorithm's ending time
		end_time = time.time()

		results['cost'] = bssf
		results['time'] = end_time - start_time
		results['count'] = len(solutionDict) - 1
		results['soln'] = solutionDict[bssf]
		results['max'] = maxQueueSize
		results['total'] = totalStates
		results['pruned'] = prunedStates
		return results


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
