# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class BacktrackProblem(ABC):
  @abstractmethod
  def start(self):
    pass

  @abstractmethod
  def reject(self, state):
    pass

  @abstractmethod
  def accept(self, state):
    pass

  @abstractmethod
  def next_moves(self, state):
    pass

###
    
def backtrack(problem, state=None, cache=None): 
  # state : current state/candidate (must be immutable)
  # cache : set of previous candidates
  if state is None:
    state = problem.start() 
    cache = set()
 
  if not problem.reject(state):                    # if the state is not rejected,
    if problem.accept(state):                      #  check if it is a solution
      yield state                                  #  if so, yield it
    else:
      for s2 in problem.next_moves(state):         # otherwise, build a bit more of the state
        if s2 not in cache:                        # and if it is a new original state,
          cache.add(s2)                            #  cache it (to prevent infinite loops)
          yield from backtrack(problem, s2, cache) #  and search it

###############################################################################

from abc import ABC, abstractmethod

class TreeSearchProblem(ABC):
  @abstractmethod
  def start(self):
    pass

  @abstractmethod
  def is_goal(self, state):
    pass

  @abstractmethod
  def next_moves(self, state):
    pass

###

from collections import deque

def bfs(problem):
  start, next_moves, is_goal = problem.start(), problem.next_moves, problem.is_goal

  frontier = deque([start])    # current states to evaluate, FIFO access
  paths = {start: [start]}     # path[s] gives the path from start to s
  while frontier:              # while there are states in the frontier,
    s = frontier.popleft()     #  get first state in the frontier
    if is_goal(s):             #  if it's the goal, the search ended
      yield paths[s]
    else:
      for s2 in next_moves(s): #  if it's not the goal, iterate all next moves from s,
        if s2 not in paths:    #   if it is a new state, 
          frontier.append(s2)  #    add it to frontier
          paths[s2] = paths.get(s, []) + [s2]  # and create a new path for s2 based on path s    
          
def dfs(problem):
  start, next_moves, is_goal = problem.start(), problem.next_moves, problem.is_goal

  frontier = deque([start])
  paths = {start: [start]}
  while frontier:
    s = frontier.pop() # the LIFO access is the only difference from bfs
    if is_goal(s):
      yield paths[s]
    else:
      for s2 in next_moves(s):
        if s2 not in paths:
          frontier.append(s2)
          paths[s2] = paths.get(s, []) + [s2]  

###############################################################################

from abc import ABC, abstractmethod

class HeuristicProblem(ABC):
  @abstractmethod
  def start(self):
    pass

  @abstractmethod
  def next_moves(self, state):
    pass

  @abstractmethod
  def cost(self, state, next_state):
    pass

  @abstractmethod
  def heuristic(self, state):
    pass

###

from heapq import heappop, heappush

def a_star(problem):
  """ Find a shortest sequence of states from start to a goal state (where h(goal)==0) """
  start, next_moves, heuristic, cost = problem.start(), problem.next_moves, problem.heuristic, problem.cost

  frontier  = [(heuristic(start), start)] # priority queue ordered by path length, f(s) = g(s) + h(s)
  previous  = {start: None}               # start state has no previous state; other states will
  path_cost = {start: 0}                  # the cost of the best path to a state
  path      = lambda s: ([] if (s is None) else path(previous[s])+[s])

  while frontier:
    f, s = heappop(frontier)
    if heuristic(s) == 0:
      return path(s)
    for s2 in next_moves(s):
      g = path_cost[s] + cost(s, s2)
      if s2 not in path_cost or g < path_cost[s2]:
        heappush(frontier, (g + heuristic(s2), s2))
        path_cost[s2] = g
        previous[s2] = s