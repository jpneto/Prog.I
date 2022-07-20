# -*- coding: utf-8 -*-

###################################################################
# Code by Peter Norvig
# ref: https://github.com/norvig/pytudes/blob/main/ipynb/Maze.ipynb
#
# MIT License, https://github.com/norvig/pytudes/blob/main/LICENSE
###################################################################

import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple

Edge = tuple
Tree = set

def edge(A, B) -> Edge: 
  return Edge(sorted([A, B]))

def random_tree(nodes, neighbors, pop=deque.pop) -> Tree:
    """Repeat: pop a node and add edge(node, nbr) until all nodes have been added to tree."""
    tree = Tree()
    nodes = set(nodes)
    root = nodes.pop()
    frontier = deque([root])
    while nodes:
        node = pop(frontier)
        nbrs = neighbors(node) & nodes
        if nbrs:
            nbr = random.choice(list(nbrs))
            tree.add(edge(node, nbr))
            nodes.remove(nbr)
            frontier.extend([node, nbr])
    return tree



Maze = namedtuple('Maze', 'width, height, edges')

Square = tuple

def neighbors4(square) -> {Square}:
    """The 4 neighbors of an (x, y) square."""
    (x, y) = square
    return {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}

def grid(width, height) -> {Square}: 
    """All squares in a grid of these dimensions."""
    return {(x, y) for x in range(width) for y in range(height)}

def random_maze(width, height, pop=deque.pop) -> Maze:
    """Generate a random maze, using random_tree."""
    tree = random_tree(grid(width, height), neighbors4, pop)
    return Maze(width, height, tree)


def plot_maze(maze, figsize=None, path=None, frontier=None):
    """Plot a maze by drawing lines between adjacent squares, except for pairs in maze.edges"""
    w, h  = maze.width, maze.height
    plt.figure(figsize=figsize or (w/5, h/5))
    plt.axis('off')
    plt.gca().invert_yaxis()
    exits = {edge((0, 0), (0, -1)), edge((w-1, h-1), (w-1, h))}
    edges = maze.edges | exits
    for sq in grid(w, h):
        for nbr in neighbors4(sq):
            if edge(sq, nbr) not in edges:
                plot_wall(sq, nbr)
    if path: # Plot the solution (or any path) as a red line through the maze
        X, Y = transpose((x + 0.5, y + 0.5) for (x, y) in path)
        plt.plot(X, Y, 'r-', linewidth=2)
    if frontier:
        for X,Y in frontier:
          plt.plot(X+0.5,Y+0.5, marker="o", markersize=8, 
                   markeredgecolor="green", markerfacecolor="green")
        
def transpose(matrix): return list(zip(*matrix))

def plot_wall(s1, s2):
    """Plot a wall: a black line between squares s1 and s2."""
    (x1, y1), (x2, y2) = s1, s2
    if x1 == x2: # horizontal wall
        y = max(y1, y2)
        X, Y = [x1, x1+1], [y, y]
    else: # vertical wall
        x = max(x1, x2)
        X, Y = [x, x], [y1, y1+1]
    plt.plot(X, Y, 'k-', linewidth=2)  
    
    
def breadth_paths(maze):
    before, start = (0, -1), (0, 0)
    goal = (maze.width-1, maze.height-1)
    frontier = deque([start])  # states to consider
    paths = {start: [before, start]}   # start has a one-square path
    path_history = [paths[start]]
    frontier_history = [list(frontier)]
    while frontier:
        frontier_history.append(list(frontier))
        s = frontier.popleft()
        path_history.append(paths[s])
        if s == goal:
            return path_history, frontier_history
        for s2 in neighbors4(s):
            if s2 not in paths and edge(s, s2) in maze.edges:
                frontier.append(s2)
                paths[s2] = paths.get(s, []) + [s2]    
                
def depth_paths(maze):
    before, start = (0, -1), (0, 0)
    goal = (maze.width-1, maze.height-1)
    frontier = deque([start])  # states to consider
    paths = {start: [before, start]}   # start has a one-square path
    path_history = [paths[start]]
    frontier_history = [list(frontier)]
    while frontier:
        frontier_history.append(list(frontier))
        s = frontier.pop()
        path_history.append(paths[s])
        if s == goal:
            return path_history, frontier_history
        for s2 in neighbors4(s):
            if s2 not in paths and edge(s, s2) in maze.edges:
                frontier.append(s2)
                paths[s2] = paths.get(s, []) + [s2]