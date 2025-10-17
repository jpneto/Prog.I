# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:15:54 2025

@author: jpn3t
"""
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import Image

####### Graph related non-CGT functions #########

def showGraph(g, name='temp', graphviz=True, prog='neato', weights=False):
  """ g        : networkx graph object
      name     : defines dot/png temporary filenames
      graphviz : if False uses networkx display, otherwise uses Graphviz's
      prog     : defines Graphviz's layout engine (egs: dot, neato)
      weights  : if True add weights to graph display
  """
  if graphviz:
    for node in g.nodes():
      g.nodes[node]['width'] = 0.5
    if weights:
      # for graphviz, nodes/edges with a label attr will be printed
      for edge in g.edges():
        g.edges[edge]['label'] = g.edges[edge]['weight']
    A = nx.nx_agraph.to_agraph(g)
    A.draw(f'{name}.png', prog=prog, args='-Gnodesep=0.2 -Granksep=0.25') # write graph as a png file
    A.write(f'{name}.dot')           # write graph in DOT notation
    display(Image(f'{name}.png'))    # display image
  else:
    options = { # https://networkx.org/documentation/latest/auto_examples/basic/plot_simple_graph.html
        "font_size": 12,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
        "with_labels": True,
    }
    pos = nx.nx_pydot.graphviz_layout(g, prog=prog)  # get node positions to place labels
    nx.draw_networkx(g, pos, **options)
    # format color for nodes
    colors = [g.nodes[n].get('color', 'white') for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors)
    # format color and width for edges
    colors = [g.edges[a,b].get('color', 'black') for a,b in g.edges]
    widths = [g.edges[a,b].get('width', 1)       for a,b in g.edges]
    labels = [g.edges[a,b].get('label', '')      for a,b in g.edges]
    nx.draw_networkx_edges(g, pos, edge_color=colors, width=widths, label=labels)
    # draw weights
    if weights:
      labels = nx.get_edge_attributes(g, 'weight')
      nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    
def makeGraph(nodes=None, edges=None, digraph=True):
  if digraph: g = nx.DiGraph()
  else:       g = nx.Graph()
  if nodes:   g.add_nodes_from(nodes)
  if edges:   g.add_edges_from(edges)
  return g

def make_edges(g):
  # read a graph like this:
    # graph = """
    # A>B A>C
    # B>C
    # B<D B>E C<E C<F
    # D>E E<F
    # """
  # and return a list of pairs describing edges
  # that can be used to build a networkx object, like this:
    # g = makeGraph(edges=make_edges(graph), digraph=True)
  edges = []
  for line in g.split():
    if '<' in line:
      a, b = line.split('<')
      edges.append( (b,a) )
    else:
      a, b = line.split('>')
      edges.append( (a,b) )
  return edges

assert make_edges("A>B A>C B<C") == [('A','B'), ('A','C'), ('C','B')]


# main user function
def text2graph(graph_text, show=False):
  g = makeGraph(edges=make_edges(graph))
  if show:
    showGraph(g)
  return g


############## Spran-Grundy Theory ################        
 
def mexes(ns):
  if not ns:
    return 0
  for i in range(max(ns)+2):
    if i not in ns:
      return i

def sprang_grundy(g, verbose=False):
  assigns = {}                # values to assign each node
  to_assign = list(g.nodes)   # what nodes still need assigment
  succ_values = {v:[] for v in to_assign} # which successors are computed

  if verbose:
    print('\nSprang-Grundy example:')
    print('Node order: ', end='')
    
  # compute mex values to graph
  while to_assign:
    for node in to_assign:
      if len(list(g.successors(node))) == len(succ_values[node]):
        # everything was computed
        to_assign.remove(node) # no need to look again at this node
        assigns[node] = mexes(succ_values[node])
        if verbose:
          print(node, assigns[node], end='  ')
        # communicate this new value to all its predecessors
        for succ in g.predecessors(node):
          succ_values[succ].append(assigns[node])
  
  # change graph labels to also show mex values
  new_labels = {node: f"{node}/{mex}"
                for node, mex in assigns.items()}
  if verbose:
    print('\n', sorted(new_labels.values()))
  return nx.relabel_nodes(g, new_labels)  # change graph labels

if __name__ == "__main__":
  graph = """
  A>B A>C
  B>C
  B<D B>E C<E C<F
  D>E E<F
  D<G D<H E<H E<I F<I F<J
  G<H H>I I>J
  """
  g = makeGraph(edges=make_edges(graph))
  sprang_grundy(g, verbose=True) 
  # {'B/2', 'J/0', 'A/1', 'C/0', 'I/3', 'G/1', 
  #  'F/2', 'D/0', 'H/2', 'E/1'}
  
  
############## Frankel-Smith-Pearl Theory ################        

def compatible(g, value, node, assigns):
  """ Check if there is enough information, to assign a proposed 'value' to the given 'node'
      The 'assigns' dictionary provides the needed information, besides the graph g itself """
  # check the values already assigned to the node's sucessors
  values = {assigns[succ] for succ in g.successors(node)
                          if succ in assigns}
  if value != mexes(values):
    return False  # the mex value of the node does not fit the proposed value

  # need also to check if the remaining nodes without values, or greater than n, revert to n
  for succ in g.successors(node):
    if succ not in assigns or assigns[succ] > value:
      found = False
      # to revert to n, at least one successor of each successor must be equal to n
      for next_succ in g.successors(succ):
        if next_succ in assigns and assigns[next_succ] == value:
          found = True
          break
      if not found:
        return False

  return True

def frankel_smith_pearl(g, verbose=False):
  assigns = {}                # values to assign each node
  to_assign = list(g.nodes)   # what nodes still need assigment
  succ_values = {v:[] for v in to_assign} # which successors are computed

  if verbose:
    print('\nFrankel-Smith-Pearl example:')
    print('Node order: ', end='')

  # search terminal nodes
  for node in g:
    if not list(g.successors(node)):
      to_assign.remove(node)
      assigns[node] = 0
      if verbose:
        print(node, assigns[node], end='  ')
      for succ in g.predecessors(node):
        succ_values[succ].append(assigns[node])

  for value in range(len(g)):
    double_check = to_assign[:]*len(g)
    for node in double_check:
      if node not in assigns and compatible(g, value, node, assigns):
        to_assign.remove(node)
        assigns[node] = value
        if verbose:
          print(node, assigns[node], end='  ')
        # communicate this new value to all its predecessors
        for succ in g.predecessors(node):
          succ_values[succ].append(value)

  new_labels = {node: f"{node}/{mex}" for node, mex in assigns.items()}
  ## add cycle Grundy values to the labels
  for node in set(g.nodes) - assigns.keys():
    succs = ','.join(str(assigns[succ]) for succ in g.successors(node)
                                        if succ in assigns)
    new_labels[node] = f'{node}/∞' + (succs if succs else '')

  if verbose:
    print('\n', sorted(new_labels.values()))
  return nx.relabel_nodes(g, new_labels)  # change graph labels


if __name__ == "__main__":
  graph2 = """
  A>B A>C
  B>C
  B<D B>E C<E C<F
  D>E E<F
  D<G D<H E>H E<I F<I F<J
  G<H H>I I>J
  G<K G<L H<L H<M I<M I>N J<N J>O
  K<L L<M M<N N<O
  """
  g2 = makeGraph(edges=make_edges(graph2))
  frankel_smith_pearl(g2, verbose=True) 
  # ['A/1', 'B/2', 'C/0', 'D/0', 'E/1', 'F/2', 'G/1', 'H/2', 
  #  'I/∞1,2', 'J/∞2', 'K/0', 'L/3', 'M/∞2,3', 'N/∞', 'O/∞']
  
############## Larsen-Nowakovski-Santos Theory ################        

def add_grey_nodes(g, nodes):
  for node in nodes:
    g.nodes[node]['shape'] = 'box'
    
from math import inf as Z

class Grey:
  def __init__(self):
    self.excepts = set()
  
  def __add__(self, vals):
    self.excepts |= vals
    
  def __sub__(self, vals):
    self.excepts -= vals
    
  def __repr__(self):
    if self.excepts:
      return 'Z\\\\{' + ','.join(map(str,self.excepts)) + '}'
    return 'Z'
    

def mexes_grey(node, succs, grey_nodes):
  white_vals = {val for val in succs
                    if type(val) == int}
  grey_vals = {elem for val in succs
                    if type(val) == Grey
                    for elem in val.excepts}

  if node not in grey_nodes: # it is a regular white node
    # if all sucessors are finite numbers and no successor is a grey node
    # then just use regular mex()
    if not grey_vals and all(type(x) != Grey for x in succs):
      return mexes(white_vals)
    # if all values are available, then it is a moon
    if white_vals == grey_vals or any(type(x) == Grey and not x.excepts for x in succs):
      return Z
    # otherwise, we need to check what are still the available options
    return min(grey_vals - white_vals)

  else: # it is a grey node
    val = Grey()
    val + (white_vals | grey_vals) # ie, Z \ { white values ⋃ grey values }
    return val
  

def larsen_nowakovski_santos(g, grey_nodes, verbose=False):
  """ assigns values to the direct cyclic graph, following the Larsen-Nowakovski-Santos theory """
  assigns = {}                            # values to assign each node
  to_assign = list(g.nodes)               # what nodes still need assigment
  succ_values = {v:[] for v in to_assign} # which successors are computed

  if verbose:
    print('\nLarsen-Nowakovski-Santos example:')
    print('Node order: ', end='')
    
  # search terminal nodes, and assign them the value zero
  for node in g:
    if not list(g.successors(node)):
      to_assign.remove(node)
      assigns[node] = Grey() if node in grey_nodes else 0
      if verbose:
        print(node, assigns[node], end='  ')
      
      for succ in g.predecessors(node):
        succ_values[succ].append(assigns[node])

  # compute mex values of all remaining nodes of graph
  while to_assign:
    for node in to_assign:
      if len(list(g.successors(node))) == len(succ_values[node]):
        to_assign.remove(node)
        # everything computed, compute adjusted-mex value
        assigns[node] = mexes_grey(node, succ_values[node], grey_nodes)
        if verbose:
          print(node, assigns[node], end='  ')       
        # communicate this new value to all its predecessors
        for succ in g.predecessors(node):
          succ_values[succ].append(assigns[node])

  new_labels = {} # relabel nodes to include their value
  for node, mex in assigns.items():
    if mex != Z:
      new_labels[node] = f"{node}/{mex}"
    else:
      new_labels[node] = f"{node}/(("

  if verbose:
    print('\n', sorted(new_labels.values()))
  return nx.relabel_nodes(g, new_labels)  # change graph labels


if __name__ == "__main__":
  graph3 = """
  A>B A>C
  B>C
  B<D B>E C<E C<F
  D>E E>F
  D>G D<H E<H E<I F<I F<J
  G<H H>I I>J
  G<K G<L H<L H<M I<M I>N J<N J>O
  K<L L<M M>N N>O
  """
  g3 = makeGraph(edges=make_edges(graph3))
  add_grey_nodes(g3, 'EG')
  larsen_nowakovski_santos(g3, grey_nodes='EG', verbose=True)






##########################################################
# [old code, to delete]

# def mexes_grey(node, succs, grey_nodes):
#   white_vals = {val for val in succs 
#                     if type(val) == int}
    
#   grey_vals = {elem for val in succs
#                     if type(val) == Grey
#                     for elem in val.excepts}

#   if node not in grey_nodes: # it is a regular white node

#     if not grey_vals and all(type(x) != Grey for x in succs):
#       return mexes(white_vals) # a regular node
#     if white_vals == grey_vals or \
#        any(type(x) == Grey and not x.excepts for x in succs):
#       return Z # moon value, all values are possible moves
#     return min(grey_vals - white_vals)
    
#   else: 
#     val = Grey()
#     val + (white_vals | grey_vals)
#     return val

# def larsen_nowakovski_santos(g, grey_nodes):
#   """ assigns values to the direct cyclic graph, following the Larsen-Nowakovski-Santos theory """  
#   assigns = {}                            # values to assign each node
#   to_assign = list(g.nodes)               # what nodes still need assigment
#   succ_values = {v:[] for v in to_assign} # which successors are computed

#   # search terminal nodes, and assign them the value zero
#   for node in g:
#     if not list(g.successors(node)):
#       to_assign.remove(node)
#       assigns[node] = Grey() if node in grey_nodes else 0
#       for succ in g.predecessors(node):
#         succ_values[succ].append(assigns[node])  
        
#   # compute mex values to graph
#   while to_assign:
#     for node in to_assign:
#       if len(list(g.successors(node))) == len(succ_values[node]):
#         to_assign.remove(node)
#         # everything computed, compute mex value
#         assigns[node] = mexes_grey(node, succ_values[node], grey_nodes)
#         print(node, assigns[node])
#         # communicate this new value to all its predecessors
#         for succ in g.predecessors(node):
#           succ_values[succ].append(assigns[node])  

#   print(assigns)
#   new_labels = {}
  
#   for node, mex in assigns.items():
#     if mex != Z:
#       new_labels[node] = f"{node}/{mex}"
#     else:
#       new_labels[node] = f"{node}/🌙" 
  
#   return nx.relabel_nodes(g, new_labels)  # change graph labels       
  