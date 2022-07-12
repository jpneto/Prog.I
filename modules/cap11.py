# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 15:28:50 2022

@author: jpn3t
"""
from abc import ABC, abstractmethod
from random import sample, randint, seed

from cap10 import ListMutStack, CircularArrayQueue

class BinTree(ABC):
  """ API para o tipo Árvore Binária """

  @abstractmethod
  def left(self):
    """returns the left subtree"""
    pass

  @abstractmethod
  def right(self):
    """returns the right subtree"""
    pass

  @abstractmethod
  def data(self):
    """returns the data at root"""
    pass
  
class MutBinTree(BinTree):
  """ implementação mutável de uma árvore binária """
  
  def __init__(self, data, left=None, right=None):
    self._data  = data
    self._left  = left 
    self._right = right

  @property
  def left(self):
    return self._left

  @left.setter
  def left(self, value):
    self._left = value

  @property
  def right(self):
    return self._right

  @right.setter
  def right(self, value):
    self._right = value

  @property
  def data(self):
    return self._data

  @data.setter
  def data(self, value):
    self._data = value
    
class LinkedBinTree(BinTree):
  """ implementação imutável de uma árvore binária """

  def __init__(self, data, left=None, right=None):
    self._data  = data
    self._left  = left 
    self._right = right

  @property
  def left(self):
    return self._left

  @property
  def right(self):
    return self._right

  @property
  def data(self):
    return self._data

  def __iter__(self):
    return LinkedBinTree.TreeIterator(self)

  class TreeIterator:
    """ classe interna responsável pelos iteradores """
    def __init__(self, tree):
      self._stack = ListMutStack()
      self._stack.push(tree)

    def __iter__(self):
      return self

    def __next__(self):
      if self._stack.isEmpty():
        raise StopIteration
      node = self._stack.peek()
      self._stack.pop()
      if node.right:
        self._stack.push(node.right)
      if node.left:
        self._stack.push(node.left)
      return node.data
    
class BST(MutBinTree):
  """ Implementação de uma árvore binária de pesquisa (BST)
      Os elementos a guardar têm de ser comparáveis, ie, 
      implementar o dunder __lt__.
      A invariante da classe determina que os objetos devem sempre
      representar uma BST """
      
  def __init__(self, data=None, left=None, right=None):
    self._data  = data
    self._left  = left 
    self._right = right

  def search(self, val):
    if self.data == val:
      return True
    if val < self._data:
      return self.left.search(val)
    else:
      return self.right.search(val)

  def insert(self, val):
    if self.data is None:
      self._data = val
    if val < self.data:
      if self.left is None:
        self._left = BST(val)
      else:
        self._left.insert(val)
    elif val > self.data:
      if self.right is None:
        self._right = BST(val)
      else:
        self._right.insert(val)

  def delete(self, val):
    if val < self.data and self.left:  # search value while left/right exists
      self._left = self.left.delete(val)
    elif val > self.data and self.right:
      self._right = self.right.delete(val)
    elif val == self.data:   # found value
      if self.left is None:  # only has right subtree
        return self.right
      if self.right is None: # only has left subtree
        return self.left
      # ok, the node has two children
      # let's find the next value (ie, the smallest from the right subtree)
      # place it here, and delete its old node
      min_node = self.right
      while min_node.left:   # go as left as possible
        min_node = min_node.left 
      self._data = min_node.data                     # place it here
      self._right = self.right.delete(min_node.data) # delete its old node
    return self
  
#################
  
def rndTree(size, rndSeed=None, xs=None, doOnce=True):
  """ gera recursivamente uma árvore binária aleatória """
  if size == 0:
    return None
  if doOnce:
    seed(rndSeed) # para reprodutibilidade
    xs = sample(range(5*size), size)
    doOnce = False

  size_left  = randint(0,size//2)
  size_right = size - size_left - 1
  return LinkedBinTree(xs[0], 
                       rndTree(size_left,  rndSeed, xs[1:size_left+1], doOnce), 
                       rndTree(size_right, rndSeed, xs[size_left+1: ], doOnce))  

def rndBST(size, rndSeed=None):
  """ gera uma BST aleatória com valores 0 a n-1 """
  seed(rndSeed) # para reprodutibilidade
  xs = sample(range(size), size)
  t = BST()  
  for x in xs:
    t.insert(x)
  return t

#################

def size(t):
  if t is None:
    return 0
  return 1 + size(t.left) + size(t.right)

def height(tree):
  if tree is None:
    return 0
  return 1 + max(height(tree.left), height(tree.right))

def occurrences(tree, item):
  if tree is None:
    return 0
  return (occurrences(tree.left,  item) +
          occurrences(tree.right, item) +
          (1 if tree.data == item else 0))

def preOrder(t, visit):
  if t:
    visit(t)
    preOrder(t.left,  visit)
    preOrder(t.right, visit)
    
def inOrder(t, visit):
  if t:
    inOrder(t.left,  visit)
    visit(t)
    inOrder(t.right, visit)

def postOrder(t, visit):
  if t:
    postOrder(t.left,  visit)
    postOrder(t.right, visit)
    visit(t)

def breathOrder(t, visit):
  q = CircularArrayQueue()
  q.enqueue(t)

  while not q.isEmpty():
    node = q.front()
    visit(node)
    q.dequeue()
    if node.left:
      q.enqueue(node.left)
    if node.right:
      q.enqueue(node.right)    

#################

from graphviz import Digraph # https://graphviz.readthedocs.io

def showTree(tree, styleEmpty='invis'):
  # adaptado de https://h1ros.github.io/posts/introduction-to-graphviz-in-jupyter-notebook/ 
  styleGraph = {'nodesep':'.25', 'ranksep':'.2'}
  styleNode  = {'shape':'circle', 'width':'.3', 'fontsize':'10', 'fixedsize':'True'}
  styleEdge  = {'arrowsize':'.6'} 
  def make(tree, nullIdx, styleEmpty, dot=None):
    if dot is None:
      dot = Digraph(graph_attr=styleGraph, node_attr=styleNode, edge_attr=styleEdge)
      dot.node(str(tree), str(tree.data))

    if tree.left:
      dot.node(str(tree.left), str(tree.left.data))
      dot.edge(str(tree), str(tree.left))
      dot = make(tree.left, nullIdx, styleEmpty, dot)
    else: # imprimir sub-árvores vazias invisiveis, para melhorar o aspeto final da árvore
      dot.node(str(tree.left)+str(nullIdx[0]), '', {'style':styleEmpty, 'width':'.1'})
      dot.edge(str(tree), str(tree.left)+str(nullIdx[0]), color='transparent' if styleEmpty=='invis' else 'blue', minlen='1')
      nullIdx[0]+=1

    if tree.right:
      dot.node(str(tree.right), str(tree.right.data))
      dot.edge(str(tree), str(tree.right))
      dot = make(tree.right, nullIdx, styleEmpty, dot)
    else: 
      dot.node(str(tree.right)+str(nullIdx[0]), '', {'style':styleEmpty, 'width':'.1'})
      dot.edge(str(tree), str(tree.right)+str(nullIdx[0]), color='transparent' if styleEmpty=='invis' else 'blue', minlen='1')
      nullIdx[0]+=1

    return dot
  
  dot = make(tree, [0], styleEmpty)
  #display(dot) # comentar se não estiver nos notebooks
  return dot

###########################

def sameStructure(t1, t2):
  if not t1 or not t2:       # if either is empty
    return not t1 and not t2 #  returns True iff both are empty  
  return sameStructure(t1.left, t2.left) and sameStructure(t1.right, t2.right)

def equals(t1, t2):
  if not t1 or not t2:       # if either is empty
    return not t1 and not t2 #  returns True iff both are empty
  return (t1.data == t2.data and 
          sameStructure(t1.left, t2.left) and 
          sameStructure(t1.right, t2.right))

def copy(t):
  if t is None:
    return None
  copyLeft  = copy(t.left)
  copyRight = copy(t.right)
  return LinkedBinTree(t.data, copyLeft, copyRight)

def mirror(t):
  if t is None:
    return None
  return LinkedBinTree(t.data, mirror(t.right), mirror(t.left))

def isBalanced(t):
  if t is None:
    return True
  h1, h2 = height(t.left), height(t.right)
  return abs(h1-h2) <= 1 and isBalanced(t.left) and isBalanced(t.right)

def isBST(t):
  if t is None:
    return True
  if (t.left and t.left.data > t.data) or (t.right and t.right.data < t.data):
    return False
  return isBST(t.left) and isBST(t.right)

space  = '    '
branch = '│   '
first  = '├── '
last   = '└── '

def printTree(t, prefix='', isLast=True):
  print(prefix, end='')
  if isLast:
    print(last, end='')
    prefix += space
  else:
    print(first, end='')
    prefix += branch
  print(t.data)
  if t.left:
    printTree(t.left, prefix, False)
  if t.right:
    printTree(t.right, prefix, True)

def fromList(xs, i=0):
  """ build tree from list description (use None for unoccupied positions) """
  if i>=len(xs) or xs[i] is None:
    return None
  return LinkedBinTree(xs[i], 
                       fromList(xs, 2*i+1),
                       fromList(xs, 2*i+2))

def lastIdx(t, idx=0, maxIdx=None):
  if t is None:
    return -1
  if maxIdx is None:
    maxIdx = [0]
    
  maxIdx[0] = max(maxIdx[0], idx)
  if t.left:
    lastIdx(t.left,  2*idx+1, maxIdx)
  if t.right:
    lastIdx(t.right, 2*idx+2, maxIdx)
  return maxIdx[0]

def toList(t, i=0, result=None):
  if result is None:
    result = [None] * (lastIdx(t)+1)
  if t is None or i >= len(result):
    return []
  
  result[i] = t.data
  toList(t.left,  2*i+1, result)
  toList(t.right, 2*i+2, result)
  return result


  