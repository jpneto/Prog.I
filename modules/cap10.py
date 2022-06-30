# -*- coding: utf-8 -*-
from random import randint
from abc import ABC, abstractmethod

class Stack(ABC):
  """ API para o tipo Stack """

  @abstractmethod
  def isEmpty(self):
    """returns true iff the stack is empty"""
    pass

  @abstractmethod
  def push(self, item):
    """pushes an item onto the top of the stack"""
    pass

  @abstractmethod
  def peek(self):
    """requires: not isEmpty()
       returns the top item of the stack"""
    pass

  @abstractmethod
  def pop(self):
    """requires: not isEmpty()
       removes the top item on the stack"""
    pass

#####################################################################  

class ListMutStack(Stack):
  """ mutable implementation of Stack API using lists"""

  def __init__(self):
    """ returns an empty stack """
    self._items = list()

  def isEmpty(self):
    return len(self) == 0

  def peek(self):
    return self._items[-1]

  def pop(self):
    self._items.pop()

  def push(self, item):
    self._items.append(item)

  def __len__(self):
    """returns number of elements in stack"""
    return len(self._items)

  def __str__(self):
    """returns a string representation of the stack's state"""
    return str(self._items)[:-1] + '['

#####################################################################

class ListImutStack(Stack):
  """ immutable implementation of Stack API using lists"""

  def __init__(self):
    """ returns an empty stack """
    self._items = list()

  def isEmpty(self):
    return len(self) == 0

  def peek(self):
    return self._items[-1]

  def pop(self):
    st = ListImutStack()
    st._items = self._items[:]
    st._items.pop()
    return st

  def push(self, item):
    st = ListImutStack()
    st._items = self._items[:]
    st._items.append(item)
    return st

  def __len__(self):
    """returns number of elements in stack"""
    return len(self._items)

  def __str__(self):
    """returns a string representation of the stack's state"""
    return str(self._items)[:-1] + '['
  
#####################################################################

class Queue(ABC):
  """ API para o tipo Queue """

  @abstractmethod
  def isEmpty(self):
    """returns true iff queue is empty"""
    pass

  @abstractmethod
  def enqueue(self, item):
    """inserts item at queue's end"""
    pass

  @abstractmethod
  def front(self):
    """requires: not isEmpty()
       returns the item at queue's beginning"""
    pass

  @abstractmethod
  def dequeue(self):
    """requires: not isEmpty()
       removes the item at queue's beginning"""
    pass

#####################################################################

class CircularArrayQueue(Queue):
  """ representing a queue with a circular array """

  def __init__(self, capacity=6):
    self._queue = [None]*capacity
    self._begin = 0
    self._end = 0
    self._size = 0

  def isEmpty(self):
    return self._size == 0

  def enqueue(self, item):
    if self._size == len(self._queue): # queue is full, double array size
      self._reallocate()
    self._queue[self._end] = item
    self._end = self._inc(self._end)
    self._size += 1

  def front(self):
    return self._queue[self._begin]

  def dequeue(self):
    self._queue[self._begin] = None
    self._begin = self._inc(self._begin)
    self._size -= 1

  def _inc(self, n):
    """ increment by 1, using modular arithmetic """
    return (n+1)%len(self._queue)

  def _reallocate(self):
    newQueue = [None] * (2*self._size)
    j = self._begin
    for i in range(self._size):
      newQueue[i] = self._queue[j]
      j = self._inc(j)
    self._begin = 0
    self._end   = self._size
    self._queue = newQueue

  def __len__(self):
    return self._size

  def __str__(self):
    result = []
    j = self._begin
    for i in range(self._size):
      result.append(self._queue[j])
      j = self._inc(j)
    return '<'+str(result)[1:-1]+'<'

#####################################################################

class TestMutStack(ListMutStack):
  """ subclasse de ListMutStack, esta classe inclui igualdade e clonagem, 
      para efeitos de teste """
  def __eq__(self, st):
    """ verifica se self == st """
    st1, st2 = self.copy(), st.copy()
    while not st1.isEmpty() and not st2.isEmpty():
      if st1.peek() != st2.peek():
        return False
      st1.pop()
      st2.pop()
    return st1.isEmpty() and st2.isEmpty()

  def copy(self):
    """ cria uma cópia de self, ie, um novo objecto com o mesmo estado """
    st = TestMutStack()
    st._items = self._items[:]
    return st

def rndStack(maxSize=32, maxElem=1000):
  """ gera uma stack para teste, com conteúdo aleatório """
  size = randint(0, maxSize)
  st = TestMutStack()
  for x in [randint(-maxElem, maxElem) for _ in range(size)]:
    st.push(x)
  return st

#####################################################################

class Deque(ABC):
  """ API para o tipo Deque """

  @abstractmethod
  def isEmpty(self):
    """returns true iff deque is empty"""
    pass

  @abstractmethod
  def first(self):
    """requires: not isEmpty()
       returns deque's first element"""
    pass

  @abstractmethod
  def last(self):
    """requires: not isEmpty()
       returns deque's last element"""
    pass

  @abstractmethod
  def addFirst(self, item):
    """adds item at deque's front"""
    pass

  @abstractmethod
  def addLast(self, item):
    """adds item at deque's back"""
    pass

  @abstractmethod
  def delFirst(self):
    """requires: not isEmpty()
       removes deque's first element"""
    pass

  @abstractmethod
  def delLast(self):
    """requires: not isEmpty()
       removes deque's last element"""
    pass

#####################################################################  

class ListDeque(Deque):
  def __init__(self):
    self._items = []

  def isEmpty(self):
    return self._items == []

  def first(self):
    return self._items[0]

  def last(self):
    return self._items[-1]

  def addFirst(self, item):
    self._items.insert(0, item)

  def addLast(self, item):
    self._items.append(item)

  def delFirst(self):
    self._items.pop(0)

  def delLast(self):
    self._items.pop()

  def __str__(self):
    return ']'+str(self._items)[1:-1]+'['  

#####################################################################  

class DLLDeque(Deque):
  class Node:
    """internal helper class, represents a node object"""
    def __init__(self, item, prev, next):
      self.data = item
      self.prev = prev
      self.next = next

  def __init__(self):
    self._head = None
    self._tail = None

  def isEmpty(self):
    return self._head is None

  def first(self):
    return self._head.data

  def last(self):
    return self._tail.data

  def addFirst(self, item):
    self._head = DLLDeque.Node(item, None, self._head)
    if self._tail is None:    # se deque é vazia
      self._tail = self._head
    else:
      self._head.next.prev = self._head   

  def addLast(self, item):
    self._tail = DLLDeque.Node(item, self._tail, None)
    if self._head is None:
      self._head = self._tail
    else:
      self._tail.prev.next = self._tail

  def delFirst(self):
    if self._head == self._tail:  # se deque tem um só elemento
      self._head = self._tail = None
    else:
      self._head = self._head.next
      self._head.prev = None

  def delLast(self):
    if self._head == self._tail:
      self._head = self._tail = None
    else:
      self._tail = self._tail.prev
      self._tail.next = None

  def __str__(self):
    result, node = [], self._head
    while node is not None:
      result.append(node.data)
      node = node.next
    return ']'+str(result)[1:-1]+'['
  
#####################################################################  

