# -*- coding: utf-8 -*-
"""Module contains storage for combination-value pairs"""

class TreeStorage(object):
    """ Tree-hashmap based storage.
    Nodes correspond to feature indexes
    Supposed that each index in nonnegative int
    """

    data_key = -1

    def __init__(self):
        self.root = {}
        self.size = 0

    def len(self):
        return self.size

    def set_data(self, node, data):
        if not node.has_key(self.data_key):
            self.size += 1
        node[self.data_key] = data

    def get_data(self, node):
        return node[self.data_key]

    def __getitem__(self, item):
        return self.get_node(item)

    def get_node(self, combo, root=None):
        if root is None: root = self.root
        node = root
        for fidx in combo:
            if not node.has_key(fidx):
                return None
            node = node[fidx]
        return node

    def __setitem__(self, key, value):
        if value is None:
            node = self.add_node(key, data=None)
            if node.has_key(self.data_key):
                node.pop(self.data_key)
                self.size -= 1
        else:
            self.add_node(key, data=value)

    def add_node(self, combo, data=None, root=None):
        if root is None: root = self.root
        node = root
        for fidx in combo:
            node = node.setdefault(fidx, {})
        if data is not None:
            if not node.has_key(self.data_key):
                self.size += 1
            # else: print 'node exist already'
            node[self.data_key] = data
        return node

    def __iter__(self): # preorder-iterator
        for gen in self._preorder_generator(self.root, []):
            yield gen

    def _preorder_generator(self, node, combo):
        for (fidx, value) in node.iteritems():
            if fidx == self.data_key:
                yield (combo, value)
            else:
                combo.append(fidx)
                for gen in self._preorder_generator(value, combo):
                    yield gen
                combo.pop()

    def join(self, storage):
        for (combo, data) in storage:
            self[combo] = data
