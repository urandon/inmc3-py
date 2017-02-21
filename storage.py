# -*- coding: utf-8 -*-
'''
Module contains storage for combination-value pairs
'''


class TreeStorage(object):

    data_key = 'd'

    def __init__(self, data_handled=False):
        self.root = {}
        self.size = 0
        self.data_handled = data_handled

    def __len__(self):
        return self.size

    def get_node(self, combo, root=None):
        if root is None:
            root = self.root
        node = root
        for idx in combo:
            if idx not in node:
                return None
            node = node[idx]
        return node

    def __getitem__(self, item):
        return self.get_node(item)

    def set_data(self, node, data):
        if self.data_key not in node:
            self.size += 1
        node[self.data_key] = data

    def add_node(self, combo, data=None, root=None):
        if root is None:
            root = self.root
        node = root
        for idx in combo:
            node = node.setdefault(idx, {})
        self.set_data(node, data)
        return node

    def append(self, combo, data=None):
        self.add_node(combo, data=data)

    def __setitem__(self, key, value):
        self.add_node(key, data=value)

    def __iter__(self):
        if self.data_handled:
            return self.iteritems()
        else:
            return self.iterkeys()

    def iterkeys(self, combo=None, root=None):
        if root is None:
            root = self.root
        if combo is None:
            combo = []
        for idx, node in root.iteritems():
            if idx == self.data_key:
                yield combo
            else:
                for ret in self.iterkeys(combo + [idx], node):
                    yield ret

    def iteritems(self, combo=None, root=None):
        if root is None:
            root = self.root
        if combo is None:
            combo = []
        for idx, node in root.iteritems():
            if idx == self.data_key:
                yield (combo, node)
            else:
                for ret in self.iteritems(combo + [idx], node):
                    yield ret

    def join(self, storage, filterfunc=lambda x: True):
        if storage.data_handled and self.data_handled:
            for (combo, data) in storage:
                if filterfunc((combo, data)):
                    self.add_node(combo, data=data)
        elif storage.data_handled:
            for (combo, data) in storage:
                if filterfunc((combo, data)):
                    self.add_node(combo)
        else:
            for combo in storage:
                if filterfunc(combo):
                    self.add_node(combo)
