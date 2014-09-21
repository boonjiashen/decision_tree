# Implements a tree
# A node has attribute childen as a list (empty by default) and attribute
# data

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)
