# Implements a tree
# A node has attribute childen as a list (empty by default) and attribute
# data

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def print_node(self, level=0):
        "Print the tree structure"
        print level * '| ' + str(self.data)
        for child in self.children:
            child.print_node(level + 1)
