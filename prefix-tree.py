class PrefixTree:
    def __init__(self):
        self.root = PrefixNode(None, [])

    def get_root(self):
        return self.root

class PrefixNode:
    def __init__(self, val, children):
        self.val = val
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_val(self):
        return self.val