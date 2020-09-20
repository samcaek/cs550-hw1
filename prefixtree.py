class PrefixTree:
    def __init__(self):
        self.root = PrefixNode(["Null"], self, [], 0, 0, None)
        self.c = [[self.root]]

    def get_root(self):
        return self.root

    def get_nodes_in_layer(self, k):
        if len(self.c) > k :
            return self.c[k]
        else:
            return False

    def get_c(self):
        return self.c

    def set_c(self, new_c):
        self.c = new_c

    def get_items_in_layer(self, layer):
        item_list = []
        for node in self.c[layer]:
            for item in node.get_item():
                item_list.append(item)
        return list(set(item_list))

    def get_node_from_item(self, item):
        #print("Searching tree for node corresponding to: %s" % item)
        return self.root.find_node_by_name(item)


    def remove_node(self, node):
        print("Removing %s" % node.get_item())
        self.c = [[ele for ele in sub if ele != node] for sub in self.c] 
        self.root.remove_node_from_node_class(node)
                
        

class PrefixNode:
    def __init__(self, item, tree,  children, sup, layer, parent, count=0):
        self.item = item
        self.tree = tree
        self.children = []
        self.sup = 0
        self.layer = layer
        self.count = count
        self.parent = parent
        # if self.parent:
        #     self.parent.add_child(self)


    def add_child(self, child):
        self.children.append(child)
        self.count += 1
        tree = self.tree
        c = self.tree.get_c()
        if len(c) == self.layer + 1:
            c.append([])
        c[self.layer + 1].append(child)
        self.tree.set_c(c)

    def get_children(self):
        return self.children

    def get_count(self):
        return self.count

    def set_count(self, new_count):
        self.count = new_count

    def get_sup(self):
        return self.sup
    
    def set_sup(self, new_sup):
        self.sup = new_sup
        
    def get_item(self):
        return self.item
    
    def set_item(self, new_item):
        self.item = new_item

    def get_items_of_children(self):
        return [child.get_item for child in self.children]

    def get_parent(self):
        return self.parent

    def find_node_by_name(self, item):
        #print(self.get_item())
        if self.get_item() == item:
            return self
        else:
            for child in self.children:
                match = child.find_node_by_name(item)
                if match:
                    return match
    

    def remove_node_from_node_class(self, node):
        if self.get_item() == node.get_item():
            self.get_parent().remove_child(self)
            return self
        else:
            for child in self.children:
                match = child.remove_node_from_node_class(node)
                if match:
                    return match

    def get_siblings(self):
        children = self.get_parent().get_children()
        return [node for node in children if node != self]

    def remove_child(self, node):
        self.children.remove(node)