import numpy as np
import pandas as pd
import csv
import sys
sys.path.append('/Users/samcaekaert/Desktop/fall2020/csci-550-fall2020/cs550-hw1/')
import prefixtree
import itertools 
  


def main():
    mkt_data = np.array(open_file("txn_by_dept.csv"))
    title_row = mkt_data[0,:]
    mkt_data = np.delete(mkt_data, (0), axis=0)

    transaction_id_column = 0
    item_column = 1

    binary_d = create_binary_representation(mkt_data, transaction_id_column, item_column)

    i = binary_d.columns
    minsup = 50
    f = apriori(binary_d, i, minsup)
    print(f)

def open_file(file_name):
    with open(file_name, newline='') as csvfile:
        return list(csv.reader(csvfile))

def apriori(D, I, minsup):
    # Function implements the apriori algorithm for generating frequent itemsets. 
    # param mkt_data list: A list of lists containing mkt data
    # I (script I) list: The items. Can be thought of as the column names in D
    # minsup int: The minimum support threshold. 
    # Returns: F, the set of items with sup > minsup

    F = set()
    return_f = {}
    C = prefixtree.PrefixTree()
    #    print(C.get_root().get_val())
    for i in I:
        root_node = C.get_root()
        new_node = prefixtree.PrefixNode([i], C, [], 0, 1, root_node)
        root_node.add_child(new_node)

    k = 1
    c_k = C.get_nodes_in_layer(1)


    #
    #print(c_k)

    # Line 5-9
    while C.get_nodes_in_layer(k):
        compute_support(C, D, k)
        c_k = C.get_nodes_in_layer(k)
        for leaf in c_k:
            # if leaf.get_children():
            #     continue
            
            if leaf.get_sup() > minsup:
                F = F.union(set(leaf.get_item()))
                #return_f[k] = F.union(set(leaf.get_item())) - return_f.get(k)
            else:
                print("k is %s" % k)
                C.remove_node(leaf)

        # Line 10, 11
        extend_prefix_tree(C, k)
        k += 1

        print("end of while loop")
        print([len(c) for c in C.get_c()])

    return F


def extend_prefix_tree(C, k):
    c_k = C.get_nodes_in_layer(k)
    for leaf_a in c_k:
        for leaf_b in leaf_a.get_siblings(): # Such that b > a...
            X_ab = list(set(leaf_a.get_item()).union(leaf_b.get_item()))
            # Line 19
            x_j = proper_subset(X_ab)
            print("x_ab is %s -- x_j is %s" % (X_ab, x_j))
            my_b = True
            for x in x_j:
                
                if not all(True if x.count(item) <= C.get_items_in_layer(k).count(item) else False for item in x):
                    print("Didnt find %s in %s" % (x,  C.get_items_in_layer(k)))
                    my_b = False
            if my_b:
                leaf_a_b = prefixtree.PrefixNode(X_ab, C, [], 0, k+1, leaf_a)
                print("Adding leaf_a_b as %s" % leaf_a_b.get_item())
                leaf_a.add_child(leaf_a_b)
        # Line 21
        if not leaf_a.get_children():
            print("%s has no extensions. Deleting" % leaf_a.get_item())
            C.remove_node(leaf_a)



def proper_subset(items_ab):
    # Returns X_j from line 19 of the apriori algorithm. 
    if len(items_ab) == 2:
        return [[i] for i in items_ab]
    else:
        return generate_k_subsets(items_ab, len(items_ab) - 1)

def strong_rule_set():
    pass

def rank_rules():
    pass



def create_binary_representation(mkt_data, transaction_id_column, item_column):
    # mkt_data: a list of lists
    # Returns D from apriori in DM chapter 8.
    transactions = mkt_data[:,transaction_id_column]
    unique_transactions = list(set(transactions))
    unique_transactions.sort()
    items = mkt_data[:,item_column]
    unique_items = set(mkt_data[:,item_column])
    unique_items = list(unique_items)
    unique_items.sort()

    binary_matrix = np.zeros((len(unique_transactions), len(unique_items)))
    for i,transaction in enumerate(unique_transactions):
        for row in mkt_data:
            if transaction == row[transaction_id_column]:
                index = unique_items.index(row[item_column])
                #binary_matrix[i,index] = binary_matrix[i,index] + 1
                binary_matrix[i,index] = 1

    matrix = pd.DataFrame(data=binary_matrix, columns=unique_items, index=unique_transactions)

    return matrix

def compute_support(prefix_tree, D, k):
    c_k = prefix_tree.get_items_in_layer(k)
    for t in D.iterrows():
        i_t = [D.columns[x] for x in range(len(t[1])) if t[1][x] == 1]
        #print("i_t is: ")
        #print(i_t)
        if k == 1:
            k_subsets = i_t
        else: 
            k_subsets = generate_k_subsets(i_t, k)
        
        for X in k_subsets:
            if X in c_k:
                node = prefix_tree.get_node_from_item([X])
                node.set_sup(node.get_sup() + 1)
                print("Support for %s is %s" % (node.get_item(), node.get_sup()))

def sup(X,D):
    col_index = [D.columns.get_loc(x) for x in X]
    sup = 0
    for index, row in D.iterrows():
        if sum(row[col_index]) == len(X):
            sup += 1

    print("Getting sup of %s -- It is %s" % (X,sup))
    return(sup)

def generate_k_subsets(i_t, k):
    return list(itertools.combinations(i_t, k)) 


if __name__ == "__main__":
    main()