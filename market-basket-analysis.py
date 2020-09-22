
import numpy as np
import pandas as pd
import csv
import sys
sys.path.append('/Users/samcaekaert/Desktop/fall2020/csci-550-fall2020/cs550-hw1/')
import prefixtree
import itertools 
import pprint
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='Short sample app')

    parser.add_argument('--dataset', action="store", default="book_example.csv", required=False)
    parser.add_argument('--transaction-column', action="store", default = 0)
    parser.add_argument('--item-column', action="store", default = 1)
    parser.add_argument('--minsup', action="store", default = 3)
    parser.add_argument('--minconf', action="store", default = 0.5)
    parser.add_argument('--num-rules', action="store", default = 10)
    parser.add_argument('--max-rows', action="store", default = False)

    args = parser.parse_args()

    mkt_data = np.array(open_file(args.dataset))    
    title_row = mkt_data[0,:]
    mkt_data = np.delete(mkt_data, (0), axis=0)

    transaction_id_column = args.transaction_column
    item_column = args.item_column
    minsup = int(args.minsup)
    min_conf = float(args.minconf)
    max_rows = int(args.max_rows)
    k = int(args.num_rules)

    binary_d = create_binary_representation(mkt_data, transaction_id_column, item_column)
    if max_rows:
        binary_d = binary_d.ix[:max_rows,:]

    i = binary_d.columns
    print(len(i))

    f_dict, support_dict = apriori_modified(binary_d, i, minsup)
    print("\nThe frequent itemsets for minsup=%s are: " % minsup)
    pprint.pprint(f_dict)

    f_set = []
    for f in f_dict.values():
        if f == minsup:
            continue
        f_set = f_set + f


    # print("support_dict is:")
    # print(support_dict)
    rules = association_rules(f_set, min_conf, support_dict, binary_d)

    for rule in rules:
        rule["rsup"] = rsup(rule, binary_d)
        rule["lift"] = lift(rule, binary_d)

    #print_rules(rules, binary_d, min_conf, minsup)

    cool_rules = rank_rules(rules, "lift", "confidence", k)
    #print_rules(cool_rules, binary_d, min_conf, minsup)

def rank_rules(rules, metric_1, metric_2, k):
    #rules: list of dictionaries
    #metric 1: string
    #metric_2: string
    #k: int. Number of rules to return
    #Returns: list of rules sorted by rank of metric_1, metric_2, and the size of the rule (|X| + |Y|)

    if not rules:
        print("There were no rules given the paramaters entered previously!")
        return False

    for rule in rules:
        rule["combined_metric"] = 0.5*rule.get(metric_1) + 0.5*rule.get(metric_2)
        rule["size"] = len(rule.get("X")) + len(rule.get("Y"))

    sorted_by_metric = sorted(rules, key=lambda k: k['combined_metric'], reverse=True) 
    sorted_by_metric_1 = sorted(rules, key=lambda k: k[metric_1], reverse=True) 

    prev_rule = rules[0]
    for rule in sorted_by_metric_1:
        rule["metric_1_rank"] = sorted_by_metric_1.index(rule)
        if rule.get(metric_1) == prev_rule.get(metric_1):
            rule["metric_1_rank"] = prev_rule.get("metric_1_rank")
        prev_rule = rule

    sorted_by_metric_2 = sorted(sorted_by_metric_1, key=lambda k: k[metric_2], reverse=True) 
    prev_rule = rules[0]
    for rule in sorted_by_metric_2:
        rule["metric_2_rank"] = sorted_by_metric_2.index(rule)
        if rule.get(metric_2) == prev_rule.get(metric_2):
            rule["metric_2_rank"] = prev_rule.get("metric_2_rank")
        prev_rule = rule

    sorted_by_metric_3 = sorted(sorted_by_metric_2, key=lambda k: k["size"], reverse=True) 
    prev_rule = rules[0]
    for rule in sorted_by_metric_3:
        rule["size_rank"] = sorted_by_metric_3.index(rule)
        if rule.get("size") == prev_rule.get("size"):
            rule["size_rank"] = prev_rule.get("size_rank")
        prev_rule = rule

    for rule in sorted_by_metric_3:
        rule["combined_rank"] = rule.get("metric_1_rank") + rule.get("metric_2_rank") + rule.get("size_rank")

    sorted_by_combined_rank = sorted(rules, key=lambda k: k['combined_rank']) 

    

    print("\nThe %s most interesting rules based on combined rank are:" % (k))
    for rule in sorted_by_combined_rank[0:k]:
        print("%s ---> %s, sup(Z)=%s, conf = %s, rsup = %s, lift = %s, %s-rank = %s, %s-rank = %s, size-rank = %s, combined_rank = %s" % (rule.get("X"), rule.get("Y"), rule.get("sup_z"), rule.get("confidence"), rule.get("rsup"), rule.get("lift"),metric_1,rule.get("metric_1_rank"), metric_2,rule.get("metric_2_rank"), rule.get("size_rank"), rule.get("combined_rank")))

    return sorted_by_metric[0:k]



def rsup(rule, D):
    return round(sup(rule.get("X") + rule.get("Y"), D) / len(D.index) , 4)

def rconf(rule, D):
    return round(sup(rule.get("X") + rule.get("Y"), D) / sup(rule.get("X"), D), 4)

def lift(rule, D):
    return round(rconf(rule, D) / (sup(rule.get("Y"), D) / len(D.index)),4)


def print_rules(rules, D, min_conf, minsup):
    print(D)
    print("")
    print("Printing association rules for the above data with confidence %s and minsupport of %s:" % (min_conf, minsup))
    print("")
    for rule in rules:
        print("%s ---> %s, sup(Z)=%s, conf = %s, rsup = %s, lift = %s" % (rule.get("X"), rule.get("Y"), rule.get("sup_z"), rule.get("confidence"), rule.get("rsup"), rule.get("lift")))

def association_rules(F, min_conf, support_dict, D):
    # Returns: a list of association rules according to:
    #   https://dataminingbook.info/book_html/chap8/book-watermark.html

    return_list = [] # Returns a list of dictionaries. Where each dict is a rule
    for z in F:
        if len(z) < 2:
            continue
        A = powerset_k(z, len(z))
        A.remove([])
        while A:
            a_to_support_dict = {}
            for val in A:
                val_str = str(val)
                a_to_support_dict[val_str] = support_dict.get(val_str)
            max_element_in_A = max(a_to_support_dict, key=a_to_support_dict.get)
            max_element_list = re.findall(r"'(.*?)'", max_element_in_A, re.DOTALL)
            A = [l for l in A if l != max_element_list]
            sup_z = sup(z, D)
            sup_X = sup(max_element_list, D)
            c = sup_z/sup_X
            Y = [item for item in z if item not in max_element_list]
            if c >= min_conf:
                #print("X is %s ---> Y is Z\X = %s, sup(Z) is %s, c is %s." % (max_element_list, Y, sup_z, c))
                return_list.append({
                    "X": max_element_list,
                    "Y": Y,
                    "sup_z": sup_z,
                    "confidence": c
                })
            else:
                for l in A:
                    if max_element_list in l:
                        A.remove(l)

    return return_list
        
def brute_force(binary_d, i, minsup):
    f = set()
    f_dict = {}
    print(i)
    print(type(i))
    i_list = [j for j in i]
    all_combinations = powerset_k(i_list, 2) # Too long for testing...

    print(all_combinations)
    exit(1)
    for item_set in all_combinations:
        sup_x = sup(item_set,binary_d)
        if sup_x >= minsup:
            f_dict[len(item_set)] = f_dict.get(len(item_set), []) + item_set
            f = f.union(set(item_set))

    print(f)
    print(f_dict)
    print(len(f_dict))
    for value in f_dict.values():
        print(len(value))

def apriori_modified(binary_d, i, minsup):
    # Function implements a modified apriori algorithm for generating frequent itemsets. 
    # This version does not use a prefix tree an instead builds a list of items based on the same logic
    #   See: https://dataminingbook.info/book_html/chap8/book-watermark.html

    # binary_d list: A list of lists containing mkt data
    # i (script I) list: The items. Can be thought of as the column names in D
    # minsup int: The minimum support threshold. 
    # Returns: F, the set of items with sup > minsup

    support_dict = {}
    f = set()
    f_dict = {"minsup": minsup}
    print(i)

    print("Generating the frequent itemsets for:")
    print(binary_d)
    #print(type(i))
    i_list = [j for j in i]
    max_itemset_size = 20

    layer_1 = powerset_only_k(i_list, 1)
    for set_size in range(max_itemset_size):
        combinations = powerset_only_k(i_list, 1)
        print("layer is %s" % set_size)
        #print("length of combinations is %i" % len(combinations))

        new_combination_list = []
        if set_size > 0:
            layer_1_minsup = f_dict.get(1)
            prev_layer_minsup = f_dict.get(set_size)
            #print("prev layer minsup is")
            if not prev_layer_minsup:
                return f_dict, support_dict
            for i,c1 in enumerate(layer_1_minsup):
                for j,c2 in enumerate(prev_layer_minsup):
                    if c1 != c2 and not set(c1).issubset(set(c2)):
                        if set_size == 1:
                            if i <= j:
                                new_combinations = c1 + c2
                                new_combination_list = new_combination_list + [new_combinations]

                        else:
                            new_combinations = c1 + c2
                            new_combination_list = new_combination_list + [new_combinations]

            combinations = new_combination_list
            
        if len(combinations) == 0:
            break

        for item_set in combinations:
            sup_x = sup(item_set, binary_d)
            support_dict[str(sorted(item_set))] = sup_x
            if sup_x >= minsup:
                f_dict[len(item_set)] = f_dict.get(len(item_set), []) + [item_set]
                f_dict[len(item_set)] = sorted(list(map(list, set(tuple(sorted(i)) for i in f_dict.get(len(item_set))))))


        prev_combinations = combinations





    print("support dict is")
    print(support_dict)

    return f_dict, support_dict

def powerset_k(s, k):
    return_list = []
    for j in range(k):
        current_list = [list(i) for i in itertools.combinations(s, j)]
        return_list = return_list + current_list

    return return_list

def powerset_only_k(s, k):
    current_list = [list(i) for i in itertools.combinations(s, k)]
    return current_list

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
            if leaf.get_children():
                continue
            
            if leaf.get_sup() > minsup:
                F = F.union(set(leaf.get_item()))
                #return_f[k] = F.union(set(leaf.get_item())) - return_f.get(k)
            else:
                print("k is %s. Node %s its support is %s" % (k, leaf.get_item(), leaf.get_sup()))
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

def create_binary_representation(mkt_data, transaction_id_column, item_column):
    # mkt_data: a list of lists
    # Returns binary matrix (D) from https://dataminingbook.info/book_html/chap8/book-watermark.html
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
    #   https://dataminingbook.info/book_html/chap8/book-watermark.html
    c_k = prefix_tree.get_nodes_in_layer(k)
    for t in D.iterrows():
        i_t = [D.columns[x] for x in range(len(t[1])) if t[1][x] == 1]
        #print("i_t is: ")
        #print(i_t)
        if k == 1:
            k_subsets = [[i] for i in i_t] 
        else: 
            k_subsets = generate_k_subsets(i_t, k)
        
        print("k_subsets is:")
        print(k_subsets)
        c_k_items = [node.get_item() for node in c_k]
        print("c_k_items is %s" % c_k_items)
        for X in k_subsets:
            print("X is")
            print(X)
            if X in c_k_items:
                node = prefix_tree.get_node_from_item(X)
                node.set_sup(node.get_sup() + 1)
                print("Support for %s is %s" % (node.get_item(), node.get_sup()))

def sup(X,D):
    #  https://dataminingbook.info/book_html/chap8/book-watermark.html
    col_index = [D.columns.get_loc(x) for x in X]
    sup = 0
    for index, row in D.iterrows():
        if sum(row[col_index]) == len(X):
            sup += 1

    #print("Getting sup of %s -- It is %s" % (X,sup))
    return(sup)

def generate_k_subsets(i_t, k):
    return [list(i) for i in itertools.combinations(i_t, k)]


if __name__ == "__main__":
    main()