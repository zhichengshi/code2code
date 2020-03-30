"""Functions to help with sampling trees."""

import pickle
import numpy as np
from extractStatement import extractSTBaseRoot
import random
from config import label_size
import xml.etree.ElementTree as ET
from nodeMap import node_dict
from tqdm import tqdm

'''
trees: Statement AST list
vector:node embedding matrix
vector_lookup:search the node num
'''


def process_one_tree(trees, vectors, vector_lookup):  # 每一棵tree中的节点第一个值是当前节点的索引，其他元素是该节点的孩子节点
    nodes_batch = []
    children_batch = []
    for tree in trees:
        nodes = []
        children = []

        queue = [(tree, -1)]

        # level visit
        while queue:
            node, parent_ind = queue.pop(0)

            # neglect the comment node
            if node[0] == node_dict['comment']:
                continue

            node_index = len(nodes)
            # add children and the parent index to the queue
            if len(node)-1 > 0:
                queue.extend(zip(node[1:], [node_index]*(len(node)-1)))
            
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_index)
            nodes.append(vectors[vector_lookup[node[0]]])

        nodes_batch.append(nodes)
        children_batch.append(children)

    return _pad(nodes_batch, children_batch)


def gennerate_sample_from_list(path, vectors, vector_lookup):
    with open(path, "rb") as f:
        datas = pickle.load(f)
        random.shuffle(datas)

        for data in tqdm(datas):
            label = data[0]
            tree = data[1]

            subtrees = extractSTBaseRoot(tree)
            # process one tree which has been splited into statement trees
            nodes, children, max_children_size = process_one_tree(subtrees, vectors, vector_lookup)

            if max_children_size > 1000:
                continue

            # 根据标记获得对应于该标记的向量
            label_vector = np.eye(label_size, dtype=int)[int(label) - 1]

            yield [nodes], [children], [len(subtrees)], [label_vector]


#
def _pad(nodes, children):
    if not nodes:
        return [], [], []
    # 树的最多节点个数
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])

    # 最大孩子节点的个数
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children, max_children
