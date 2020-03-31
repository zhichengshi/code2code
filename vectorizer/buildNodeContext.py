import sys
import xml.etree.ElementTree as ET
import _pickle as pkl
import copy
import os 
from tqdm import tqdm
# 带双亲节点的树节点
class treeNode:
    def __init__(self, parent=None, current=None):
        self.parent = parent
        self.current = current  # 该变量的数据结构和原AST中的结构一样

# 构造情景向量数据集
def generateContextDataset(nodes, upSize, downSize, siblingUsing):
    # 获得当前节点的所有情景节点
    def generateContext(node, upSize, downSize, siblingUsing):
        contextNodes = set()
        contextNodes = findDescendants(contextNodes, node.current, downSize, 0)  # 加入子孙节点
        n = 0
        parent = node.parent
        while n < upSize and parent != None:  # 加入祖先节点
            contextNodes.add(parent.current[0])
            parent = parent.parent
            n += 1

        if siblingUsing == True and node.parent != None:  # 判断是否使用兄弟节点，如果使用则在情景向量中加入兄弟节点
            for child in node.parent.current[1:]:
                contextNodes.add(child[0])

        return contextNodes

    # 根据窗口大小，获得当前节点所有的子孙节点
    def findDescendants(result, node, maxdepth, depth):
        if depth >= maxdepth:
            return set()

        # 将当前节点的孩子节点放入result
        for child in node[1:]:
            result.add(child[0])

        # 将孩子节点的孩子节点放入result
        for child in node[1:]:
            descendantNodes = findDescendants(result, child, maxdepth, depth + 1)
            result = result | descendantNodes #求并集

        return result

    contextDataset = []
    for node in nodes:  # 遍历树上的所有节点，构造出所有节点的情景向量对
        contextNodes = generateContext(node, upSize, downSize, siblingUsing)
        for contextNode in contextNodes:
            if node.current[0]!=contextNode:
                contextDataset.append((node.current[0], contextNode))

    return contextDataset

def preprocessAst(root):

    # 给所有节点加上双亲节点
    def reconstructTree(parent, current, nodes):
        nodes.append(treeNode(parent, current))
        for child in current[1:]:
            parent = treeNode(parent, current)
            reconstructTree(parent, child, nodes)
        return nodes

    nodes = reconstructTree(None, root, [])
    return nodes


if __name__ == "__main__":
    path_dataset='dataset/dataset.pkl'
    path_context="dataset/context.pkl"

    with open(path_dataset, "rb") as f:
        dataset = pkl.load(f)

    contextDataset = []
    for root in tqdm(dataset):
        nodes = preprocessAst(root[1])
        contextDataset += generateContextDataset(nodes, 1, 2, True)

    print("情景对的个数是：",len(contextDataset))
    with open(path_context,"wb") as f:
        pkl.dump(contextDataset,f)
