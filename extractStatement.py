import sys
import xml.etree.ElementTree as ET
from nodeMap import node_dict
'''
根据函数体源码生成的xml提取statement子树序列
'''


# 带双亲节点的树节点
class treeNode:
    def __init__(self, ele, parent):
        self.parent = parent
        self.ele = ele


# 根据根节点提取AST
def extractSTBaseRoot(root):
    split_node_indices = set([node_dict.get('if'), node_dict.get('while'), node_dict.get('for'), node_dict.get('unit')])

    # 根据深度优先遍历得到的列表，提取statement子树
    def extractStatement(tree):
        statementList = []
        for block in tree:  # block 表示每棵树的子树
            if block.ele[0] in split_node_indices:
                statementList.append(block)
                if block.parent != None:
                    block.parent.remove(block.ele)
        return [ block.ele for block in statementList]

    # 深度优先遍历树，树的节点为带双亲节点的结构
    def createTreeDeepFirst(root, nodes, parent):
        nodes.append(treeNode(root, parent))
        for node in root:
            if isinstance(node, list):
                createTreeDeepFirst(node, nodes, root)

    treeDeepFirstList = []
    createTreeDeepFirst(root, treeDeepFirstList, None)
    statementList = extractStatement(treeDeepFirstList)
    return statementList
