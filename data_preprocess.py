import _pickle as pkl
from nodeMap import node_dict
from tqdm import tqdm

def treeToIndex(root):
    token = root.tag
    result = [node_dict[token]]
    for child in root:
        result.append(treeToIndex(child))
    return result


path = 'dataset/test.pkl'
dump_path='dataset/dataset.pkl'

with open(path, 'rb') as f:
    data = pkl.load(f)

with open(dump_path,"rb") as f:
    dataset=pkl.load(f)

for row in tqdm(data):
    label = row[0]
    root = row[1]
    index_tree=treeToIndex(root)
    dataset.append((label,index_tree))



with open(dump_path,"wb") as f:
    pkl.dump(dataset,f)
    