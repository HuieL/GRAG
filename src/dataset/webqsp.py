import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.utils.graph_retrieval import retrive_on_graphs
from torch_geometric.data.data import Data
import warnings

warnings.filterwarnings("ignore")
model_name = 'sbert'
path = 'dataset/webqsp'
# Original graphs and related node & edge text attributes ... 
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'

# Retrieved components ...
cached_graph = f'{path}/cached_graphs'
cached_desc = f'{path}/cached_desc'

class WebQSPDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

def preprocess(topk=10, k = 2, topk_entity = 10, augment = "none"):
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    path_sims = f'{path}/sims_{k}hop_{augment}'
    os.makedirs(path_sims, exist_ok=True)

    dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        graph = torch.load(f'{path_graphs}/{index}.pt')
        nodes = pd.read_csv(f'{path_nodes}/{index}.csv')
        edges = pd.read_csv(f'{path_edges}/{index}.csv')
        q_emb = q_embs[index]

        try:
            sims = torch.load(f'{path_sims}/{index}.pt')
            subg, desc = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k = k, topk_entity = topk_entity, augment = augment, sims = sims)
        except:
            sims, subgraph = retrive_on_graphs(graph, q_emb, nodes, edges, topk=topk, k = k, topk_entity = topk_entity, augment = augment)
            subg, desc = subgraph
            torch.save(sims, f'{path_sims}/{index}.pt')

        data = Data(x = subg.x, 
                    edge_index = subg.edge_index, 
                    edge_attr = subg.edge_attr, 
                    question_node = q_emb.repeat(subg.x.size(0), 1),
                    question_edge = q_emb.repeat(subg.edge_attr.size(0), 1) if subg.edge_attr is not None else None,
                    num_nodes=subg.num_nodes)
        
        torch.save(data, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)

if __name__ == '__main__':

    preprocess(topk=20, k = 2, topk_entity = 10, augment = "none")
    dataset = WebQSPDataset()

