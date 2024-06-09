import torch
import networkx as nx
from joblib import Parallel, delayed
from torch_geometric.data.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.lm_modeling import load_model, load_text2embedding
from src.utils.text_graph import hard_prompt

llm = "meta-llama/Llama-2-13b-hf"
sentence_lm = 'sbert'
sentence_model, sentence_tokenizer, _ = load_model[sentence_lm]()
text2embedding = load_text2embedding[sentence_lm]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_path(graph, start_node, end_node):
    try:
        path = nx.shortest_path(graph, source=start_node, target=end_node)
        return path
    except nx.NetworkXNoPath:
        return None

def get_triplets(subgraph):
    if subgraph.edge_attr is None:
        return torch.mean(subgraph.x, dim=0)
    combined_attr = torch.cat([subgraph.x, subgraph.edge_attr.view(-1, 1024)], dim=0)
    return torch.mean(combined_attr, dim=0)

def get_trunk_triplets(subgraph_edge_indices, textual_nodes, textual_edges, edges_mapping):
    if subgraph_edge_indices[0] == subgraph_edge_indices[1]: 
        return textual_nodes[subgraph_edge_indices[0][0]]
    flatten_graph = ""
    edges_mapping_dict = {pair: idx for idx, pair in enumerate(edges_mapping)}

    node_texts = textual_nodes["node_attr"].to_numpy()
    edge_texts = textual_edges["edge_attr"].to_numpy()

    flatten_graph_components = []
    # Iterate over subgraph_edge_indices
    for head, tail in zip(subgraph_edge_indices[0], subgraph_edge_indices[1]):
        edge_index = edges_mapping_dict.get((head, tail))
        if edge_index is not None:
            flatten_graph_components.append(", ".join([
                node_texts[head], edge_texts[edge_index], node_texts[tail]
            ]))

    # Join all components at the end
    flatten_graph = ", ".join(flatten_graph_components)
    return flatten_graph

def get_augmented_triplets(path_model, path_tokenizer, subgraph_edge_indices, textual_nodes, textual_edges, edges_mapping):
    node_texts = textual_nodes["node_attr"].to_numpy()
    edge_texts = textual_edges["edge_attr"].to_numpy()
    if subgraph_edge_indices[0] == subgraph_edge_indices[1]: 
        return textual_nodes[subgraph_edge_indices[0][0]]

    flatten_graph = ""
    edges_mapping_dict = {pair: idx for idx, pair in enumerate(edges_mapping)}
    subgraph_edge_indices = subgraph_edge_indices.tolist()

    # Iterate over subgraph_edge_indices
    for head, tail in zip(subgraph_edge_indices[0], subgraph_edge_indices[1]):
        edge_index = edges_mapping_dict.get((head, tail))
        if edge_index is not None:
            prompts = "Write a description for given triplet and contain all given words: "
            prompts += (", ".join([
                node_texts[head], edge_texts[edge_index], node_texts[tail]
            ]))

            inputs = path_tokenizer.encode(prompts, return_tensors='pt').to(device)
            outputs = path_model.generate(inputs, max_length=128, num_return_sequences=1)
            textual_path = path_tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            flatten_graph += textual_path.split("\n")[-1]   

    return flatten_graph

def process_path(path, textual_nodes, textual_edges, edges_mapping_dict, path_tokenizer, path_model, max_token, device):
    prompts = "Write a description for given triplet and contain all given words: "
    for i in range(len(path) - 1):
        node_text = textual_nodes[path[i]]
        edge_text = textual_edges[edges_mapping_dict[(path[i], path[i + 1])]]
        next_node_text = textual_nodes[path[i + 1]]
        prompts += ", ".join([node_text, edge_text, next_node_text])

    inputs = path_tokenizer.encode(prompts, return_tensors='pt').to(device)
    outputs = path_model.generate(inputs, max_length=max_token, num_return_sequences=1)
    textual_path = path_tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return textual_path.split("\n")[-1]

def get_augmented_path(path_model, path_tokenizer, paths, textual_nodes_df, textual_edges_df, edges_mapping):
    textual_nodes = textual_nodes_df["node_attr"].tolist()
    textual_edges = textual_edges_df["edge_attr"].tolist()
    edges_mapping_dict = {pair: idx for idx, pair in enumerate(edges_mapping)}

    # Parallel processing of paths
    results = Parallel(n_jobs=-1)(delayed(process_path)(
        path=paths[pair], 
        textual_nodes=textual_nodes, 
        textual_edges=textual_edges, 
        edges_mapping_dict=edges_mapping_dict, 
        path_tokenizer=path_tokenizer, 
        path_model=path_model, 
        max_token=2048, 
        device="cuda"  # Assuming using GPU
    ) for pair in paths.keys())

    flatten_graph = " ".join(results)
    return flatten_graph

def merge_graphs(graph_list, q_emb):
    if len(graph_list) == 1:
        return graph_list[0]
    
    x, edge_index, edge_attr = [], [], []
    cum_nodes = 0 

    for graph in graph_list:
        x.append(graph.x)
        if hasattr(graph, 'edge_index') and graph.edge_index is not None:
            new_edge_index = graph.edge_index + cum_nodes
            edge_index.append(new_edge_index)
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                edge_attr.append(graph.edge_attr)

        cum_nodes += graph.num_nodes

    x = torch.cat(x, dim=0)
    question_node = q_emb.unsqueeze(0).repeat(x.shape[0], 1)

    edge_index = torch.cat(edge_index, dim=1) if edge_index else torch.empty((2, 0), dtype=torch.long)
    if edge_attr:
        if edge_attr[0].dim() == 2:
            edge_attr = torch.cat(edge_attr, dim=0)
            question_edge = q_emb.repeat(edge_attr.shape[0], 1)
        elif edge_attr[0].dim() == 3:
            edge_attr = torch.cat(edge_attr, dim=1)
            question_edge = q_emb.repeat(2, edge_attr.shape[1], 1)
        else:
            raise ValueError("Unexpected edge attribute dimensionality.")
        
    merge_graph = Data(x=x, edge_index=edge_index, question_node=question_node)
    if merge_graph.edge_index.size(1) == 0:
        num_nodes = merge_graph.num_nodes
        self_loops = torch.arange(num_nodes, dtype=torch.long)
        merge_graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
        edge_attr = torch.zeros((num_nodes, q_emb.shape[0]), dtype=q_emb.dtype)
        question_edge = torch.zeros((num_nodes, q_emb.shape[0]), dtype=q_emb.dtype)  # no edge attributes

    merge_graph.edge_attr = edge_attr
    merge_graph.question_edge = question_edge

    return merge_graph

def find_topk_subgraph(graph, q_emb, top_k_indices, edges_mapping, textual_nodes, textual_edges, k, topk_entity):
    topk_graph, topk_desc = [], ""
    for node_id in top_k_indices:
        try:
            subgraph_nodes, subgraph_edge_indices, _, _ = k_hop_subgraph(
                node_idx=node_id, 
                num_hops=k, 
                edge_index=graph.edge_index, 
                relabel_nodes=False
            )

            if graph.edge_attr.dim() == 2:
                if subgraph_edge_indices.size(1) == 0: subgraph_edge_attr = None
                else: 
                    subgraph_edge_index = subgraph_edge_indices.T.tolist()
                    graph_edge_index = graph.edge_index.T.tolist()
                    indices = torch.tensor([graph_edge_index.index(pair) for pair in subgraph_edge_index])
                    subgraph_edge_attr = graph.edge_attr[indices]
            else: 
                subgraph_edge_attr = graph.edge_attr[subgraph_edge_indices]

            # Extract the subgraph
            subgraph = Data(
                x=graph.x[subgraph_nodes],
                edge_index=subgraph_edge_indices,
                edge_attr=None if graph.edge_attr is None else subgraph_edge_attr,
            )
        except:
            subgraph = Data(x=graph.x[node_id].unsqueeze(0), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=None)
            subgraph_nodes = torch.arange(subgraph.num_nodes)

        sim_nodes = torch.nn.CosineSimilarity(dim=-1)(q_emb, subgraph.x)
        if sim_nodes.shape[0] > topk_entity:
            top_k_nodes = torch.topk(sim_nodes, topk_entity, largest=True).indices.tolist()
        else:
            top_k_nodes = torch.arange(sim_nodes.shape[0]).tolist()
        subnodes_desc = textual_nodes.iloc[subgraph_nodes].iloc[top_k_nodes]

        if subgraph.edge_attr is None: 
            topk_desc += subnodes_desc.to_csv(index=False)+ '\n'
            subgraph.edge_index = torch.tensor([], dtype=torch.long).view(2, -1)
        else:
            sim_edges = torch.nn.CosineSimilarity(dim=-1)(q_emb, subgraph.edge_attr)
            if graph.edge_attr.dim() == 2:
                if sim_edges.shape[0] > topk_entity:
                    top_k_edges = torch.topk(sim_edges, topk_entity, largest=True).indices.view(-1).unique().tolist()
                else:
                    top_k_edges = torch.arange(sim_edges.shape[0]).tolist()
            else: 
                if sim_edges.shape[1] > topk_entity:
                    top_k_edges = torch.topk(sim_edges, topk_entity, largest=True).indices.view(-1).unique().tolist()
                else:
                    top_k_edges = torch.arange(sim_edges.shape[1]).tolist()

            # Use hard_promt() if you would like to achieve a lossless graph textual description here.
            subgraph_edge_indices = [edges_mapping.index((src, dst)) for _, (src, dst) in enumerate(subgraph.edge_index.T.numpy()) if (src, dst) in edges_mapping]
            subedges_desc = textual_edges.iloc[subgraph_edge_indices].iloc[top_k_edges]
            topk_desc += subnodes_desc.to_csv(index=False)+ '\n' + subedges_desc.to_csv(index=False, columns=['src', 'edge_attr', 'dst']) + '\n'

            nodes_mapping = {n: i for i, n in enumerate(subgraph_nodes.tolist())}
            new_heads = [nodes_mapping[i] for i in subgraph.edge_index[0].tolist()]
            new_tails = [nodes_mapping[i] for i in subgraph.edge_index[1].tolist()]
            edge_index = torch.LongTensor([new_heads, new_tails])
            subgraph.edge_index = edge_index 

        topk_graph.append(subgraph)

    return merge_graphs(topk_graph, q_emb), topk_desc

def retrive_on_graphs(graph, q_emb, textual_nodes, textual_edges, topk=10, k = 2, topk_entity = 5, augment = "none", sims = None):

    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
        graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
        if sims is not None:
            return graph, desc
        else:
            return torch.zeros(0,  dtype=q_emb.dtype), (graph, desc)

    edges_mapping = [(src, dst) for _, (src, dst) in enumerate(graph.edge_index.T.numpy())]
    if sims is not None: 
        top_k_indices = torch.topk(sims, topk, largest=True).indices.tolist()
        return find_topk_subgraph(graph, q_emb, top_k_indices, edges_mapping, textual_nodes, textual_edges, k, topk_entity)
    
    if augment in {"path", "triplet"}:
        kwargs = {
            "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        path_tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=False, revision=kwargs["revision"])
        path_tokenizer.pad_token_id = 0
        path_tokenizer.padding_side = 'left'
        path_model = AutoModelForCausalLM.from_pretrained(
            llm,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        for _, param in path_model.named_parameters():
                    param.requires_grad = False      
    elif augment == "trunk" or augment == "none": 
        pass
    else:
        raise ValueError("Select 'augment' mode from: 'trunk', 'path', 'triplet' or 'none'.")

    flatten_graphs, subg_embs = [], []
    for node_id in range(graph.num_nodes):
        # Get the subgraph
        try:
            subgraph_nodes, subgraph_edge_indices, _, _ = k_hop_subgraph(
                node_idx=node_id, 
                num_hops=k, 
                edge_index=graph.edge_index, 
                relabel_nodes=False
            )

            if graph.edge_attr.dim() == 2:
                if subgraph_edge_indices.size(1) == 0: subgraph_edge_attr = None
                else: 
                    subgraph_edge_index = subgraph_edge_indices.T.tolist()
                    graph_edge_index = graph.edge_index.T.tolist()
                    sampled_edge_indices = torch.tensor([graph_edge_index.index(pair) for pair in subgraph_edge_index])
                    subgraph_edge_attr = graph.edge_attr[sampled_edge_indices]
            else: 
                subgraph_edge_attr = graph.edge_attr[subgraph_edge_indices]

            # Extract the subgraph
            subgraph = Data(
                x=graph.x[subgraph_nodes],
                edge_index=subgraph_edge_indices,
                edge_attr=None if graph.edge_attr is None else subgraph_edge_attr,
            )
        except:
            subgraph = Data(x = graph.x[node_id].unsqueeze(0), edge_index = torch.tensor([], dtype=torch.long).view(2, -1),  edge_attr = None)

        # Not available for large graph like webqsp
        if augment == "path":
            if graph.edge_index[0] == graph.edge_index[1]:
                flatten_graphs.append(textual_nodes[subgraph_nodes])
                continue
            paths = {}
            G = to_networkx(graph)
            for neighbor in subgraph_nodes.tolist():
                if neighbor != node_id:
                    path = find_path(G, node_id, neighbor)
                    if path is not None and len(path) > k: paths[(node_id, neighbor)] = path
            flatten_graph = get_augmented_path(path_model, path_tokenizer, paths, textual_nodes, textual_edges, edges_mapping)
            flatten_graphs.append(flatten_graph)
        elif augment == "triplet": 
            flatten_graph = get_augmented_triplets(path_model, path_tokenizer, subgraph_edge_indices, textual_nodes, textual_edges, edges_mapping)
            flatten_graphs.append(flatten_graph)
        elif augment == "trunk": 
            flatten_graph = get_trunk_triplets(subgraph_edge_indices, textual_nodes, textual_edges, edges_mapping)
            flatten_graphs.append(flatten_graph)
        elif augment == "none": 
            subg_embs.append(get_triplets(subgraph))
        else:
             raise ValueError("Select 'augment' mode from: 'trunk', 'path', 'triplet' or 'none'.")
    
    if augment != "none":  subg_embs = text2embedding(sentence_model, sentence_tokenizer, device, flatten_graphs)
    else: subg_embs = torch.stack(subg_embs)
    sims = torch.nn.CosineSimilarity(dim=-1)(q_emb, subg_embs)

    if graph.num_nodes > topk:
        top_k_indices = torch.topk(sims, topk, largest=True).indices.tolist()
    else:
        top_k_indices = [node_idx for node_idx in range(graph.num_nodes)]

    return sims, find_topk_subgraph(graph, q_emb, top_k_indices, edges_mapping, textual_nodes, textual_edges, k, topk_entity)
