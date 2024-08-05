import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from collections import deque,  defaultdict

def reorder(level_number):
    new_level_number = {}
    indents = {}
    major_counter = 1  # Start the major section from 1
    sub_counters = defaultdict(lambda: 1)  # Sub-counters for each major section

    for node, number in sorted(level_number.items(), key=lambda x: x[1]):  # Sort by existing numbering
        number = ".".join(number.split('.')[1:]) if node != 0 else number  # Remove first part for non-root nodes
        parts = number.split('.')
        depth = len(parts) - 1  # Depth is the number of dots

        if node == 0:
            # Root node handling
            new_number = str(major_counter)
            major_counter += 1
        else:
            parent_number = '.'.join(parts[:-1])
            if parts[-1] == '1' and parent_number:  # Reset the sub-counter when a new subtree starts
                sub_counters[parent_number] = 1
            if parent_number:
                new_number = f"{parent_number}.{sub_counters[parent_number]}"
            else:
                new_number = str(sub_counters[parent_number])  # No dot for first level numbers
            sub_counters[parent_number] += 1
        
        indents[node] = '  ' * depth  # Adding space based on depth
        new_level_number[node] = f"{new_number}"

    return new_level_number, indents

def hard_prompt(graph, nodes, edges, node_id, k=1):
    try:
        subgraph_nodes, subgraph_edge_indices, _, _ = k_hop_subgraph(
            node_idx=node_id, 
            num_hops=k, 
            edge_index=graph.edge_index, 
            relabel_nodes=True
        )
    except:
        raise ValueError("This node is isolated!")
    
    node_mapping = {old: new for new, old in enumerate(subgraph_nodes.numpy())}
    sampled_nodes = nodes.loc[subgraph_nodes.numpy()].copy()
    sampled_nodes['node_id'] = sampled_nodes['node_id'].map(node_mapping)
    sampled_nodes.reset_index(drop=True, inplace=True)

    edge_mapping = [(src, dst) for src, dst in graph.edge_index.t().numpy()]
    mapped_edges = [(node_mapping[src], node_mapping[dst]) if src in node_mapping and dst in node_mapping else None for src, dst in edge_mapping]
    valid_edges = [idx for idx, edge in enumerate(mapped_edges) if edge is not None]

    sampled_edges = edges.iloc[valid_edges].copy()
    sampled_edges[['src', 'dst']] = [mapped_edges[idx] for idx in valid_edges]

    ### Perform BFS on this subgraph to find which edges in the tree:
    visited = [False] * subgraph_nodes.shape[0]
    queue = deque([(node_id, "1")])  # Correct initialization of the queue with a tuple
    level = {node_id: 0}
    level_number = {node_id: "1"}  # To store the unique level identifier
    is_leaf = {node_id: True}  # Initially assume all nodes are leaves, then update
    parent_dict = {node_id: None}  # Root node has no parent

    visited[node_id] = True
    edges_in_tree = []
    all_edges = set()

    while queue:
        node, num = queue.popleft()  # Correctly unpack tuple as initialized
        current_level = level[node]
        if current_level >= k:  # Limit the BFS to depth k
            continue

        # Check neighbors following the direction of the edge
        edges = subgraph_edge_indices.t()
        neighbors = edges[edges[:, 0] == node][:, 1].tolist()  # Convert to list for iteration
        child_count = defaultdict(int)
        has_child = False  # To check if the current node has any child

        for neighbor in neighbors:
            all_edges.add((node, neighbor))  # Keep directed edges as (node, neighbor)
            if not visited[neighbor]:
                visited[neighbor] = True
                edges_in_tree.append((node, neighbor))
                has_child = True  # The current node has at least one child
                
                child_count[node] += 1  # Increment child count for the current node
                new_number = f"{num}.{child_count[node]}"
                level[neighbor] = current_level + 1
                level_number[neighbor] = new_number
                is_leaf[neighbor] = True  # Initialize new nodes as leaves
                queue.append((neighbor, new_number))  # Correctly append tuple to queue
                parent_dict[neighbor] = node  # Record the parent of each node

        if has_child:
            is_leaf[node] = False     
    
    order, indents = reorder(level_number)
    hidden_edges = list(all_edges.difference(set(edges_in_tree)))
    hidden_indices = [[h_e[0] for h_e in hidden_edges], [h_e[1] for h_e in hidden_edges]]

    ### Write the edges not in the BFS tree as the text edge and update the nodes description of the tree.
    edge_index = torch.tensor(list(zip(*edges_in_tree)), dtype=torch.long)
    tree_graph = Data(
        edge_index=edge_index,
        text_nodes=sampled_nodes,
        text_edges=sampled_edges[sampled_edges.apply(lambda x: (x['src'], x['dst']) in edges_in_tree, axis=1)]
    )

    # sampled_nodes['node_attr'] = sampled_nodes['node_attr'].apply(lambda x: f"({x})")
    # sampled_edges['edge_attr'] = sampled_edges['edge_attr'].apply(lambda x: f"[{x}]")

    node_descriptions = {}
    edge_attr_map = {(row['src'], row['dst']): row['edge_attr'] for _, row in sampled_edges.iterrows()}
    for node in order.keys():
        if not is_leaf[node]: 
            if node == node_id:
                description = (f"[ROOT] {sampled_nodes.loc[node, 'node_attr']}.\n"
                               f"{sampled_nodes.loc[node, 'node_attr']} is connected to: ")
            else:
                description = (f"{indents[node]}{order[node]} {sampled_nodes.loc[node, 'node_attr']} via {edge_attr_map.get((parent_dict[node], node))}.\n"
                                f"{indents[node]}{sampled_nodes.loc[node, 'node_attr']} is connected to: ")
        else:
            description = f"{indents[node]}{order[node]} {sampled_nodes.loc[node, 'node_attr']} via {edge_attr_map.get((parent_dict[node], node))}."
        if node in hidden_indices[0]:
            if is_leaf[node]: 
                description += f"\n{indents[node]}{sampled_nodes.loc[node, 'node_attr']} is connected to: "
            pairs = [hidden_edges[i] for i, head in enumerate(hidden_indices[0]) if head == node]
            # Preliminary test showed that only index of node will not work in zero-shot senario.
            # sampled_hidden = [f"{order[he[1]]} via {edge_attr_map.get(he)};" for he in pairs]
            sampled_hidden = [f"{sampled_nodes.loc[he[1], 'node_attr']} via {edge_attr_map.get(he)};" for he in pairs]
            description += " ".join(sampled_hidden)
        if not is_leaf[node] and node != node_id: 
            description += f"\n{sampled_nodes.loc[node, 'node_attr']} is connected to:"
        node_descriptions[node] = description
    flatten_graph = "\n".join(node_descriptions.values())

    return flatten_graph, tree_graph

def edges_prompt(graph, nodes, edges, node_id, k=1):
    _, tree_graph = hard_prompt(graph, nodes, edges, node_id, k)
    
    edge_index = tree_graph.edge_index.t().numpy()
    text_nodes = tree_graph.text_nodes
    text_edges = tree_graph.text_edges

    textual_edges = []
    for src, dst in edge_index:
        src_attr = text_nodes.loc[src, 'node_attr']
        dst_attr = text_nodes.loc[dst, 'node_attr']
        edge_attr = text_edges.loc[(text_edges['src'] == src) & (text_edges['dst'] == dst), 'edge_attr'].values[0]
        textual_edges.append(f"{src_attr}, [{edge_attr}], {dst_attr}")

    return "\n".join(textual_edges)
