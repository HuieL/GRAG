import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv, GATConv

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, 2 * hidden_dim))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(2 * hidden_dim, 2 * hidden_dim))
        # Output a scalar
        self.layers.append(nn.Linear(2 * hidden_dim, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x) 
        return x

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mlp_layers = 3, num_heads=4, operator = "euclidean"):
        super(GAT, self).__init__()
        self.operator = operator
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

        # Define two MLPs, one for node features and one for edge attributes
        self.mlp_x = MLP(in_channels, hidden_channels, mlp_layers)
        self.mlp_edge_attr = MLP(in_channels, hidden_channels, mlp_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp_x.apply(self.reset_weights)
        self.mlp_edge_attr.apply(self.reset_weights)

    def reset_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def euclidean_distance(self, x, y):
        return torch.sqrt((x - y) ** 2)

    def manhattan_distance(self, x, y):
        return torch.abs(x - y)

    def element_distance(self, x, y):
        return x - y

    def forward(self, x, edge_index, question_node, edge_attr, question_edge):
        # Compute distance between x and question and process through MLP
        # x_dist = torch.cat([x, question.repeat(x.size(0), 1)], dim=1)
        if self.operator == "euclidean":
            x_dist = self.euclidean_distance(x, question_node)
        elif self.operator == "manhattan":
            x_dist = self.manhattan_distance(x, question_node)
        elif self.operator == "element":
            x_dist = self.element_distance(x, question_node)
        else:
            raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
        x_weights = self.mlp_x(x_dist).sigmoid()  # Using sigmoid to scale weights between 0 and 1
        x *= x_weights

        if not torch.any(edge_attr):
            pass
        else:
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.view(-1, edge_attr.size(2))  # Flatten if directed graph

            # edge_dist = torch.cat([edge_attr, question.repeat(edge_attr.size(0), 1)], dim=1)
            if self.operator == "euclidean":
                edge_dist = self.euclidean_distance(edge_attr, question_edge)
            elif self.operator == "manhattan":
                edge_dist = self.manhattan_distance(edge_attr, question_edge)
            elif self.operator == "element":
                edge_dist = self.element_distance(edge_attr, question_edge)
            else:
                raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
            edge_weights = self.mlp_edge_attr(edge_dist).sigmoid()
            edge_weights = edge_weights.view(*edge_attr.shape[:-1], 1)  # Reshape weights to match edge_attr
            edge_attr *= edge_weights

            if edge_attr.dim() > 2:
                edge_attr = edge_attr.view(2, -1, edge_attr.size(-1))  # Reshape back for directed graph

        if not torch.any(edge_attr):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index=edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index=edge_index)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr

class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mlp_layers = 3, num_heads=-1, operator = "euclidean"):
        super(GraphTransformer, self).__init__()
        self.operator = operator
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads,dropout=dropout,))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, dropout=dropout,))
        self.dropout = dropout

        # Define two MLPs, one for node features and one for edge attributes
        self.mlp_x = MLP(input_dim=in_channels * 2, hidden_dim=1024, num_layers=mlp_layers)
        self.mlp_edge_attr = MLP(input_dim=in_channels * 2, hidden_dim=1024, num_layers=mlp_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp_x.apply(self.reset_weights)
        self.mlp_edge_attr.apply(self.reset_weights)

    def reset_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def euclidean_distance(self, x, y):
        return torch.sqrt((x - y) ** 2)

    def manhattan_distance(self, x, y):
        return torch.abs(x - y)

    def element_distance(self, x, y):
        return x - y

    def forward(self, x, edge_index, question_node, edge_attr, question_edge):
        # Compute distance between x and question and process through MLP
        if self.operator == "euclidean":
            x_dist = self.euclidean_distance(x, question_node)
        elif self.operator == "manhattan":
            x_dist = self.manhattan_distance(x, question_node)
        elif self.operator == "element":
            x_dist = self.element_distance(x, question_node)
        else:
            raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
        x_weights = self.mlp_x(x_dist).sigmoid()  # Using sigmoid to scale weights between 0 and 1
        x *= x_weights

        if not torch.any(edge_attr):
            pass
        else:
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.view(-1, edge_attr.size(2))  # Flatten if directed graph

            # edge_dist = torch.cat([edge_attr, question.repeat(edge_attr.size(0), 1)], dim=1)
            if self.operator == "euclidean":
                edge_dist = self.euclidean_distance(edge_attr, question_edge)
            elif self.operator == "manhattan":
                edge_dist = self.manhattan_distance(edge_attr, question_edge)
            elif self.operator == "element":
                edge_dist = self.element_distance(edge_attr, question_edge)
            else:
                raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
            edge_weights = self.mlp_edge_attr(edge_dist).sigmoid()
            edge_weights = edge_weights.view(*edge_attr.shape[:-1], 1)  # Reshape weights to match edge_attr
            edge_attr *= edge_weights

            if edge_attr.dim() > 2:
                edge_attr = edge_attr.view(2, -1, edge_attr.size(-1))  # Reshape back for directed graph

        if not torch.any(edge_attr):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index=edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index=edge_index)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mlp_layers = 3, num_heads=-1, operator = "euclidean"):
        super(GCN, self).__init__()
        self.operator = operator
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        # Define two MLPs, one for node features and one for edge attributes
        self.mlp_x = MLP(input_dim=in_channels * 2, hidden_dim=1024, num_layers=mlp_layers)
        self.mlp_edge_attr = MLP(input_dim=in_channels * 2, hidden_dim=1024, num_layers=mlp_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.mlp_x.apply(self.reset_weights)
        self.mlp_edge_attr.apply(self.reset_weights)

    def euclidean_distance(self, x, y):
        return torch.sqrt((x - y) ** 2)

    def manhattan_distance(self, x, y):
        return torch.abs(x - y)

    def element_distance(self, x, y):
        return x - y
    
    def forward(self, x, adj_t, question_node, edge_attr, question_edge):
        # Compute distance between x and question and process through MLP
        if self.operator == "euclidean":
            x_dist = self.euclidean_distance(x, question_node)
        elif self.operator == "manhattan":
            x_dist = self.manhattan_distance(x, question_node)
        elif self.operator == "element":
            x_dist = self.element_distance(x, question_node)
        else:
            raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
        x_weights = self.mlp_x(x_dist).sigmoid()  # Using sigmoid to scale weights between 0 and 1
        x *= x_weights

        if not torch.any(edge_attr):
            pass
        else:
            if edge_attr.dim() == 3:
                edge_attr = edge_attr.view(-1, edge_attr.size(2))  # Flatten if directed graph

            # edge_dist = torch.cat([edge_attr, question.repeat(edge_attr.size(0), 1)], dim=1)
            if self.operator == "euclidean":
                edge_dist = self.euclidean_distance(edge_attr, question_edge)
            elif self.operator == "manhattan":
                edge_dist = self.manhattan_distance(edge_attr, question_edge)
            elif self.operator == "element":
                edge_dist = self.element_distance(edge_attr, question_edge)
            else:
                raise ValueError("Choose from 'euclidean', 'manhattan' and 'element'.")
            edge_weights = self.mlp_edge_attr(edge_dist).sigmoid()
            edge_weights = edge_weights.view(*edge_attr.shape[:-1], 1)  # Reshape weights to match edge_attr
            edge_attr *= edge_weights

            if edge_attr.dim() > 2:
                edge_attr = edge_attr.view(2, -1, edge_attr.size(-1))  # Reshape back for directed graph

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr

load_gnn_model = {
    'gat': GAT,
    'trans': GraphTransformer,
    'gcn': GCN,
}
