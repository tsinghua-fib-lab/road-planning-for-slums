import numpy as np
import torch


class NullModel:
    def __init__(self):
        self.training = None
        self.device = None

    def train(self, mode=None):
        pass

    def to(self, device=None):
        pass

    @staticmethod
    def parameters() -> None:
        return None


class RandomPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        edge_mask = x[0][-2]
        action = torch.zeros(1)
        valid_actions = torch.nonzero(edge_mask.flatten()).flatten()

        if len(valid_actions) > 0:
            index = torch.randint(0, len(valid_actions), (1,1))
            action[0] = valid_actions[index]

        return action

# class TravelDistancePolicy(NullModel):
#     def __init__(self):
#     super().__init__()

#     @staticmethod
#     def select_action(x, mean_action=True):

#         return actions
class APolicy(NullModel):
    def __init__(self):
        super().__init__()
    @staticmethod
    def select_action(x, mean_action=True):
        edge_mask = x[0][-2]
        action = torch.zeros(1)
        valid_actions = torch.nonzero(edge_mask.flatten()).flatten()

        if len(valid_actions) > 0:
            index = torch.randint(0, len(valid_actions), (1,1))
            action[0] = valid_actions[index]
        
        return action
    
class RoadCostPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, mean_action=True):
        edge_mask = x[0][-2]
        edge_part = x[0][-4]
        edge_part = edge_part.T
        edge_cost = edge_part[1]
        action = torch.zeros(1)

        edge_cost_mask = edge_cost * edge_mask.bool()
        if torch.sum(edge_cost_mask) > 0:
            index = np.argwhere(edge_cost_mask == min(edge_cost_mask[np.nonzero(edge_cost_mask)])).flatten()
            action[0] = index[0]

        return action

class GAPolicy(NullModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def select_action(x, gene, mean_action=True):

        def compute_edge_features(h_nodes, edge_part_feature, edge_index):
            """
            Gather node embeddings to edges.

            Args:
                h_nodes (torch.Tensor): Node embeddings. Shape: (batch, max_num_nodes, node_dim).
                edge_index (torch.Tensor): Edge indices. Shape: (batch, max_num_edges, 2).
                edge_length (torch.Tensor): Edge length. Shape: (batch, max_num_edges).
                edge_mask (torch.Tensor): Edge mask. Shape: (batch, max_num_edges).
                edge_fc_layer (torch.nn.Module): Edge fc layer.

            Returns:
                h_edges (torch.Tensor): edge embeddings. Shape: (batch, max_num_edges, node_dim).
            """
            h_edges1 = torch.gather(h_nodes, 0, edge_index[:, 0].unsqueeze(-1).expand(-1, h_nodes.size(-1)))
            h_edges2 = torch.gather(h_nodes, 0, edge_index[:, 1].unsqueeze(-1).expand(-1, h_nodes.size(-1)))
            h_edges = (h_edges1 + h_edges2)/2
            h_edges = torch.cat([h_edges, edge_part_feature],dim=-1)
    
            return h_edges

        
        gene = torch.Tensor(gene)
        node_feature, edge_part_feature, edge_index, edge_mask = x[0][1], x[0][2], x[0][3], x[0][4]
        actions = torch.zeros(1)

        edge_feature = compute_edge_features(node_feature, edge_part_feature, edge_index)
        road_logits = torch.sum(edge_feature*torch.unsqueeze(torch.unsqueeze(gene, 0), 0), dim=2)
        road_paddings = torch.ones_like(edge_mask, dtype=edge_feature.dtype)*(-2.**32+1)
        masked_road_logits = torch.where(edge_mask.bool(), road_logits, road_paddings)
        road_dist = torch.distributions.Categorical(logits=masked_road_logits)
        if mean_action:
            road_action = road_dist.probs.argmax().to(edge_feature.dtype)
        else:
            road_action = road_dist.sample().to(edge_feature.dtype)
        actions = road_action

        return actions
