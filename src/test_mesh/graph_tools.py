import numpy as np
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch_geometric.datasets.graph_generator.grid_graph import GridGraph
from torch_geometric.utils import to_networkx
import networkx as nx


def generate_normalized_grid_graph(x_points: int, y_points: int, remove_self_loops: bool, dtype: any):
    assert x_points > 0
    assert y_points > 0

    data = GridGraph(width=x_points, height=y_points, dtype=dtype)()
    data.pos[:, 0] = data.pos[:, 0]/(x_points-1)
    data.pos[:, 1] = data.pos[:, 1]/(y_points-1)
    data.x = data.pos

    if remove_self_loops:
        self_loops_id = torch.where(
            (data.edge_index[0, :]-data.edge_index[1, :]) == 0)[0]

        prev_sli = self_loops_id[0]
        new_edge_index = [data.edge_index[:, :prev_sli]]

        for sli in self_loops_id:
            new_edge_index.append(data.edge_index[:, prev_sli+1:sli])
            prev_sli = sli

        new_edge_index.append(data.edge_index[:, sli+1:])
        new_edge_index = torch.concatenate(new_edge_index, axis=-1)
        data.edge_index = new_edge_index

    return data


def draw_graph(data: Data):
    G = to_networkx(data, to_undirected=False)
    nx.draw(
        G, pos=np.array(data.pos), with_labels=True, node_size=500, font_color='yellow')
    plt.show()
