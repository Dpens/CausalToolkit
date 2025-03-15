import io

import pandas as pd
import numpy as np
import PIL
import torch
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.figure as figure
import matplotlib.cm as cm

from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader as torch_dataloader


def data_process(feature, batch_size):
    feature = torch.FloatTensor(feature)
    data = TensorDataset(feature, feature)
    data_loader = torch_dataloader(data, batch_size=batch_size)

    return data_loader


def get_graph_figure(G1: nx.Graph, G2: nx.Graph, **kwargs) -> figure:
    """Returns a figure of NetworkX graph with intersectioned edges & reversed edges drawn.

    Args:
        G1 (nx.Graph): Graph to draw.
        G2 (nx.Graph): Graph to compare with. (Ground truth)

    Returns:
        figure: Matplotlib figure.Figure with graph drawn.
    """
    options = {
        "prog": "circo",
        "graph": {
            "font_size": 15,
            "node_size": 2000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
            "with_labels": True,
        },
        "intersectioned": {
            "edge_color": "green",
            "width": 8,
            "alpha": 0.3,
        },
        "reversed": {
            "edge_color": "orange",
            "width": 8,
            "alpha": 0.3,
        }
    }
    options.update(kwargs)

    ax = plt.subplot()

    if G2 is not None:
        I = nx.intersection(G1, G2)
        R = nx.intersection(G1, nx.reverse(G2))

        pos = nx.nx_agraph.graphviz_layout(G2, prog=options.get("prog"))

        nx.draw_networkx_edges(I, pos, ax=ax, **options.get("intersectioned"))
        nx.draw_networkx_edges(R, pos, ax=ax, **options.get("reversed"))
    else:
        pos = nx.nx_agraph.graphviz_layout(G1, prog=options.get("prog"))

    nx.draw(G1, pos, ax=ax, **options.get("graph"))

    fig = ax.get_figure()
    plt.close()

    return fig


def get_adj_figure(adj_A, **kwargs) -> figure:
    """Returns a figure of adjacency matrix.

    Args:
        adj_A (pd.DataFrame or np.ndarray): Adjacency matrix to plot.

    Returns:
        figure: Plot of adjacency matrix.
    """
    options = {
        "vmin": -1,
        "vmax": 1,
        "cmap": cm.RdBu_r,
    }
    options.update(kwargs)

    ax = plt.subplot()

    plt.imshow(adj_A, **options)
    plt.colorbar()

    fig = ax.get_figure()
    plt.close()

    return fig


def fig2img(fig: figure) -> PIL.Image:
    """Converts matplotlib figure to PIL image.

    Args:
        fig (figure): Matplotlib figure.Figure.

    Returns:
        PIL.Image: PIL image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img



def get_plot_imgs(graph, G: nx.Graph, ground_truth_G: nx.Graph):
    graph_fig = get_graph_figure(G, ground_truth_G)
    graph_img = fig2img(graph_fig)

    adj_fig = get_adj_figure(graph)
    adj_img = fig2img(adj_fig)

    return graph_img, adj_img