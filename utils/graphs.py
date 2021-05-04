import torch
import numpy as np
import vigra
import math
import nifty
import elf
import nifty.graph.agglo as nagglo
from elf.segmentation.watershed import watershed, apply_size_filter
import matplotlib.pyplot as plt


def collate_edges(edges):
    "batches a list of graphs defined by edge arrays"
    n_offs = [0]
    e_offs = [0]
    for i in range(len(edges)):
        n_offs.append(n_offs[-1] + edges[i].max() + 1)
        e_offs.append(e_offs[-1] + edges[i].shape[1])
        edges[i] += n_offs[-2]

    return torch.cat(edges, 1), (n_offs, e_offs)


def separate_nodes(nodes, n_offs):
    r_nodes = []
    for i in range(len(n_offs) - 1):
        r_nodes.append(nodes[n_offs[i]: n_offs[i+1]] - n_offs[i])
    return r_nodes


def separate_edges(edges, e_offs, n_offs):
    r_edges = []
    for i in range(len(e_offs) - 1):
        r_edges.append(edges[e_offs[:, i]: e_offs[:, i+1]] - n_offs[i])
    return r_edges


def get_edge_indices(edges, edge_list):
    """
    :param edges: double nested list of edges in a graph shape(n, 2). must be sorted like (smaller, larger) and must not contain equal values
    :param edge_list: list of edges. Each edge must be present in edges. Can contain multiple entries of one edge
    :return: the indices for edges that each entry in edge list correpsonds to
    """
    indices = []
    for sg in edge_list:
        indices.append(torch.nonzero(((edges[0] == sg[0].unsqueeze(-1)) & (edges[1] == sg[1].unsqueeze(-1))), as_tuple=True)[1])
        assert indices[-1].shape[0] == sg.shape[1], "edges must be sorted and unique"
    return indices

def squeeze_repr(nodes, edges, seg):
    """
    This functions renames the nodes to [0,..,len(nodes)-1] in a superpixel rag consisting of nodes edges and a segmentation
    :param nodes: pt tensor
    :param edges: pt tensor
    :param seg: pt tensor
    :return: none
    """

    _nodes = torch.arange(0, len(nodes), device=nodes.device)
    indices = torch.where(edges.unsqueeze(0) == nodes.unsqueeze(-1).unsqueeze(-1))
    edges[indices[1], indices[2]] = _nodes[indices[0]]
    indices = torch.where(seg.unsqueeze(0) == nodes.unsqueeze(-1).unsqueeze(-1))
    seg[indices[1], indices[2]] = _nodes[indices[0]].float().type(seg.dtype)



def get_angles_smass_in_rag(edges, segmentation):
    """
        CAUTION: this is specificly targeted for the leptin dataset
    """
    sigm_flatness = torch.numel(segmentation) / 500
    sigm_shift = 0.8
    c_norm = math.sqrt(segmentation.shape[-1]**2 + segmentation.shape[-2]**2) / 8
    nodes = torch.unique(segmentation)
    random_rotation = (torch.rand((1,)).item() * 2) - 1

    meshgrid = torch.stack(torch.meshgrid(torch.arange(segmentation.shape[0], device=segmentation.device),
                                          torch.arange(segmentation.shape[1], device=segmentation.device)))
    one_hot = segmentation[None] == nodes[:, None, None]
    sup_sizes = one_hot.flatten(1).sum(-1)
    cart_cms = (one_hot[None] * meshgrid[:, None]).flatten(2).sum(2) / sup_sizes[None]

    a = (cart_cms[0].float() - 390)  # This is not the image center because the Leptin data is not exactly concentric to it
    b = (cart_cms[1].float() - 340)
    c = torch.sqrt(a**2 + b**2)
    normed_c = torch.sigmoid(c / c_norm - 1) * 2 - 1
    ang = torch.atan(a/(b + 1e-10)).abs()
    ang[(b < 0) & (a >= 0)] = np.pi - ang[(b < 0) & (a >= 0)]
    ang[(b < 0) & (a < 0)] = np.pi + ang[(b < 0) & (a < 0)]
    ang[(b >= 0) & (a < 0)] = 2 * np.pi - ang[(b >= 0) & (a < 0)]
    ang /= np.pi
    ang -= 1
    # ang += random_rotation
    # ang[a > 1] = (ang[a > 1] - 1) - 1
    # ang[ang < -1] = 1 + (ang[ang < -1] + 1)
    cms = torch.stack([ang, normed_c, c], 1)

    vec = cart_cms[:, edges[0]] - cart_cms[:, edges[1]]
    angles = torch.atan(vec[0] / (vec[1] + np.finfo(float).eps))
    angles = 2 * angles / np.pi
    sup_sizes = torch.sigmoid(sup_sizes / sigm_flatness - sigm_shift) * 2 - 1
    return angles, torch.cat([sup_sizes[:, None], cms[:, :-1]], 1), cms[:, -1]


def get_joint_sg_logprobs_edges(logprobs, scale, obs, sg_ind, sz):
    return logprobs[obs.subgraph_indices[sg_ind].view(-1, sz)].sum(-1).sum(-1), \
           (1 / 2 * (1 + (2 * np.pi * scale[obs.subgraph_indices[sg_ind].view(-1, sz)] ** 2).log())).sum(-1).sum(-1)

def get_joint_sg_logprobs_nodes(logprobs, scale, obs, sg_ind, sz):
    sgs = obs.subgraphs[sg_ind].view(2, -1, sz).permute(1, 2, 0)
    joint_logprobs = torch.zeros(sgs.shape[0], device=logprobs.device)
    sg_entropy = torch.zeros(sgs.shape[0], device=logprobs.device)
    for i, sg in enumerate(sgs):
        un = torch.unique(sg)
        joint_logprobs[i] = logprobs[un].sum()
        sg_entropy[i] = (1 / 2 * (
                    1 + (2 * np.pi * scale[un] ** 2).log())).sum()
    return joint_logprobs, sg_entropy

def run_watershed(hmap_, min_size=None, nhood=4):
    hmap = hmap_.astype(np.float32)
    compute_maxima = vigra.analysis.localMaxima if hmap.ndim == 2 else vigra.analysis.localMaxima3D
    seeds = compute_maxima(hmap.astype(np.float32), marker=np.nan, allowAtBorder=True, allowPlateaus=True, neighborhood=nhood)
    seeds = vigra.analysis.labelMultiArrayWithBackground(np.isnan(seeds).view('uint8'))

    ws, _ = watershed(hmap, seeds)
    if min_size is not None:
        ws, _ = apply_size_filter(ws, hmap, min_size)
    return ws

def get_soln_graph_clustering(sp_seg, edge_ids, node_features, n_max_object):
    labels = []
    node_labels = []
    cluster_policy = nagglo.nodeAndEdgeWeightedClusterPolicy
    for i, sp_seg in enumerate(sp_seg):
        single_node_features = node_features.detach().cpu().numpy()
        rag = nifty.graph.undirectedGraph(single_node_features.shape[0])
        rag.insertEdges((edge_ids).detach().cpu().numpy())

        edge_weights = np.ones(rag.numberOfEdges, dtype=np.int)
        edge_sizes = np.ones(rag.numberOfEdges, dtype=np.int)
        node_sizes = np.ones(rag.numberOfNodes, dtype=np.int)

        policy = cluster_policy(
            graph=rag,
            edgeIndicators=edge_weights,
            edgeSizes=edge_sizes,
            nodeFeatures=single_node_features,
            nodeSizes=node_sizes,
            numberOfNodesStop=n_max_object,
            beta=1,
            sizeRegularizer=0
        )

        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()

        node_labels.append(clustering.result())
        rag = elf.segmentation.features.compute_rag(np.expand_dims(sp_seg.cpu(), axis=0))
        labels.append(elf.segmentation.features.project_node_labels_to_pixels(rag, node_labels[-1]).squeeze())
    return torch.from_numpy(np.stack(labels).astype(np.float)).to(node_features.device), \
           torch.from_numpy(np.concatenate(node_labels).astype(np.float)).to(node_features.device)


if __name__ == "__main__":
    # edges = np.array([[1, 3], [2, 4], [1, 2], [2, 3], [3, 5], [3, 6], [1, 5], [2, 8], [4, 8], [4, 9], [5, 9], [8, 9]])
    # edge_list = edges[np.random.choice(np.arange(edges.shape[0]), size=(20 * 10))]
    # edge_indices = get_edge_indices(torch.from_numpy(edges).transpose(0, 1), torch.from_numpy(edge_list).transpose(0, 1))

    sp = torch.zeros((100, 100), dtype=torch.float)
    sp[:50, :50] = 0.0
    sp[50:, :50] = 1.0
    sp[:50, 50:] = 2.0
    sp[50:, 50:] = 3.0
    sp[40:60, 40:60] = 4.0
    edges = torch.tensor([[0, 4], [1, 4], [2, 4], [3, 4]], dtype=torch.long).T


