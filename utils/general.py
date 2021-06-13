import numpy as np
import torch
import math
import random
import matplotlib.pyplot as plt
from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
from elf.segmentation.multicut import transform_probabilities_to_costs, multicut_kernighan_lin
from torch import multiprocessing as mp
from sklearn.decomposition import PCA
from scipy.cluster.vq import kmeans2, whiten
import cv2
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian

# Global counter
# from traitlets.tests.test_traitlets import test_subclass_override_not_registered


class Counter():
  def __init__(self):
    self.val = mp.Value('i', 0)
    self.lock = mp.Lock()

  def increment(self):
    with self.lock:
      self.val.value += 1

  def set(self, val):
    with self.lock:
      self.val.value = val

  def reset(self):
    with self.lock:
      self.val.value = 0

  def value(self):
    with self.lock:
      return self.val.value


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def get_all_arg_combos(grid, paths):
    key = random.choice(list(grid))
    new_paths = []
    new_grid = grid.copy()
    del new_grid[key]
    for val in grid[key]:
        if paths:
            for path in paths:
                path[key] = val
                new_paths.append(path.copy())
        else:
            new_paths.append({key: val})
    if new_grid:
        return get_all_arg_combos(new_grid, new_paths)
    return new_paths

def get_angles(x):
    """
        for a set of vectors this returns the angle [-pi, pi]
        of the vector with each vector in the unit othonormal basis.
        x should be a set of normalized vectors (NCHW)
    """
    ob = torch.eye(x.shape[1], device=x.device)
    return torch.acos(torch.matmul(ob[None, None, None], x.permute(0, 2, 3, 1)[..., None])).squeeze(-1).permute(0, 3, 1, 2)

def calculate_naive_gt_edge_costs(edges, sp_gt):
    return (sp_gt.squeeze()[edges.astype(np.int)][:, 0] != sp_gt.squeeze()[edges.astype(np.int)][:, 1]).float()


# Knuth's algorithm for generating Poisson samples
def poisson(self, lmbd):
    L, k, p = math.exp(-lmbd), 0, 1
    while p > L:
        k += 1
        p *= random.uniform(0, 1)
    return max(k - 1, 0)

# Adjusts learning rate
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calculate_gt_edge_costs(neighbors, new_seg, gt_seg, thresh):
    dev = gt_seg.device
    rewards = torch.zeros(len(neighbors), device=dev)
    new_seg += 1
    neighbors += 1
    gt_seg += 1

    for idx, neighbor in enumerate(neighbors):
        mask_n1, mask_n2 = new_seg == neighbor[0], new_seg == neighbor[1]
        mask = mask_n1 + mask_n2
        mskd_gt_seg = mask * gt_seg
        mskd_new_seg = mask * new_seg
        n_obj_gt = torch.unique(mskd_gt_seg)
        n_obj_new = torch.unique(mskd_new_seg)
        n_obj_gt = n_obj_gt[1:] if n_obj_gt[0] == 0 else n_obj_gt
        if len(n_obj_gt) == 1:
            rewards[idx] = 0
        else:
            n_obj_new = n_obj_new[1:] if n_obj_new[0] == 0 else n_obj_new
            assert len(n_obj_new) == 2
            overlaps = torch.zeros((len(n_obj_gt), 2), device=dev)
            for j, obj in enumerate(n_obj_gt):
                mask_gt = mskd_gt_seg == obj
                overlaps[j, 0] = torch.sum(mask_gt * mask_n1) / torch.sum(mask_n1)
                overlaps[j, 1] = torch.sum(mask_gt * mask_n2) / torch.sum(mask_n2)
            if torch.sum(overlaps.max(dim=1).values > thresh) >= 2:
                rewards[idx] = 1
            else:
                rewards[idx] = 0
    new_seg -= 1
    neighbors -= 1
    gt_seg -= 1
    return rewards


def project_overseg_to_seg(over_seg, seg):
    _over_seg = over_seg + 1
    _seg = seg + 1

    onehot = _over_seg[None] == torch.unique(_over_seg)[:, None, None]
    overlaps = onehot * _seg[None]
    labels = torch.tensor([torch.bincount(ol)[1:].argmax().item() for ol in overlaps.flatten(1)], device=seg.device)
    projection = (onehot * labels[:, None, None]).sum(0)

    # make label ids consecutive
    uni = torch.unique(projection)
    onehot = projection[None] == uni[:, None, None]
    projection = (onehot * torch.arange(0, len(uni), device=seg.device)[:, None, None]).sum(0)

    return projection


def bbox(array2d_c):
    assert len(array2d_c.shape) == 3
    y_vals = []
    x_vals = []
    for array2d in array2d_c:
        y = np.where(np.any(array2d, axis=1))
        x = np.where(np.any(array2d, axis=0))
        ymin, ymax = y[0][[0, -1]] if len(y[0]) != 0 else (0, 0)
        xmin, xmax = x[0][[0, -1]] if len(x[0]) != 0 else (0, 0)
        y_vals.append([ymin, ymax+1])
        x_vals.append([xmin, xmax+1])
    return y_vals, x_vals


def ind_flat_2_spat(flat_indices, shape):
    spat_indices = np.zeros([len(flat_indices)] + [len(shape)], dtype=np.integer)
    for flat_ind, spat_ind in zip(flat_indices, spat_indices):
        rm = flat_ind
        for dim in range(1, len(shape)):
            sz = np.prod(shape[dim:])
            spat_ind[dim - 1] = rm // sz
            rm -= spat_ind[dim - 1] * sz
        spat_ind[-1] = rm
    return spat_indices


def ind_spat_2_flat(spat_indices, shape):
    flat_indices = np.zeros(len(spat_indices), dtype=np.integer)
    for i, spat_ind in enumerate(spat_indices):
        for dim in range(len(shape)):
            flat_indices[i] += max(1, np.prod(shape[dim + 1:])) * spat_ind[dim]
    return flat_indices


def add_rndness_in_dis(dis, factor):
    assert isinstance(dis, np.ndarray)
    assert len(dis.shape) == 2
    ret_dis = dis - ((dis - np.transpose([np.mean(dis, axis=-1)])) * factor)
    return dis


def pca_svd(X, k, center=True):
    # code from https://gist.github.com/project-delphi/e1112dbc0940d729a90f59846d25342b
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n, n])
    H = torch.eye(n) - h
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    explained_variance = torch.mul(s[:k], s[:k])/(n-1)  # remove normalization?
    return components, explained_variance


# def get_contour_from_2d_binary(mask: torch.Tensor):
#     """
#     :param mask: n_dim should be three (N|H|W). can be bool or long but should be binary if long.
#     :return: tensor of the same shape and type bool containing all inner contours of objects in mask
#     """
#     max_p = torch.nn.MaxPool2d(3, stride=1, padding=1)
#     return ((max_p(mask) != mask) | (-max_p(-mask) != mask)).long()

def get_colored_edges_in_sseg(sseg: torch.Tensor, edges: torch.Tensor, scores: torch.Tensor):
    sseg = sseg + 1
    edges = edges + 1
    max_p = torch.nn.MaxPool2d(3, stride=1, padding=1)
    maxp_seg = max_p(sseg)
    minp_seg = -max_p(-sseg)
    bnds = ((maxp_seg != sseg) * maxp_seg + (minp_seg != sseg) * minp_seg).long()

    #chunk the whole op since gpu too small
    scored_bnds = torch.zeros_like(sseg.squeeze())
    chunks = 500
    slc_sz = edges.shape[-1] // chunks
    slices = [slice(slc_sz * step, slc_sz * (step + 1), 1) for step in range(chunks)]
    if edges.shape[-1] != chunks * slc_sz:
        slices.append(slice(slc_sz * chunks, edges.shape[-1], 1))

    for slc in slices:
        scattered_pairs = (sseg[None] == edges[:, slc, None, None]).sum(0)
        bnd_pairs = (bnds[None] == edges[:, slc, None, None]).sum(0)
        scored_bnds += (scattered_pairs * bnd_pairs * scores[slc, None, None]).sum(0)

    sseg = sseg - 1
    edges = edges - 1
    bnd_mask = bnds[0] != 0
    return torch.stack([(0.5 + scored_bnds) * (bnd_mask & (scored_bnds < 0.5)), scored_bnds * (scored_bnds > 0.5), torch.zeros_like(scored_bnds)], -1), scored_bnds, bnd_mask

def sync_segmentations(seg_base, seg_var, sync_bg_as_id0=False):
    ids = torch.unique(seg_base)
    max_id = ids.max()
    seg_var = seg_var.clone() + max_id
    all_var_bins = torch.bincount(seg_var.ravel())

    for id in ids:
        mask = (seg_base == id).long()
        seg_var_mask = seg_var * mask
        seg_var_ids = torch.unique(seg_var_mask)[1:]
        if id == 0:
            for var_id in seg_var_ids:
                overlap = (var_id == seg_var_mask).sum()
                var_mask = (var_id == seg_var)
                if sync_bg_as_id0 and var_id == max_id:
                    seg_var[var_mask] = id
                elif overlap > var_mask.sum() / 2:
                    seg_var[var_mask] = id
                all_var_bins = torch.bincount(seg_var.ravel())
        else:
            bins = torch.bincount(seg_var_mask.ravel())
            sorted_indices = torch.argsort(bins[seg_var_ids])
            for id_var in reversed(seg_var_ids[sorted_indices]):
                if bins[id_var] > all_var_bins[id_var] / 2:
                    seg_var[seg_var == id_var] = id
                    all_var_bins = torch.bincount(seg_var.ravel())
                    break
    return seg_var

def maskout(img):
    img = (img * 255).astype(np.uint8)
    img = img/img.max()
    size = (5, 5)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    for _ in range(5):
        img = cv2.dilate(img, kernel)
    for _ in range(5):
        img = cv2.erode(img, kernel)
    img = (img - img.mean()).clip(0,1)
    img = img / img.max()
    img = (img > 0.1).astype(np.float32)
    for _ in range(10):
        img = cv2.dilate(img, kernel)
    for _ in range(10):
        img = cv2.erode(img, kernel)
    for _ in range(3):
        img = cv2.dilate(img, kernel)
    img = (img * 255).astype(np.uint8)
    img = cv2.medianBlur(img, 11)
    img = img.astype(np.float32)/255
    return img

def multicut_from_probas(segmentation, edges, edge_weights):
    rag = compute_rag(segmentation)
    edge_dict = dict(zip(list(map(tuple, edges)), edge_weights))
    costs = np.empty(len(edge_weights))
    for i, neighbors in enumerate(rag.uvIds()):
        if tuple(neighbors) in edge_dict:
            costs[i] = edge_dict[tuple(neighbors)]
        else:
            costs[i] = edge_dict[(neighbors[1], neighbors[0])]
    costs = transform_probabilities_to_costs(costs)
    node_labels = multicut_kernighan_lin(rag, costs)

    return project_node_labels_to_pixels(rag, node_labels).squeeze()


def check_no_singles(edges, num_nodes):
    return all(np.unique(edges.ravel()) == np.array(range(num_nodes)))


def collate_graphs(node_features, edge_features, edges, shuffle=False):
    for i in len(node_features):
        edges[i] += i
    return torch.stack(node_features), torch.stack(edges), torch.stack(edge_features)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def pca_project(embeddings, n_comps=3):
    assert embeddings.ndim == 3
    # reshape (C, H, W) -> (C, H * W) and transpose
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1).T
    # init PCA with 3 principal components: one for each RGB channel
    pca = PCA(n_components=n_comps)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(flattened_embeddings)
    # reshape back to original
    shape = list(embeddings.shape)
    shape[0] = n_comps
    img = flattened_embeddings.T.reshape(shape)
    # normalize to [0, 255]
    img = 255 * (img - np.min(img)) / np.ptp(img)
    return np.moveaxis(img.astype('uint8'), 0, -1)

def cluster_embeddings(embeddings, n_clusters):
    shape = embeddings.shape
    centroid, label = kmeans2(whiten(embeddings.reshape(-1, shape[-1])), n_clusters, minit='points', iter=20)
    pred_label = label.reshape(shape[:-1])
    return pred_label


def get_scores(prediction, target, tau):
    pass


def pca_project_1d(embeddings, n_comps=3):
    assert embeddings.ndim == 2
    # reshape (C, H, W) -> (C, H * W) and transpose
    pca = PCA(n_components=n_comps)
    # fit the model with embeddings and apply the dimensionality reduction
    flattened_embeddings = pca.fit_transform(embeddings)
    # reshape back to original
    return flattened_embeddings.transpose()


def plt_bar_plot(values, labels, colors=['#cd025c', '#032f3e', '#b635aa', '#e67716', '#e09052']):
    """
    grouped bars with each group in first dim of values and hight in second dim
    :param values:
    :return: plt figure
    """
    plt.clf()
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    # set width of bar
    barWidth = 1 / (values.shape[0] + 1)
    r = np.arange(values.shape[1])

    for idx, bars in enumerate(values):
        ax.bar(r, bars, color=colors[idx], width=barWidth, edgecolor='white', label=labels[idx])
        r = [x + barWidth for x in r]
    ax.legend()

    return fig

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8), zeroth=0):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h, l, s = np.random.uniform(*h, n), np.random.uniform(*l, n), np.random.uniform(*s, n)
    cols = np.stack([colorsys.hls_to_rgb(_h, _l, _s) for _h, _l, _s in zip(h, l, s)], axis=0)
    cols[0] = zeroth
    return matplotlib.colors.ListedColormap(cols)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_contour_from_2d_binary(imgs):
    img_out = []
    device = imgs.device

    for img in imgs:
        img = img.detach().cpu().numpy()
        edge_map = find_boundaries(img)
        edge_map = gaussian(edge_map, sigma=1)
        img_out.append(edge_map)

    return torch.from_numpy(np.array(img_out)).float().to(device)