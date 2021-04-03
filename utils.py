'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import pandas as pd
from collections import defaultdict, OrderedDict
from torch.distributions.poisson import Poisson
from torch.distributions.exponential import Exponential

import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.metrics import precision_score, mean_absolute_error
import numpy as np
from tqdm import tqdm
import copy
from scipy.optimize import linear_sum_assignment
CUDA = torch.cuda.is_available()

# poiss_obj = poisson()
# exp_obj = expon()

def enumerate_all(bag_size, n_class):
    global_lis = []
    dfs(bag_size, n_class, [], global_lis)
    return global_lis

def dfs(sum, left, lis, global_lis):
    if left == 0 and sum == 0:
        global_lis.append(copy.copy(lis))
    if sum >= 0 and left > 0:
        for i in range(sum+1):
            lis.append(i)
            dfs(sum - i, left-1, lis, global_lis)
            lis.pop()


# def search_count_one(y_pred, bag_size, n_class, distribution):
#     # y_pred: (n_class)
#     assert len(y_pred) == n_class
#     if distribution == "exponential":
#         dist = Exponential(y_pred)
#     elif distribution == "poisson":
#         dist = Poisson(y_pred)
#     else:
#         raise NotImplementedError
#     best_com = None
#     best_log_prob = -1e9
#     for com in enumerate_all(bag_size, n_class):
#         x = torch.tensor(com).float()
#         if CUDA:
#             x = x.cuda()
#         dist.log_prob()
#         if log_prob > best_log_prob:
#             best_com = com
#             best_log_prob = log_prob
#     assert best_com is not None
#     res = torch.tensor (best_com).long ()
#     if CUDA:
#         res = res.cuda()
#     return res

def search_count_all(y_pred, bag_size, n_class):
    # (B, n_class)
    print("Estimating all the counts")
    dist = Poisson(y_pred)

    max_log_p = torch.zeros(y_pred.size(0))
    max_log_p.fill_(-float("inf"))
    count_mat = torch.zeros_like(y_pred)

    if CUDA:
        max_log_p = max_log_p.cuda()
        count_mat = count_mat.cuda()

    for com in tqdm(enumerate_all (bag_size, n_class)):
        x = torch.tensor(com).float()
        assert x.sum().item() == float(bag_size)
        if CUDA:
            x = x.cuda()
        log_p = dist.log_prob(x).sum(dim=1) # (B, )
        mask = log_p > max_log_p
        max_log_p = torch.where(mask, log_p, max_log_p)
        count_mat[mask] = x

    print("expected results")
    print(y_pred[0:4].round())
    print("estimated results")
    print(count_mat[0:4])
    return count_mat

def _seach_one(y_log_pred, y_true):
    # (bag, n_class)
    # (n_class)
    # return new_y_pred (bag)

    bag, n_class = y_log_pred.shape
    cost_matrix = np.zeros((bag, bag))
    cost_matrix.fill(np.inf)
    target_bag_label = np.zeros(bag, dtype=int)
    i = 0
    for label in range (n_class) :
        class_count = int(y_true[label])
        target_bag_label[i:i+class_count] = label
        i += class_count

    assert i == bag, (i, y_true)
    for instance in range(bag):
        # for each instance, get the transition cost from label i -> j
        for target in range(bag):
            target_label = target_bag_label[target]
            cost = -y_log_pred[instance][target_label]
            # assign it to the cost matrix
            cost_matrix[instance][target] = cost

    instance_ids, assignment_ids = linear_sum_assignment(cost_matrix)
    new_y_pred = torch.zeros(bag).long()
    if CUDA:
        new_y_pred = new_y_pred.cuda()
    for instance_id, assignment_id in zip(instance_ids, assignment_ids):
        new_y_pred[instance_id] = target_bag_label[assignment_id]
    NLL = cost_matrix[instance_ids, assignment_ids].sum()
    return new_y_pred, NLL

def _get_entropy(y_count):
    # (B, n_class) calculate the entropy for each bag
    # get the probability distribution for each class in a bag
    assert not y_count.requires_grad
    y_count = y_count + 0.1
    prob_distribution = (y_count) / y_count.sum(dim=1).unsqueeze(1)
    entropy = - prob_distribution * (prob_distribution).log2() # (b, n_class)
    assert not torch.isnan(entropy).any()
    entropy_sum = entropy.sum(dim=1) # (B, )
    assert (entropy_sum >= 0).all() and (entropy_sum <= math.log2(10)).all()
    return entropy_sum

def _search_batch(y_prob, y_true):
    """Search
    y_pred: FloatTensor: (*, bag, n_class)
    y_true: LongTensor: (*, n_class)
    return shape:
    """
    y_pred = []
    nlls = []
    n = y_prob.size(0)
    log_y_prob = y_prob.log2().cpu().numpy()
    y_true = y_true.cpu().numpy()
    for i in range(n):
        new_y_pred, NLL = _seach_one(log_y_prob[i], y_true[i])
        y_pred.append(new_y_pred.unsqueeze(0))
        nlls.append(NLL)
    y_pred = torch.cat(y_pred, dim=0) # (batch, bag)
    return y_pred

def get_count_from_instance_prediction(normalized_activation_map):
    """Search
    return the count_vec: (batch, n_class)
    """
    assert len (normalized_activation_map.shape) == 3
    batch, _, n_class = normalized_activation_map.size()
    pred = normalized_activation_map.argmax (dim=2)
    count_vec = torch.zeros(batch, n_class)
    if CUDA:
        count_vec = count_vec.cuda ()
    for label in range(n_class):
        count_vec[:, label] = torch.sum(pred == label, dim=1)
    return count_vec

def evaluate_instance_prediction(normalized_activation_map, instance_targets):
    if len(normalized_activation_map.shape) == 3:
        pred = normalized_activation_map.argmax(dim=2)  # (bs, bag)
        bs, bag_size, nclass = normalized_activation_map.size ()
    elif len(normalized_activation_map.shape) == 2:
        pred = normalized_activation_map
        bs, bag_size = normalized_activation_map.size ()
        nclass = 10
    else:
        raise NotImplementedError
    n_class = 10
    tp_mat = torch.zeros(n_class, bag_size)
    fp_mat = torch.zeros(n_class, bag_size)

    for i in range(bs):
        for label in range(n_class):
            cnt = int((instance_targets[i] == label).sum().item())
            tp = int(((instance_targets[i] == label) * (instance_targets[i] == pred[i])).sum().item())
            fp = int(((pred[i] == label) * (instance_targets[i] != label)).sum().item())
            tp_mat[label, cnt-1] += tp
            fp_mat[label, cnt-1] += fp

    prec_heatmap = tp_mat / (tp_mat + fp_mat)
    print("prec_heatmap")
    print(prec_heatmap)

def _compute_instance_metrics(normalized_activation_map, instance_targets):
    # (bs, bag, nclass)
    # instance_targets (B, bag)
    if len(normalized_activation_map.shape) == 3:
        pred = normalized_activation_map.argmax(dim=2)  # (bs, bag)
        bs, bag_size, nclass = normalized_activation_map.size ()
    elif len(normalized_activation_map.shape) == 2:
        pred = normalized_activation_map
        bs, bag_size = normalized_activation_map.size ()
        nclass = 10
    else:
        raise NotImplementedError

    instance_targets = instance_targets.cpu().numpy()
    pred = pred.cpu().numpy()
    precision = precision_score(instance_targets.flatten(),
                                pred.flatten(),
                                average="macro",
                                labels=list(range(nclass)),
                                zero_division=0,
                                )
    precisions = []
    sparsity_lis = []
    for i in range (bs) :
        sparsity = (instance_targets[i]==0).mean()
        sparsity_lis.append(sparsity)
        prec = precision_score(instance_targets[i],
                                pred[i],
                                average="macro",
                                labels=list(range(nclass)),
                                zero_division=0,)
        precisions.append(prec)

    # d = defaultdict(list)
    # d2 = OrderedDict()
    # precisions = np.array(precisions)
    # entropy_lis = np.array(sparsity_lis).round(1)
    # for idx, v in enumerate(entropy_lis):
    #     d[v].append(precisions[idx])
    # for idx, v in d.items():
    #     d2[idx] = sum(v) / len(v)
    # print("precision grouped by sparsity")
    # print(d2)
    return precision

def _compute_bag_metrics(y_pred, bag_targets):
    # y_pred:      (b, n_class)
    # bag_targets: (b, n_class)
    y_pred = y_pred.cpu().numpy()
    bag_targets = bag_targets.cpu().numpy()
    pos_label_mask = bag_targets > 0
    count_pred = y_pred[pos_label_mask].flatten().astype(float)
    count_true = bag_targets[pos_label_mask].flatten().astype(float)
    # [0, 2, 0], [1, 1, 2] -> MAE = 1
    mse = mean_absolute_error(count_true, count_pred)
    precision = precision_score((bag_targets > 0).astype(int),
                                (y_pred > 0).astype(int),
                                average='samples',
                                labels=list(range(10)),
                                zero_division=0)
    return precision, mse

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
term_width = 125

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
