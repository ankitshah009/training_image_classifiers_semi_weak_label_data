import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
import math
import numpy as np
from tqdm import tqdm

CLASS_SET = [b"airplane", b"automobile", b"bird", b"cat", b"deer",
             b"dog", b"frog", b"horse", b"ship", b"truck"]


class Stats :
    def __init__(self, all_data, is_train) :
        self.all_data = all_data
        self.used_set = {i : set () for i in range (10)}
        self.label_count = {i : [] for i in range (10)}
        self.is_train = is_train
        self.sparsity = []

    def update(self, bag) :
        for filename in bag[0] :
            label = self.all_data[filename]
            self.used_set[label].add (filename)

        count_vec = bag[1]
        sparsity = sum([1 if i == 0 else 0 for i in count_vec]) / len(count_vec)
        self.sparsity.append(sparsity)
        for label in range (10) :
            self.label_count[label].append (count_vec[label])

    def summary(self) :
        text = ""
        for label in range (10) :
            use_instance = len (self.used_set[label])
            tot_instance = 5000 if self.is_train else 1000
            presence_rate = sum ([1 if count > 0 else 0 for count in self.label_count[label]]) / len(self.label_count[label])
            avg_count = [count for count in self.label_count[label] if count > 0]
            avg_count = sum (avg_count) / len (avg_count)
            text += f"[{label}]: {use_instance / tot_instance:.4f}, presence rate: {presence_rate:.4f}, count avg: {avg_count:.4f}\n"
        sparsity = np.array(self.sparsity)
        print(f"sparsity: avg. {sparsity.mean():.4f} std. - {sparsity.std():.4f}")
        print (text)

def exponential(beta):
    return math.floor (np.random.exponential (beta))

def poisson(lambda_):
    return np.random.poisson (lambda_)

def uniform(beta):
    global bag_size
    return int(np.random.uniform (low=0, high=bag_size))

def one_bag(data, data_p, bag_size=8) :
    global beta, distribution
    total_num = np.zeros (10).astype (int)
    data_p = data_p / data_p.sum()
    bags = np.random.choice(list(range(len(data))), p=data_p, size=bag_size, replace=False)
    return bags

def generate_positve_samples(all_data, is_train, n_samples=3000, bag_size=16) :
    """Gener
    Note: all instances are used more than once
    """
    global MAX_SAMPLE_TIMES
    all_data_list = list(all_data.keys())
    pos_bags = []
    stat = Stats (all_data, is_train)
    pos_p = np.ones (len (all_data)) * MAX_SAMPLE_TIMES
    for _ in tqdm (range (n_samples)) :
        bag = one_bag (all_data_list, pos_p, bag_size=bag_size)
        for idx in bag :
            pos_p[idx] -= 1

        all_files = []
        lens = [0 for _ in range (10)]
        for idx in bag :
            all_files.append (all_data_list[idx])
            lens[all_data[all_data_list[idx]]] += 1

        """ permute """
        all_files = np.random.permutation(all_files)
        assert len (all_files) == bag_size
        pos_bags.append ((all_files, lens))
        stat.update ((all_files, lens))
    return pos_bags, stat

def save_bags(pos_bags, path, tag, trial, is_train=True) :
    import os
    if not os.path.exists (path) :
        os.mkdir (path)
    path = os.path.join (path, tag)
    if not os.path.exists (path) :
        os.mkdir (path)
    pos_file = os.path.join (path, "train.trial={}.csv".format (trial) if is_train else "test.trial={}.csv".format (trial))
    with open (pos_file, "w") as f :
        f.write ("instances, count\n")
        for files, counts in pos_bags :
            files = [i.decode () for i in files]
            line = "{}, {}\n".format (" ".join (files), " ".join ([str(i) for i in counts]))
            f.write (line)


def load_all_data(pathes) :
    """
    :param pathes: List of pathes for cifar10
    :return: Dict[filename: Str, label: int]
    """
    all = {}
    for path in pathes :
        dta = __unpickle (path)
        filenames = dta[b"filenames"]
        labels = dta[b"labels"]
        for file, label in zip (filenames, labels) :
            assert file not in all.keys ()
            all[file] = label
    return all


def __unpickle(file) :
    with open (file, 'rb') as fo :
        dict = pickle.load (fo, encoding='bytes')
    return dict


def __id_to_label(path="./data/cifar-10-batches-py/batches.meta") :
    dta = __unpickle (path)
    return dta[b'label_names']


if __name__ == '__main__' :
    meta = "./data/cifar-10-batches-py/batches.meta"
    train_path = ["./data/cifar-10-batches-py/data_batch_%d" % i for i in [1, 2, 3, 4, 5]]
    test_path = ["./data/cifar-10-batches-py/test_batch"]
    n_samples_train = 3000
    n_samples_test = 600
    bag_size = 32
    train = load_all_data (train_path)
    test = load_all_data (test_path)
    id_to_label = __id_to_label ()
    import argparse

    parser = argparse.ArgumentParser ("Data Generation")
    parser.add_argument ("--bag-size", "-bs", default=8, type=int)
    parser.add_argument ("--n-train-samples", "-ntr", default=1500, type=int)
    parser.add_argument ("--n-test-samples", "-nte", default=300, type=int)
    parser.add_argument ("--reuse", default=2, type=int)
    parser.add_argument ("--trial", default=5, type=int)
    parser.add_argument ("--beta", default=2.0, type=float)

    args = parser.parse_args ()
    beta = args.beta
    MAX_SAMPLE_TIMES = args.reuse
    supervised_train = set ()
    supervised_test = set ()
    for j in range (args.trial):
        random.seed (2020 + j)
        np.random.seed (2020 + j)
        print ("==> generate data for trail {}".format (j))
        pos_bags_train, stat = generate_positve_samples (train,is_train=True,
                                                         n_samples=args.n_train_samples,
                                                         bag_size=args.bag_size)
        stat.summary ()
        pos_bags_test, stat = generate_positve_samples (test,is_train=False,
                                                        n_samples=args.n_test_samples,
                                                        bag_size=args.bag_size)
        stat.summary ()
        tag = "reuse={}.n={}.bag={}.beta={}".format (MAX_SAMPLE_TIMES, args.n_train_samples, args.bag_size, beta)
        tag += ".uniform"
        save_bags (pos_bags_train, trial=j, path="./meta",
                   tag=tag,
                   is_train=True)

        save_bags (pos_bags_test, trial=j, path="./meta",
                   tag=tag,
                   is_train=False)
