#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def analyze_data_distribution(dataset, dict_users, is_iid=True):
    """
    分析并打印数据分布信息
    :param dataset: 数据集
    :param dict_users: 用户数据索引字典
    :param is_iid: 是否为IID分布
    """
    print("\n数据分布分析" + ("(IID)" if is_iid else "(Non-IID)"))
    print("-" * 40)
    
    # 随机选择3个客户端进行分析
    num_users = len(dict_users)
    sample_users = np.random.choice(range(num_users), min(3, num_users), replace=False)
    
    for user in sample_users:
        user_labels = []
        for idx in dict_users[user]:
            # 获取样本标签
            user_labels.append(dataset[idx][1])
        
        # 统计每个标签的数量
        label_counts = {}
        for label in user_labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        
        print(f"客户端 {user} 的标签分布:")
        for label, count in sorted(label_counts.items()):
            print(f"  标签 {label}: {count} 个样本 ({count/len(user_labels)*100:.2f}%)")
        print("-" * 40)
    
    print("")


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
