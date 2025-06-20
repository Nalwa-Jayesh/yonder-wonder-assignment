# Helper functions 
import numpy as np
from collections import defaultdict

def stratified_split(dataset, train_ratio=0.7, random_seed=42):
    """
    Perform stratified split to ensure balanced age distribution in train/test sets
    """
    np.random.seed(random_seed)
    age_groups = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        age_groups[label].append(idx)
    train_indices = []
    val_indices = []
    for age, indices in age_groups.items():
        np.random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
    return train_indices, val_indices 