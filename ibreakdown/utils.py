import numpy as np


def normalize_array(instance):
    return instance.reshape(1, -1)


def magnituge(v):
    return np.linalg.norm(np.array(v[1]), axis=0)


def to_matrix(data):
    return data if not hasattr(data, 'values') else data.values


def features_groups(num_features, check_interactions=True):
    result = list(range(0, num_features))
    if check_interactions:
        for i in range(0, num_features):
            for j in range(i + 1, num_features):
                result.append((i, j))
    return result


def feature_group_values(feature_groups, instance):
    featrue_values = [instance[:, group][0] for group in feature_groups]
    return featrue_values
