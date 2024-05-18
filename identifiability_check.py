import ultra.utils.data_utils as data_utils
import numpy as np
import igraph as ig
from collections import defaultdict
from tqdm import tqdm

def identifiability_check(
        data: data_utils.Raw_data,
        context_types=None):
    feature_to_bias = defaultdict(lambda: set())
    bias_factors = set()
    bias_factors_labeler = {}
    bias_id_to_bias_factor = []
    dataset_feature = np.asarray(data.features)
    init_list = np.asarray(data.initial_list)
    for qid in tqdm(range(len(data.initial_list)), desc="Read ranking list"):
        for pos in range(10):
            did = init_list[qid, pos]
            feature = tuple(dataset_feature[did])
            if context_types is None:
                bias = pos
            else:
                bias = f"{context_types[qid, pos]}#{pos}"
            bias_factors.add(bias)
            if bias not in bias_factors_labeler:
                bias_factors_labeler[bias] = len(bias_factors_labeler)
                bias_id_to_bias_factor.append(bias)
            feature_to_bias[feature].add(bias)

    edges = set()
    for _, vtypes in tqdm(feature_to_bias.items(), desc="Construct edges for IG"):
        vtypes = list(vtypes)
        for i in range(len(vtypes)):
            for j in range(i + 1, len(vtypes)):
                edges.add((bias_factors_labeler[vtypes[i]], bias_factors_labeler[vtypes[j]]))

    print(f"\n============ Identifiability Graph Statistics ============")
    print(f"# Features: {len(feature_to_bias)}")
    print(f"# Bias (nodes): {len(bias_factors)}")
    print(f"# Edges: {len(list(edges))}")

    g = ig.Graph(len(bias_factors), edges)
    components = list(g.connected_components())
    components = sorted(components, key=lambda x: len(x), reverse=True)

    print(f"# Components: {len(components)}")
    for i in range(min(10, len(components))):
        print(f"# Nodes in the Top-{i + 1} component: count = {len(components[i])}, ratio = {len(components[i]) / len(bias_factors)}")

    print(f"=========================================================")
    print(f"Identifiable: {len(components) == 1}")

    return g

if __name__ == "__main__":

    import argparse
    import os
    import pickle

    parser = argparse.ArgumentParser(description='Check identifiability for a given dataset')
    parser.add_argument("--data_path", type=str, default="dataset_fully_simulation/K=1")
    parser.add_argument("--file_prefix", type=str, default='train')
    parser.add_argument("--context_path", type=str, default=None)

    args = parser.parse_args()

    context_types = None
    if args.context_path is not None and os.path.isfile(args.context_path):
        with open(args.context_path, "rb") as f:
            context_types, _, _ = pickle.load(f)

    data = data_utils.read_data(args.data_path + "/", args.file_prefix, 0)
    g = identifiability_check(data, context_types)