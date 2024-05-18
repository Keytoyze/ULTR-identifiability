import ultra.utils.data_utils as data_utils
import numpy as np
import igraph as ig
from collections import defaultdict
from tqdm import tqdm
import pickle
import copy
from sklearn.neighbors import KDTree
import argparse

parser = argparse.ArgumentParser(description='Simulate context types on a dataset, then do node merging.')
parser.add_argument("--data_path", type=str, default="Yahoo_letor/tmp_data")
parser.add_argument("--file_prefix", type=str, default="train")
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--context_path", type=str, default="Yahoo_letor/tmp_data/context.pkl")
parser.add_argument("--n_context", type=int, default=5000)
parser.add_argument("--n_processes", type=int, default=1)

args = parser.parse_args()

data = data_utils.read_data(args.data_path, "train", 0)

np.random.seed(args.random_seed)

context = np.random.randint(0, args.n_context, (len(data.initial_list), 10))
vertical_feature = np.random.normal(loc=0, scale=0.35, size=(args.n_context, 10))


feature_to_bias = defaultdict(lambda: set())
bias_factors = set()
bias_factors_labeler = {}
bias_id_to_bias_feature = []
bias_id_to_bias_factor = []
dataset_feature = data.features

for qid in tqdm(range(len(data.initial_list)), desc="Read queries"):
    for pos in range(10):
        did = data.initial_list[qid, pos]
        feature = tuple(data.features[did])
        bias = f"{context[qid, pos]}#{pos}"
        bias_factors.add(bias)
        if bias not in bias_factors_labeler:
            bias_factors_labeler[bias] = len(bias_factors_labeler)
            bias_feature = list(vertical_feature[context[qid, pos]])
            bias_feature.append(pos * 10)
            bias_id_to_bias_feature.append(bias_feature)
            bias_id_to_bias_factor.append(bias)
        feature_to_bias[feature].add(bias)

edges = set()
for x, vtypes in tqdm(feature_to_bias.items(), desc="Construct IG edges"):
    vtypes = list(vtypes)
    for i in range(len(vtypes)):
        for j in range(i + 1, len(vtypes)):
            edges.add((bias_factors_labeler[vtypes[i]], bias_factors_labeler[vtypes[j]]))

print(f"# Features: {len(feature_to_bias)}")
print(f"# Bias: {len(bias_factors)}")
print(f"# Edges: {len(list(edges))}")

g = ig.Graph(len(bias_factors), edges)
components = list(g.connected_components())
components = sorted(components, key=lambda x: len(x), reverse=True)

print(f"#Components: {len(components)}")
for i in range(10):
    print(len(components[i]), len(components[i]) / len(bias_factors))


edges_to_merge = []
edges_to_merge_dict = {}

main_vectors = []
main_context = []
cost = 0

for node in components[0]:
    v = bias_id_to_bias_feature[node]
    main_vectors.append(v)
    main_context.append(node)
main_kd_tree = KDTree(main_vectors, leaf_size=2)

components_delete = copy.deepcopy(components[1:])

def get_result(data):
    cur_gi, components_delete, main_context, main_kd_tree = data
    cur_g = components_delete[cur_gi]
    cur_context = [bias_id_to_bias_feature[n] for n in cur_g]
    distance, index = main_kd_tree.query(cur_context, k=1)
    min_idx = np.argmin(distance)
    min_src_node = cur_g[min_idx]
    min_dst_node = main_context[index[min_idx][0]]

    v1 = vertical_feature[int(bias_id_to_bias_factor[min_src_node].split("#")[0])].reshape((1, -1))
    v2 = vertical_feature[int(bias_id_to_bias_factor[min_dst_node].split("#")[0])].reshape((1, -1))

    dis = float(np.linalg.norm(v1 - v2, axis=-1)[0])
    return dis, min_src_node, min_dst_node, cur_gi

if __name__ == "__main__":
    import multiprocessing

    pool = multiprocessing.Pool(processes=args.n_processes)

    for iter in tqdm(range(len(components_delete))):

        cur_min_cost = 10000000
        cur_min_graph = None
        cur_min_graph_i = None
        cur_min_cost_edge = None

        d = []
        for iii in range(len(components_delete)):
            data = iii, components_delete, main_context, main_kd_tree
            d.append(data)

        results = map(get_result, d)

        for dis, min_src_node, min_dst_node, cur_gi in results:
            if dis < cur_min_cost:
                cur_min_cost = dis
                cur_min_cost_edge = (min_src_node, min_dst_node)
                cur_min_graph = components_delete[cur_gi]
                cur_min_graph_i = cur_gi

        cost += cur_min_cost
        del components_delete[cur_min_graph_i]
        edges_to_merge_dict[bias_id_to_bias_factor[min_src_node]] = bias_id_to_bias_factor[min_dst_node]
        edges.add(cur_min_cost_edge)

        for node in cur_min_graph:
            v = bias_id_to_bias_feature[node]
            main_vectors.append(v)
            main_context.append(node)
        main_kd_tree = KDTree(main_vectors, leaf_size=2)

    print("After merging: ")
    print(f"# Features: {len(feature_to_bias)}")
    print(f"# Bias: {len(bias_factors)}")
    print(f"# Edges: {len(list(edges))}")
    print(f"# Cost: {cost}")

    g = ig.Graph(len(bias_factors), edges)
    components = list(g.connected_components())
    print(f"# Components: {len(components)}")

    with open(args.context_path, 'wb') as f:
        pickle.dump([context, vertical_feature, edges_to_merge_dict], f)