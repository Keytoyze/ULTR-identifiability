import argparse
import os
import numpy as np
import json
import os
from tqdm import tqdm
import numba
from numba.typed import Dict
import copy
from algorithm import dla, regression_em, two_tower

parser = argparse.ArgumentParser(description='Test on fully simulation dataset')
parser.add_argument("--data_path", type=str, default="dataset_fully_simulation/K=1")
parser.add_argument("--file_prefix", type=str, default='train')
parser.add_argument("--number_of_clicks", type=int, default=1000000)
parser.add_argument("--algorithm", type=str, choices=['dla', 'regression_em', 'two_tower'], default='dla')
parser.add_argument("--epochs", type=int, default=-1, help='training epochs. <0: using default epochs (DLA=20000, RegressionEM=100000, TwoTower=20000)')

# ablation studies for node merging and node intervention
parser.add_argument("--node_merging_strategies", type=str, default="")
parser.add_argument("--random_node_intervention", action='store_true')

parser.add_argument("--no_debias", action='store_true')
args = parser.parse_args()

path = args.data_path

query_to_docs = []
query_to_label = []
query_to_bias_factors = []

test_query_to_docs = []
test_query_to_label = []

algorithm_map = {
    'dla': dla.run,
    'regression_em': regression_em.run, 
    'two_tower': two_tower.run
}
epoch_map = {
    'dla': dla.DEFAUL_EPOCH,
    'regression_em': regression_em.DEFAUL_EPOCH, 
    'two_tower': two_tower.DEFAUL_EPOCH
}

epochs = args.epochs
if epochs < 0:
    epochs = epoch_map[args.algorithm]

label_fin = open(os.path.join(path, "train", "train.labels"))
for line in tqdm(label_fin, "read labels"):
    arr = line.strip().split(' ')                            
    query_to_label.append(arr[1:])


init_list_fin = open(os.path.join(path, "train", "train.init_list"))
for i0, line in tqdm(enumerate(init_list_fin), "read init_list"):
    arr = line.strip().split(' ')
    query_to_docs.append(arr[1:])

if args.random_node_intervention:
    qid = np.random.randint(0, len(query_to_docs))
    # We randomly select a position between 0-3, and connect with another position from 4-10.
    # This method is adhoc for K=2 cases.
    pos_1 = np.random.randint(0, 4)
    pos_2 = np.random.randint(4, 10)
    print(f"Random intervention: query = {qid}, from {pos_1} to {pos_2}")
    label_list = copy.deepcopy(query_to_label[qid])
    doc_list = copy.deepcopy(query_to_docs[qid])
    label_list[pos_1], label_list[pos_2] = label_list[pos_2], label_list[pos_1]
    doc_list[pos_1], doc_list[pos_2] = doc_list[pos_2], doc_list[pos_1]
    query_to_label.append(label_list)
    query_to_docs.append(doc_list)

query_to_label = np.asarray(query_to_label, dtype=np.int)
query_to_docs = np.asarray(query_to_docs, dtype=np.int)

true_label = np.zeros((10000, ), dtype=np.int)

relevance_level = np.asarray([0.1 + 0.9 * (2 ** i - 1) / (2 ** 5 - 1) for i in range(5)])
true_observation = np.asarray([0.68, 0.61, 0.48, 0.34, 0.28, 0.2, 0.11, 0.1, 0.08, 0.06])
datasets_doc = []
datasets_pos = []
datasets_ctr = []
for q in range(len(query_to_docs)):
    for i in range(10):
        d = query_to_docs[q, i]
        true_label[d] = query_to_label[q, i]

        ctr = np.random.binomial(
            n=args.number_of_clicks, 
            p=relevance_level[true_label[d]] * true_observation[i]
        ) / args.number_of_clicks
        datasets_doc.append(d)
        if args.no_debias:
            datasets_pos.append(0)
        else:
            datasets_pos.append(i)
        datasets_ctr.append(ctr)

datasets_doc = np.asarray(datasets_doc)
datasets_pos = np.asarray(datasets_pos)
datasets_ctr = np.asarray(datasets_ctr)

# merge
position_map = {}
if args.node_merging_strategies != "":
    for merge_strategy_item in args.node_merging_strategies.split(","):
        pos1, pos2 = merge_strategy_item.split("-")
        datasets_pos[datasets_pos == int(pos2)] = int(pos1)
        position_map[int(pos2)] = int(pos1)

true_relevance = np.asarray(relevance_level)[true_label]

def ndcg_at_k(pred, label, k=10):
    idx = np.argsort(pred)[::-1][:k]
    label_k = label[idx]
    dcg = np.sum((2 ** label_k - 1) / np.log2(np.arange(2, k+2)))
    label_sorted = np.sort(label)[::-1][:k]
    idcg = np.sum((2 ** label_sorted - 1) / np.log2(np.arange(2, k+2)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def batch_ndcg_at_k(pred, label, k=10):
    results = []
    for i in range(len(pred)):
        results.append(ndcg_at_k(pred[i], label[i], k))
    return sum(results) / len(results)


def training_callback(step, pred_relevance, pred_observation):

    for s, t in position_map.items():
        pred_observation[s] = pred_observation[t]

    pred_click = pred_relevance[datasets_doc] * pred_observation[datasets_pos]

    if step % 100 == 0:
        
        normalize_pred_observation = pred_observation / pred_observation[0]
        normalize_true_observation = true_observation / true_observation[0]            
        error_click_mse = float(np.square(pred_click - datasets_ctr).mean())

        pred = pred_relevance[query_to_docs]
        ndcg_1 = batch_ndcg_at_k(pred, query_to_label, k=1)
        ndcg_3 = batch_ndcg_at_k(pred, query_to_label, k=3)
        ndcg_5 = batch_ndcg_at_k(pred, query_to_label, k=5)
        ndcg_10 = batch_ndcg_at_k(pred, query_to_label, k=10)

        mcc_relevance = float(np.corrcoef(pred_relevance, true_relevance)[0, 1])
        mcc_observation = float(np.corrcoef(pred_observation, true_observation)[0, 1])

        print(f"======= epoch: {step} =======")
        print(f"Pred observation: {list(map(lambda x: round(x, 3), normalize_pred_observation.tolist()))}")
        print(f"True observation: {list(map(lambda x: round(x, 3), normalize_true_observation.tolist()))}")
        print(f"Click MSE: {error_click_mse}")
        print(f"NDCG (1 / 3 / 5 / 10): {ndcg_1:.3f} / {ndcg_3:.3f} / {ndcg_5:.3f} / {ndcg_10:.3f}")
        print(f"MCC (relevance / observation): {mcc_relevance:.3f} | {mcc_observation:.3f}")


algorithm_map[args.algorithm](
    epochs=epochs,
    datasets_doc=datasets_doc,
    datasets_pos=datasets_pos,
    datasets_ctr=datasets_ctr,
    no_debias=args.no_debias,
    callback=training_callback
)