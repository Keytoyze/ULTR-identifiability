import argparse
import numpy as np
import json
import lightgbm
import numba
import ultra.utils.data_utils as data_utils
import pickle
from numba.typed import Dict


parser = argparse.ArgumentParser(description='Test on fully simulation dataset')
parser.add_argument("--data_path", type=str, default="Yahoo_letor/tmp_data")
parser.add_argument("--context_path", type=str, default="Yahoo_letor/tmp_data/context.pkl")
parser.add_argument("--click_model", type=str, default='click_model.json')
parser.add_argument("--number_of_clicks", type=int, default=1000000)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--no_debias", action='store_true')
parser.add_argument("--no_identification", action='store_true')

args = parser.parse_args()

click_config = json.load(open(args.click_model))
w = np.asarray(click_config['w']).reshape((1, -1))

train_data = data_utils.read_data(args.data_path, "train", 0)
test_data = data_utils.read_data(args.data_path, "test", 0)

try:
    max_candidate_num = max(train_data.rank_list_size, test_data.rank_list_size)
    train_data.pad(max_candidate_num)
    test_data.pad(max_candidate_num)
except:
    pass

with open(args.context_path, "rb") as f:
    query_to_context, context_to_bf, merging_dict = pickle.load(f)
    context_to_ratio = np.sum(w * context_to_bf, axis=1) + 1
    context_to_ratio = np.where(context_to_ratio > 0, context_to_ratio, np.zeros_like(context_to_ratio))

query_to_docs = np.asarray(train_data.initial_list)[:, :10].astype(np.int)
query_to_label = np.asarray(train_data.labels)[:, :10].astype(np.int)
test_query_to_docs = np.asarray(test_data.initial_list).astype(np.int)
test_query_to_label = np.asarray(test_data.labels).astype(np.int)
docs_to_features = np.asarray(train_data.features)
test_docs_to_features = np.asarray(test_data.features)

label_to_true_relevance = np.asarray(click_config['click_prob'])
position_to_true_observation = np.asarray(click_config['exam_prob'])
context_num = np.max(query_to_context) + 1

print(f"# Queries: {len(query_to_docs)}, # Docs: {len(docs_to_features)}, # Bias: {context_num * 10}")

datasets_features = []
datasets_bias_id = []
datasets_ctr = []
datasets_labels = []
datasets_group = []
datasets_true_observation = []
datasets_true_relevance = []
datasets_doc_id = []
doc_id_to_features = []
feature_labeler = {}
for q in range(len(query_to_docs)):
    cur_group = 0
    cur_q_to_doc_id = []
    for i in range(10):
        cur_doc = query_to_docs[q, i]
        if cur_doc == -1:
            continue
        cur_label = query_to_label[q, i]
        cur_relevance = label_to_true_relevance[cur_label]
        cur_context = query_to_context[q, i]
        
        cur_observation = position_to_true_observation[i] ** context_to_ratio[cur_context]

        ctr = np.random.binomial(
            n=args.number_of_clicks, 
            p=cur_relevance * cur_observation
        ) / args.number_of_clicks
        datasets_true_observation.append(cur_observation)
        datasets_true_relevance.append(cur_relevance)
        datasets_features.append(docs_to_features[cur_doc])
        tp = tuple(docs_to_features[cur_doc])
        if tp not in feature_labeler:
            feature_labeler[tp] = len(feature_labeler)
            doc_id_to_features.append(docs_to_features[cur_doc])
        datasets_doc_id.append(feature_labeler[tp])

        if args.no_identification:
            merging_context, merging_position = cur_context, i
        elif args.no_debias:
            merging_context, merging_position = 0, 0
        else: # node merging
            cur_context_repr = f"{cur_context}#{i}"
            merging_context_repr = merging_dict.get(cur_context_repr, cur_context_repr)
            merging_context, merging_position = merging_context_repr.split("#")
        datasets_bias_id.append(int(merging_context) * 10 + int(merging_position))
        datasets_ctr.append(ctr)
        datasets_labels.append(cur_label)
        cur_group += 1
    datasets_group.append(cur_group)

q_to_did = []
for q in range(len(query_to_docs)):
    cur_group = 0
    cur_q_to_doc_id = []
    for i in range(10):
        if query_to_docs[q, i] < 0:
            doc_id = 0
        else:
            doc_id = feature_labeler[tuple(docs_to_features[query_to_docs[q, i]])]
        cur_q_to_doc_id.append(doc_id)
    q_to_did.append(cur_q_to_doc_id)
q_to_did = np.asarray(q_to_did, dtype=np.int)

datasets_features = np.asarray(datasets_features)
datasets_bias_id = np.asarray(datasets_bias_id)
datasets_ctr = np.asarray(datasets_ctr)
datasets_labels = np.asarray(datasets_labels)
datasets_true_observation = np.asarray(datasets_true_observation)
datasets_doc_id = np.asarray(datasets_doc_id)
doc_id_to_features = np.asarray(doc_id_to_features)

datasets_pred_relevance = np.random.random((len(datasets_features), ))
bias_id_to_pred_observation = np.random.random((context_num * 10, ))
doc_id_to_pred_relevance = np.random.random((len(feature_labeler), ))

@numba.jit()
def ndcg_at_k(pred, label, k=10):
    idx = np.argsort(pred)[::-1][:k]
    label_k = label[idx]
    dcg = np.sum((2 ** label_k - 1) / np.log2(np.arange(2, k+2)))
    label_sorted = np.sort(label)[::-1][:k]
    idcg = np.sum((2 ** label_sorted - 1) / np.log2(np.arange(2, k+2)))
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

@numba.jit()
def batch_ndcg_at_k(pred, label, k=10):
    results = []
    for i in range(len(pred)):
        valid_flag = label[i] >= 0
        cur_k = min(np.sum(valid_flag), k)
        p = pred[i][valid_flag]
        l = label[i][valid_flag]
        results.append(ndcg_at_k(p, l, cur_k))
    return sum(results) / len(results)

pbar = range(args.epochs)

best_error_click = 1000000
als_lr = 1
clip_high = 1
clip_low = 0

@numba.jit()
def train_relevance(datasets_doc_id, datasets_ctr, datasets_pred_observation, 
                    datasets_pred_relevance):
    doc_id_to_up = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    doc_id_to_down = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    for i in range(len(datasets_doc_id)):
        doc_id = datasets_doc_id[i]
        if doc_id not in doc_id_to_up:
            doc_id_to_up[doc_id] = 0
            doc_id_to_down[doc_id] = 1e-8
        doc_id_to_up[doc_id] += datasets_ctr[i] * datasets_pred_observation[i]
        doc_id_to_down[doc_id] += datasets_pred_observation[i] ** 2
    doc_id_to_pred_relevance = np.zeros((len(doc_id_to_up), ), dtype=np.float)
    for doc_id in doc_id_to_up:
        doc_id_to_pred_relevance[doc_id] = doc_id_to_up[doc_id] / doc_id_to_down[doc_id]
    doc_id_to_pred_relevance = np.clip(doc_id_to_pred_relevance, clip_low, clip_high)
    datasets_pred_relevance = doc_id_to_pred_relevance[datasets_doc_id] * als_lr + datasets_pred_relevance * (1 - als_lr)
    datasets_pred_click = datasets_pred_relevance * datasets_pred_observation
    error_click = float(np.square(datasets_pred_click - datasets_ctr).mean())
    return doc_id_to_pred_relevance, datasets_pred_relevance, datasets_pred_click, error_click


@numba.jit()
def train_observation(datasets_bias_id, datasets_ctr, datasets_pred_relevance, 
                        datasets_pred_observation):
    bias_to_up = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    bias_to_down = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    for i in range(len(datasets_bias_id)):
        bias_id = datasets_bias_id[i]
        if bias_id not in bias_to_up:
            bias_to_up[bias_id] = 0
            bias_to_down[bias_id] = 1e-8
        bias_to_up[bias_id] += datasets_ctr[i] * datasets_pred_relevance[i]
        bias_to_down[bias_id] += datasets_pred_relevance[i] ** 2
    bias_id_to_pred_observation = np.zeros((context_num * 10, ), dtype=np.float)
    for bias_id in bias_to_up:
        bias_id_to_pred_observation[bias_id] = bias_to_up[bias_id] / bias_to_down[bias_id]
    bias_id_to_pred_observation = np.clip(bias_id_to_pred_observation, clip_low, clip_high)
    datasets_pred_observation = als_lr * bias_id_to_pred_observation[datasets_bias_id] + (1 - als_lr) * datasets_pred_observation
    datasets_pred_click = datasets_pred_relevance * datasets_pred_observation
    error_click = float(np.square(datasets_pred_click - datasets_ctr).mean())
    return bias_id_to_pred_observation, datasets_pred_observation, datasets_pred_click, error_click


for step in pbar:

    datasets_pred_observation = bias_id_to_pred_observation[datasets_bias_id]
    datasets_pred_observation = datasets_pred_observation
    if step == len(pbar) - 1:
        model = lightgbm.LGBMRegressor(
            num_leaves=255,
            learning_rate=0.1,
            n_estimators=500,
            min_data_in_leaf=0,
            min_sum_hessian_in_leaf=100,
            n_jobs=8,
        )
        # Fit relevance model
        target = datasets_ctr / (datasets_pred_observation + 1e-5)
        model.fit(datasets_features, target)

    doc_id_to_pred_relevance, datasets_pred_relevance, datasets_pred_click, error_click = train_relevance(datasets_doc_id, datasets_ctr, datasets_pred_observation, datasets_pred_relevance)

    train_features = docs_to_features[query_to_docs]
    Q, L, F = train_features.shape

    if step == len(pbar) - 1:

        train_scores = doc_id_to_pred_relevance[q_to_did]
        train_ndcg_1 = batch_ndcg_at_k(train_scores, query_to_label, k=1)
        train_ndcg_3 = batch_ndcg_at_k(train_scores, query_to_label, k=3)
        train_ndcg_5 = batch_ndcg_at_k(train_scores, query_to_label, k=5)
        train_ndcg_10 = batch_ndcg_at_k(train_scores, query_to_label, k=10)

        test_features = test_docs_to_features[test_query_to_docs]
        Q, L, F = test_features.shape
        test_features = test_features.reshape((Q * L, F))
        test_scores = model.predict(test_features).reshape((Q, L))
        ndcg_1 = batch_ndcg_at_k(test_scores, test_query_to_label, k=1)
        ndcg_3 = batch_ndcg_at_k(test_scores, test_query_to_label, k=3)
        ndcg_5 = batch_ndcg_at_k(test_scores, test_query_to_label, k=5)
        ndcg_10 = batch_ndcg_at_k(test_scores, test_query_to_label, k=10)

        test_scores_flat = test_scores.reshape((-1, ))
        test_query_to_label_flat = test_query_to_label.reshape((-1, ))
        valid_flag = test_query_to_label_flat != -1
    else:
        ndcg_1 = 0
        ndcg_3 = 0
        ndcg_5 = 0
        ndcg_10 = 0
        train_ndcg_1 = 0
        train_ndcg_3 = 0
        train_ndcg_5 = 0
        train_ndcg_10 = 0

    
    datasets_pred_click = datasets_pred_relevance * datasets_pred_observation
    error_click = float(np.square(datasets_pred_click - datasets_ctr).mean())
    mcc_relevance = float(np.corrcoef(datasets_pred_relevance, datasets_true_relevance)[0, 1])
    print(f"[{step}][Relevance] Error click = {error_click}")

    bias_id_to_pred_observation, datasets_pred_observation, datasets_pred_click, error_click = train_observation(datasets_bias_id, datasets_ctr, datasets_pred_relevance, datasets_pred_observation)
    print(f"[{step}][Observation] Error click = {error_click}")

print("Result: ")
print({
    "ndcg_1": float(ndcg_1),
    "ndcg_3": float(ndcg_3),
    "ndcg_5": float(ndcg_5),
    "ndcg_10": float(ndcg_10),
    "train_ndcg_1": float(train_ndcg_1),
    "train_ndcg_3": float(train_ndcg_3),
    "train_ndcg_5": float(train_ndcg_5),
    "train_ndcg_10": float(train_ndcg_10),
    "click_mae": float(error_click),
    "mcc_relevance": float(mcc_relevance)
})

