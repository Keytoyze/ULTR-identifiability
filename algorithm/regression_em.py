import numba
from numba.typed import Dict
import numpy as np

EM_step_size = 1
DEFAUL_EPOCH = 100000

@numba.jit(nopython=True)
def training(pred_relevance, pred_observation, datasets_doc, datasets_pos, datasets_ctr):

    datasets_relevance_pred = datasets_ctr / (pred_observation[datasets_pos] + 1e-8)
    datasets_pred_observation = pred_observation[datasets_pos]
    doc_id_to_relevances_sum = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )
    doc_id_to_relevances_count = Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.float64
    )

    for i in range(len(datasets_relevance_pred)):
        doc_id = datasets_doc[i]
        if doc_id not in doc_id_to_relevances_sum:
            doc_id_to_relevances_sum[doc_id] = 0
            doc_id_to_relevances_count[doc_id] = 1e-8
        p_e0_r1_c0 = (1 - datasets_pred_observation[i]) * datasets_relevance_pred[i] / (1 - datasets_pred_observation[i] * datasets_relevance_pred[i])
        p_r1 = datasets_ctr[i] + (1 - datasets_ctr[i]) * p_e0_r1_c0
        doc_id_to_relevances_sum[doc_id] += p_r1
        doc_id_to_relevances_count[doc_id] += 1
    for doc_id in doc_id_to_relevances_sum:
        pred_relevance[doc_id] = pred_relevance[doc_id] * (1 - EM_step_size) + doc_id_to_relevances_sum[doc_id] / doc_id_to_relevances_count[doc_id] * EM_step_size

    for pos_id in range(len(pred_observation)):
        ptr = datasets_pos == pos_id
        doc_relevance = pred_relevance[datasets_doc[ptr]]
        if len(doc_relevance) == 0:
            continue
        doc_ctr = datasets_ctr[ptr]
        cur_observation = pred_observation[pos_id]

        p_e1_r0_c0 = cur_observation * (1 - doc_relevance) / (1 - cur_observation * doc_relevance)
        p_e1 = doc_ctr + (1 - doc_ctr) * p_e1_r0_c0

        pred_observation[pos_id] = pred_observation[pos_id] * (1 - EM_step_size) + np.mean(p_e1) * EM_step_size

def run(epochs, datasets_doc, datasets_pos, datasets_ctr, no_debias, callback):

    pred_relevance = np.random.random((10000, )) / 2
    pred_observation = np.random.random((10, ))
    if no_debias:
        pred_observation = np.ones_like(pred_observation)

    pbar = range(epochs)

    for step in pbar:

        training(pred_relevance, pred_observation, datasets_doc, datasets_pos, datasets_ctr)
        pred_relevance = np.clip(pred_relevance, 0, 1)
        pred_observation = np.clip(pred_observation, 0, 1)

        callback(step, pred_relevance, pred_observation)