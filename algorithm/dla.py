import numba
from numba.typed import Dict
import numpy as np

DEFAUL_EPOCH = 20000

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
        doc_id_to_relevances_sum[doc_id] += datasets_ctr[i] * datasets_pred_observation[i]
        doc_id_to_relevances_count[doc_id] += datasets_pred_observation[i] ** 2
    for doc_id in doc_id_to_relevances_sum:
        pred_relevance[doc_id] = doc_id_to_relevances_sum[doc_id] / doc_id_to_relevances_count[doc_id]

    for pos_id in range(len(pred_observation)):
        ptr = datasets_pos == pos_id
        doc_relevance = pred_relevance[datasets_doc[ptr]]
        if len(doc_relevance) == 0:
            continue
        doc_ctr = datasets_ctr[ptr]
        pred_observation[pos_id] = np.mean(doc_ctr * doc_relevance / (doc_relevance * doc_relevance + 1e-8))


def run(epochs, datasets_doc, datasets_pos, datasets_ctr, no_debias, callback):

    pred_relevance = np.random.random((10000, ))
    pred_observation = np.random.random((10, ))
    if no_debias:
        pred_observation = np.ones_like(pred_observation)

    pbar = range(epochs)

    for step in pbar:

        training(pred_relevance, pred_observation, datasets_doc, datasets_pos, datasets_ctr)
        pred_relevance = np.clip(pred_relevance, 0, 1)
        pred_observation = np.clip(pred_observation, 0, 1)

        callback(step, pred_relevance, pred_observation)