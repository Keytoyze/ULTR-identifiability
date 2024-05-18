import numpy as np
import torch

DEFAUL_EPOCH = 20000



def run(epochs, datasets_doc, datasets_pos, datasets_ctr, no_debias, callback):

    last_best_loss = 100000
    pred_relevance = np.zeros((10000, ), dtype=np.float32)
    datasets_ctr_th = torch.FloatTensor(datasets_ctr)
    pred_relevance_th = torch.tensor(pred_relevance, requires_grad=True)
    pred_observation = np.random.randn(10).astype(np.float32)
    if no_debias:
        pred_observation = np.zeros_like(pred_observation)
    pred_observation_th = torch.tensor(pred_observation, requires_grad=True)

    optimizer = torch.optim.Adam([pred_relevance_th, pred_observation_th], lr=1e-2)

    pbar = range(epochs)

    for step in pbar:

        optimizer.zero_grad()

        datasets_pred_relevance_th = torch.sigmoid(pred_relevance_th[datasets_doc])
        datasets_pred_observation_th = torch.sigmoid(pred_observation_th[datasets_pos])
        datasets_pred_click_th = datasets_pred_relevance_th * datasets_pred_observation_th

        loss = torch.nn.BCELoss()(datasets_pred_click_th, datasets_ctr_th)
        if loss.item() < last_best_loss:
            last_best_loss = loss.item()
        loss.backward()
        optimizer.step()

        pred_relevance = torch.sigmoid(pred_relevance_th).detach().numpy()
        pred_observation = torch.sigmoid(pred_observation_th).detach().numpy()

        pred_relevance = np.clip(pred_relevance, 0, 1)
        pred_observation = np.clip(pred_observation, 0, 1)

        callback(step, pred_relevance, pred_observation)