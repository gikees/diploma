import torch
import random
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.utils import coalesce
from sklearn.metrics import roc_auc_score, average_precision_score


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def forward(self, x_i, x_j):
        # скалярное произведение
        return (x_i * x_j).sum(dim=-1, keepdim=True)

def negative_sampling_hetero(num_nodes, pos_edge_index, num_neg_samples=None):
    """
    Упрощённая функция для negative sampling в случае одного edge_type
    (где src_type == dst_type).
    :param num_nodes: общее число узлов данного типа
    :param pos_edge_index: тензор [2, num_pos], в котором хранятся "положительные" ребра
    :param num_neg_samples: сколько негативных хотим
    """
    if num_neg_samples is None:
        num_neg_samples = pos_edge_index.size(1)

    neg_src = []
    neg_dst = []

    existing = set((int(pos_edge_index[0,i]), int(pos_edge_index[1,i]))
                   for i in range(pos_edge_index.size(1)))

    # Наивный способ: генерируем случайные пары, пока не наберём нужное кол-во
    while len(neg_src) < num_neg_samples:
        u = random.randrange(num_nodes)
        v = random.randrange(num_nodes)
        # избегаем p=(u,u) если нужно
        if u != v:
            if (u,v) not in existing:
                existing.add((u,v))
                neg_src.append(u)
                neg_dst.append(v)
    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long)
    return neg_edge_index


@torch.no_grad()
def test(model, predictor, pos_edges, node_type, data):
    model.eval()
    predictor.eval()

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {}
    for et in data.edge_types:
        edge_index_dict[et] = data[et].edge_index

    embedding_dict = model(x_dict, edge_index_dict)

    src = pos_edges[0]
    dst = pos_edges[1]
    emb_src = embedding_dict[node_type][src]
    emb_dst = embedding_dict[node_type][dst]
    pos_score = predictor(emb_src, emb_dst).view(-1)

    neg_edge_index = negative_sampling_hetero(
        embedding_dict[node_type].size(0),
        pos_edges,
        num_neg_samples=pos_score.size(0)
    )
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    emb_src_neg = embedding_dict[node_type][neg_src]
    emb_dst_neg = embedding_dict[node_type][neg_dst]
    neg_score = predictor(emb_src_neg, emb_dst_neg).view(-1)

    labels = torch.cat([
        torch.ones_like(pos_score),
        torch.zeros_like(neg_score)
    ], dim=0)
    preds = torch.cat([pos_score, neg_score], dim=0)

    y_true = labels.cpu().numpy()
    y_scores = preds.detach().cpu().numpy()

    roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return roc, ap

def train_one_split(data, node_type, train_pos, val_pos, test_pos, device):
    # Создаем модель и линк-прогнозировщик
    model = GNNEncoder(hidden_channels=32)
    model = to_hetero(model, data.metadata(), aggr='sum').to(device)
    predictor = LinkPredictor().to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.001, weight_decay=1e-4
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Функция train на несколько эпох
    def train_step():
        model.train()
        predictor.train()
        optimizer.zero_grad()

        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = {}
        for et in data.edge_types:
            edge_index_dict[et] = data[et].edge_index

        embedding_dict = model(x_dict, edge_index_dict)

        pos_src = train_pos[0]
        pos_dst = train_pos[1]
        emb_src = embedding_dict[node_type][pos_src]
        emb_dst = embedding_dict[node_type][pos_dst]
        pos_pred = predictor(emb_src, emb_dst).view(-1)

        neg_edge_index = negative_sampling_hetero(
            embedding_dict[node_type].size(0),
            train_pos,
            num_neg_samples=pos_src.size(0)
        )
        neg_src = neg_edge_index[0]
        neg_dst = neg_edge_index[1]
        emb_src_neg = embedding_dict[node_type][neg_src]
        emb_dst_neg = embedding_dict[node_type][neg_dst]
        neg_pred = predictor(emb_src_neg, emb_dst_neg).view(-1)

        labels = torch.cat([torch.ones_like(pos_pred),
                            torch.zeros_like(neg_pred)])
        preds = torch.cat([pos_pred, neg_pred])
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Тренируем N эпох
    epochs = 50
    for epoch in range(1, epochs+1):
        loss_val = train_step()
        if epoch % 10 == 0:
            val_roc, val_ap = test(model, predictor, val_pos, node_type, data)
            print(f"Epoch={epoch}, Loss={loss_val:.4f}, Val ROC={val_roc:.4f}, AP={val_ap:.4f}")

    # По окончании - тест
    test_roc, test_ap = test(model, predictor, test_pos, node_type, data)
    print(f"\nTest ROC={test_roc:.4f}, Test AP={test_ap:.4f}")
    return test_roc, test_ap

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_type = "skzi"  # например "skzi", "organizatsiya", "persona"

    data = torch.load("./data/pyg/hetero_graph.pth").to(device)
    edge_type = (node_type, "duplicate", node_type)
    edge_index = data[edge_type].edge_index
    num_edges = edge_index.size(1)

    num_runs = 10  # Сколько раз делаем random split (каждый раз новый)
    test_rocs, test_aps = [], []

    for run in range(num_runs):
        print(f"\n=== Random split #{run+1} ===")
        # 1) Делаем новую перестановку
        perm = torch.randperm(num_edges)

        # 2) Берем 80/10/10
        train_size = int(0.8 * num_edges)
        val_size   = int(0.1 * num_edges)
        test_size  = num_edges - train_size - val_size

        train_indices = perm[:train_size]
        val_indices   = perm[train_size : train_size+val_size]
        test_indices  = perm[train_size+val_size:]

        train_pos = edge_index[:, train_indices]
        val_pos   = edge_index[:, val_indices]
        test_pos  = edge_index[:, test_indices]

        # 3) Обучаем модель и замеряем тест
        test_roc, test_ap = train_one_split(
            data, node_type,
            train_pos, val_pos, test_pos, device
        )
        test_rocs.append(test_roc)
        test_aps.append(test_ap)

    # Усредняем результат
    mean_roc = np.mean(test_rocs)
    std_roc  = np.std(test_rocs)
    mean_ap  = np.mean(test_aps)
    std_ap   = np.std(test_aps)

    print("\n=== Final result over random splits ===")
    print(f"ROC-AUC = {mean_roc:.4f} ± {std_roc:.4f}")
    print(f"AP      = {mean_ap:.4f} ± {std_ap:.4f}")

if __name__ == "__main__":
    main()
