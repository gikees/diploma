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


def train(node_type):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    # получаем словарь x_dict (просто единички или что-то из data[node_type].x)
    # а также edge_index_dict (data[edge_type].edge_index) и передаём в model
    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {}
    for et in data.edge_types:
        edge_index_dict[et] = data[et].edge_index.to(device)

    # получаем эмбеддинги для всех типов
    embedding_dict = model(x_dict, edge_index_dict)

    # Берём только нужный тип узлов (skzi)
    # (если у нас (skzi, duplicate, skzi)),
    # соответственно, src/dst живут среди skzi-индексов
    # model(...) вернёт embedding_dict['skzi'] -- shape [num_org_nodes, hidden_dim]

    # Положительные примеры
    pos_src = train_pos[0]
    pos_dst = train_pos[1]
    emb_src = embedding_dict[node_type][pos_src]
    emb_dst = embedding_dict[node_type][pos_dst]
    pos_pred = predictor(emb_src, emb_dst).view(-1)  # [num_pos]

    # Негативные примеры
    neg_edge_index = negative_sampling_hetero(
        num_nodes=embedding_dict[node_type].size(0),
        pos_edge_index=train_pos,
        num_neg_samples=pos_src.size(0)  # чтобы кол-во neg = col-во pos (можно сделать *2, что потенциально улучшит качество)
    ).to(device)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    emb_src_neg = embedding_dict[node_type][neg_src]
    emb_dst_neg = embedding_dict[node_type][neg_dst]
    neg_pred = predictor(emb_src_neg, emb_dst_neg).view(-1)

    # Считаем лосс
    # Позитивным -> label=1, негативным -> label=0
    labels = torch.cat([torch.ones_like(pos_pred),
                        torch.zeros_like(neg_pred)])
    preds = torch.cat([pos_pred, neg_pred])
    loss = loss_fn(preds, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(pos_edges, node_type):
    model.eval()
    predictor.eval()

    x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
    edge_index_dict = {}
    for et in data.edge_types:
        edge_index_dict[et] = data[et].edge_index.to(device)

    # Получаем эмбеддинги
    embedding_dict = model(x_dict, edge_index_dict)

    # Положительные
    src = pos_edges[0]
    dst = pos_edges[1]
    emb_src = embedding_dict[node_type][src]
    emb_dst = embedding_dict[node_type][dst]
    pos_score = predictor(emb_src, emb_dst).view(-1)

    # Негативные
    neg_edge_index = negative_sampling_hetero(
        num_nodes=embedding_dict[node_type].size(0),
        pos_edge_index=pos_edges,
        num_neg_samples=pos_score.size(0)
    ).to(device)
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    neg_score = predictor(
        embedding_dict[node_type][neg_src],
        embedding_dict[node_type][neg_dst]
    ).view(-1)

    # Собираем метки и предсказания
    labels = torch.cat([
        torch.ones_like(pos_score),
        torch.zeros_like(neg_score)
    ], dim=0)
    preds = torch.cat([pos_score, neg_score], dim=0)

    # Переводим в numpy
    y_true = labels.cpu().numpy()
    y_scores = preds.detach().cpu().numpy()  # logits

    # ROC-AUC и Average Precision из sklearn
    roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return roc, ap


# Определяем тип узла, для которого тестируем дубликаты (например, node_type)
node_type = "persona"  # "skzi", "organizatsiya", "persona"

# # Формируем путь к файлу с реальными дубликатами
# duplicates_file = f"./data/real_duplicates_{node_type}.csv"

# # Загружаем реальные дубликаты, если файл существует
# if os.path.exists(duplicates_file):
#     real_duplicates_df = pd.read_csv(duplicates_file)
#     print(f"Загружены реальные дубликаты из {duplicates_file}")
# else:
#     real_duplicates_df = None
#     print(f"Файл {duplicates_file} не найден, пропускаем тестирование на реальных дубликатах.")

# Загрузка графа
data = torch.load("./data/pyg/hetero_graph.pth")
edge_type = (node_type, "duplicate", node_type)
edge_index = data[edge_type].edge_index  # shape [2, num_edges]
num_edges = edge_index.size(1)

# Train/Val/Test split
# Индексы рёбер
indices = torch.arange(num_edges)
indices = indices[torch.randperm(num_edges)]  # перемешиваем

train_size = int(0.8 * num_edges)
val_size = int(0.1 * num_edges)
test_size = num_edges - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size: train_size + val_size]
test_indices = indices[train_size + val_size:]

# Получаем реальные рёбра для train/val/test
train_pos = edge_index[:, train_indices]
val_pos = edge_index[:, val_indices]
test_pos = edge_index[:, test_indices]

# data[edge_type].edge_index = train_pos # в графе оставляем только train
# Оставляем 80% связей, а 20% удаляем (будут скрытыми дубликатами)
mask = torch.rand(train_pos.size(1)) < 0.8  # True для 80% связей
train_pos_visible = train_pos[:, mask]  # Оставляем только 80%
hidden_duplicates = train_pos[:, ~mask]  # 20% "скрытых" дубликатов

# Обновляем обучающий граф, удаляя часть связей
data[edge_type].edge_index = train_pos_visible


# Оборачиваем в to_hetero, чтобы уметь обрабатывать разные типы узлов/рёбер:
model = GNNEncoder(hidden_channels=32)
model = to_hetero(model, data.metadata(), aggr='sum')
predictor = LinkPredictor()  # dot product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # наш HeteroGNN
predictor = predictor.to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(predictor.parameters()),
    lr=0.001, weight_decay=1e-4
)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Собираем train_pos, val_pos, test_pos (как выше)
train_pos = train_pos.to(device)
val_pos = val_pos.to(device)
test_pos = test_pos.to(device)

# # Запускаем цикл обучения
# for epoch in range(1, 51):
#     loss = train(node_type)
#     if epoch % 1 == 0:
#         val_roc, val_ap = test(val_pos, node_type)
#         print(f"Epoch {epoch} | Loss={loss:.4f} | val ROC={val_roc:.4f}, AP={val_ap:.4f}")

# # Наконец, на тесте:
# test_roc, test_ap = test(test_pos, node_type)
# print("Test ROC:", test_roc, " Test AP:", test_ap)

num_runs = 10  # Количество запусков

roc_scores = []
ap_scores = []

for run in range(num_runs):
    print(f"\n=== Запуск {run+1}/{num_runs} ===\n")

    # Сброс случайного seed (чтобы каждый раз был новый train/test split)
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)

    # Пересоздаём модель и оптимизатор
    model = GNNEncoder(hidden_channels=32)
    model = to_hetero(model, data.metadata(), aggr='sum')
    predictor = LinkPredictor()

    model = model.to(device)
    predictor = predictor.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=0.001, weight_decay=1e-4
    )

    # Запускаем обучение
    for epoch in range(1, 51):  # Можно менять число эпох
        loss = train(node_type)
        if epoch % 10 == 0:
            val_roc, val_ap = test(val_pos, node_type)
            print(f"Epoch {epoch} | Loss={loss:.4f} | Val ROC={val_roc:.4f}, AP={val_ap:.4f}")

    # Оцениваем на тесте
    test_roc, test_ap = test(test_pos, node_type)
    print(f"\nTest ROC={test_roc:.4f}, Test AP={test_ap:.4f}")

    # Сохраняем результаты
    roc_scores.append(test_roc)
    ap_scores.append(test_ap)

# Вычисляем среднее и стандартное отклонение
mean_roc = np.mean(roc_scores)
std_roc = np.std(roc_scores)
mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)

print("\n=== Итоговое качество модели ===")
print(f"Предсказания для типа: {node_type}")
print(f"ROC-AUC: {mean_roc:.3f} ± {std_roc:.3f}")
print(f"AP: {mean_ap:.3f} ± {std_ap:.3f}")
