import numpy as np
import json
from pathlib import Path

from sklearn import metrics
import torch
from tqdm import tqdm
from torch import default_generator, randperm
import torch.nn.functional as F
from torch_geometric.data import HeteroData, InMemoryDataset, Dataset, Batch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero
import torch_geometric.transforms as T

from build import build_connections_graph, build_hetero_graph_data, sub
from node_classification import DatasetMetainfo, LocalDataset, Experiment

all_edge_types = {
    ('АПС', 'Включает (не иерархия)', 'АПС'),  # 16
    ('Организация', 'Входит в группу', 'Организация'),  # 259
    ('Организация', 'Держатель акций', 'Организация'),  # 353
    ('Организация', 'Организация-разработчик', 'ПО'),  # 1301
    ('Организация', 'Патентообладатель', 'Патент'),  # 2164
    ('Организация', 'Покупатель (Заказчик)', 'Покупка-продажа СКЗИ'),  # 179
    ('Организация', 'Преобразование', 'Реорганизация'),  # 31
    ('Организация', 'Присоединение', 'Реорганизация'),  # 49
    ('Организация', 'Присуждение степени', 'Персона'),  # 25
    ('Организация', 'Продавец (Поставщик)', 'Покупка-продажа СКЗИ'),  # 160
    ('Организация', 'Разработка/модернизация', 'СКЗИ'),  # 648
    ('Организация', 'Сертификация', 'СКЗИ'),  # 56
    ('Организация', 'Слияние', 'Реорганизация'),  # 34
    ('Организация', 'Содержит', 'Подразделение'),  # 504
    ('Организация', 'Сотрудник', 'Персона'),  # 15
    ('Организация', 'Учредитель', 'Организация'),  # 11
    ('Организация', 'Является дочерней', 'Организация'),  # 1582
    ('Персона', 'Автор', 'Патент'),  # 6108
    ('Персона', 'Автор', 'Публикация (научная статья)'),  # 29592
    ('Персона', 'Владелец', 'Организация'),  # 11
    ('Персона', 'Держатель акций', 'Организация'),  # 92
    ('Персона', 'Исследователь', 'Изучение'),  # 10
    ('Персона', 'Разработка/модернизация', 'АПС'),  # 189
    ('Персона', 'Разработчик', 'ПО'),  # 140
    ('Персона', 'Руководитель', 'Организация'),  # 371
    ('Персона', 'Руководитель', 'Подразделение'),  # 187
    ('Персона', 'Сотрудник', 'Организация'),  # 12183
    ('Персона', 'Сотрудник', 'Подразделение'),  # 61
    ('Персона', 'Участник', 'Конференция'),  # 820
    ('Персона', 'Учеба', 'Организация'),  # 2668
    ('Персона', 'Учредитель', 'Организация'),  # 335
    ('Персона', 'Член', 'Совет директоров'),  # 75
    ('ПО', 'Операционная система', 'ПО'),  # 1635
    ('Подразделение', 'Включает (иерархия)', 'Подразделение'),  # 366
    ('Покупка-продажа СКЗИ', 'Объект купли-продажи', 'СКЗИ'),  # 118
    ('Публикация (научная статья)', 'Автор', 'Персона'),  # 27
    ('Публикация (научная статья)', 'Ссылается на', 'Публикация (научная статья)'),  # 25
    ('Реорганизация', 'Новая', 'Организация'),  # 40
    ('Реорганизация', 'Основная', 'Организация'),  # 56
    # ('Сертификация', 'Входит в семейство', 'Сертификация'),  # 19
    # ('Сертификация', 'Предыдущая версия', 'Сертификация'),  # 11
    ('СКЗИ', 'Включает (не иерархия)', 'СКЗИ'),  # 32
    ('СКЗИ', 'Входит в линейку/серию', 'СКЗИ'),  # 16
    ('СКЗИ', 'Использование', 'АПС'),  # 56
}

all_features = {
    "АПС": ['property_Тип АПС_1'],
    # "Атака": [],
    # "Грант": [],
    "Изучение": [],
    "Конференция": [],  # нет перечислимых полей в данных
    "Организация": ['property_Статус_1', 'property_Тип организации_1'],
    "Патент": [],
    "Персона": ['property_Ученое звание_1', 'property_Ученая степень_1', 'property_Специализация персоны_1', 'property_Пол_1'],
    "ПО": ['property_Тип ПО_1'],
    "Подразделение": [],
    "Покупка-продажа СКЗИ": [],
    "Публикация (научная статья)": [],
    "Реорганизация": ['property_Тип реорганизации_1'],
    # "Сертификация": ['property_Тип сертификации_1', 'property_Семейство сертификаций_1'],
    "СКЗИ": ['property_Подтип СКЗИ_1', 'property_Тип СКЗИ_1', 'property_Страна производства_1'],
    # "Собрание акционеров": [],
    "Совет директоров": [],
    # "Тендер": [],
    # "Уязвимость": [],
    # "Эксплоит": [],
}


def vary_all_features():

    # Model
    class GNN(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    # Datasets will contain all possible nodes and edges
    nodes = sorted(list(all_features.keys()))
    edges = sorted(list(set(e.replace('/', '_') for _, e, _ in all_edge_types)))

    # Vary all nodes and features to be predicted
    for node, feats in all_features.items():
        for feat in feats:
            print(f"\n\nPredicting '{feat}' for '{node}'.")

            # Build dataset
            metainfo = DatasetMetainfo()
            metainfo.undirected = True
            metainfo.nodes = nodes
            metainfo.edges = edges
            metainfo.edge_features = {}

            node_features = all_features.copy()
            # put feat on the first place
            rest_feats = set(feats)
            rest_feats.remove(feat)
            node_features[node] = [feat] + sorted(list(rest_feats))
            metainfo.node_features = node_features

            metainfo.node_labels = {node: feat}
            target_node = metainfo.target_node()

            # dataset = LocalDataset(metainfo, )
            dataset = LocalDataset(metainfo, post_transform=T.RemoveIsolatedNodes())
            data = dataset[0]
            # print(data)
            # return

            # print(target_node, feat)
            ys = [y for n, y in enumerate(data[sub(target_node)].y) if y != -1]
            # print("Labeled nodes", len(ys), np.bincount(ys))
            # print("Total nodes", len(data[sub(target_node)].y))
            if len(ys) < 10: continue

            # Model
            model = GNN(hidden_channels=16, out_channels=dataset.num_classes)
            model = to_hetero(model, data.metadata(), aggr='sum')

            # Experiment
            experiment = Experiment(model, data, metainfo, target_node)
            experiment.train_test_split(percent_train_class=0.8, percent_test_class=0.2)
            # experiment.set_metric(metric='F1', average='macro')
            experiment.set_metric(metric='BalancedAccuracy')  # = avg recall per class
            experiment.train_test_series(epochs=30, repeats=10)

            # return


if __name__ == '__main__':
    # build_connections_graph()

    vary_all_features()
