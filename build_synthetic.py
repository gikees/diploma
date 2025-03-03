import torch
import random
from torch_geometric.utils import coalesce
from build import build_hetero_graph_data


def duplicate_nodes_random_features(
    data,
    node_types_to_duplicate,
    duplicates_per_node=1,
    copy_feature_prob=0.5,
    drop_edge_prob=0.2,
    add_duplicate_edge=True
):
    """
    Дублирует узлы указанных node_types. Для каждой исходной вершины создаётся
    `duplicates_per_node` копий.

    - Для каждого признака (столбца в data[node_type].x) с вероятностью copy_feature_prob
      копируем значение из исходного узла, иначе обнуляем.
    - С вероятностью drop_edge_prob не копируем ребро на дубликат.
    - При add_duplicate_edge=True создаём и заполняем связь (node_type, 'duplicate', node_type)
      между (i -> i + offset), чтобы видно было, какие узлы являются дубликатами.
      Здесь offset каждый раз рассчитывается заново, т.к. x может расти после каждой итерации.

    :param data: HeteroData (после сборки build_hetero_graph_data).
    :param node_types_to_duplicate: Список типов узлов (str), которые хотим дублировать
                                    (например ['organizatsiya', 'persona']).
    :param duplicates_per_node: Сколько копий создавать для каждой вершины.
    :param copy_feature_prob: Вероятность скопировать исходное значение фичи (иначе обнуляем).
    :param drop_edge_prob: Вероятность НЕ копировать ребро для дубликата.
    :param add_duplicate_edge: Если True, добавляем ребро (node_type, 'duplicate', node_type)
                               для каждой пары (i, i+offset).
    """
    original_edge_types = list(data.edge_types)

    # Для каждого node_type, который дублируем, заранее создадим "пустой" edge_index
    # (если такого ребра ещё нет), чтобы потом добавлять в него.
    if add_duplicate_edge:
        for node_type in node_types_to_duplicate:
            if (node_type, "duplicate", node_type) not in data.edge_types:
                data[(node_type, "duplicate", node_type)].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Проходимся по каждому типу, где надо дублировать
    for node_type in node_types_to_duplicate:
        if node_type not in data.node_types:
            continue  # пропускаем, если такого типа нет

        # Исходная матрица признаков
        original_x = data[node_type].x.clone()  # (num_old_nodes, num_features)
        num_old_nodes, num_features = original_x.shape

        # Сохраняем сразу исходные рёбра, чтобы каждый раз копировать только их,
        # а не «накапливать» копии копий.
        edges_for_type = {}
        for et in original_edge_types:
            s_t, rel, d_t = et
            if s_t == node_type or d_t == node_type:
                edges_for_type[et] = data[et].edge_index.clone()

        # Делать несколько копий
        for _ in range(duplicates_per_node):
            # ===== Создаём новые фичи-дубликаты =====
            new_features = []
            for old_idx in range(num_old_nodes):
                old_vec = original_x[old_idx].clone()
                # Для каждого признака решаем, копировать его или обнулить
                for f_idx in range(num_features):
                    if random.random() >= copy_feature_prob:
                        old_vec[f_idx] = 0.0
                new_features.append(old_vec)

            new_features = torch.stack(new_features, dim=0)  # (num_old_nodes, num_features)

            # Перед добавлением узнаём, сколько узлов уже есть (x может расти после итераций).
            current_num_nodes = data[node_type].x.size(0)
            data[node_type].x = torch.cat([data[node_type].x, new_features], dim=0)

            # ===== Копируем рёбра (с учётом drop_edge_prob) =====
            for et in edges_for_type:
                s_t, rel, d_t = et
                old_edge_index = edges_for_type[et]  # исходная копия
                if old_edge_index.size(1) == 0:
                    continue

                srcs = old_edge_index[0]
                dsts = old_edge_index[1]

                new_srcs = []
                new_dsts = []

                # Если наш node_type — источник
                if s_t == node_type:
                    for i in range(srcs.size(0)):
                        if random.random() > drop_edge_prob:
                            # Смещаем индекс на current_num_nodes
                            new_src = srcs[i].item() + current_num_nodes
                            new_dst = dsts[i].item()
                            new_srcs.append(new_src)
                            new_dsts.append(new_dst)

                # Если наш node_type — целевой (dst)
                if d_t == node_type:
                    for i in range(dsts.size(0)):
                        if random.random() > drop_edge_prob:
                            new_dst = dsts[i].item() + current_num_nodes
                            new_src = srcs[i].item()
                            new_srcs.append(new_src)
                            new_dsts.append(new_dst)

                if len(new_srcs) > 0:
                    to_append = torch.tensor([new_srcs, new_dsts], dtype=torch.long)
                    combined = torch.cat([data[et].edge_index, to_append], dim=1)
                    data[et].edge_index = coalesce(combined)

            # ===== Добавляем ребро duplicate (i -> i+offset) =====
            if add_duplicate_edge:
                dup_edge_type = (node_type, "duplicate", node_type)
                old_ei = data[dup_edge_type].edge_index
                # Формируем (i, i + current_num_nodes) для i в [0..num_old_nodes-1],
                # где i - это именно «оригинальный» индекс
                srcs = torch.arange(num_old_nodes, dtype=torch.long)
                dsts = srcs + current_num_nodes
                to_append = torch.stack([srcs, dsts], dim=0)
                new_ei = torch.cat([old_ei, to_append], dim=1)
                data[dup_edge_type].edge_index = coalesce(new_ei)

    return data



if __name__ == "__main__":
    entity_folder = "./data/new_entities"   # Папка с CSV по сущностям
    relation_folder = "./data/new_relations"  # Папка с CSV по связям

    # Допустим, мы хотим добавлять в граф узлы типа "Организация" и связи типа "Сертификация".
    included_nodes = ["Организация", "СКЗИ", "Персона"]
    included_edges = ["Сертификация", 'Разработка/модернизация', 'Включает (не иерархия)', 'Входит в линейку/серию', 'Сотрудник', 'Учеба']
    # included_nodes = ["СКЗИ"]
    # included_edges = ['Включает (не иерархия)']

    # Which columns to use for making features
    node_features_names = {
        "СКЗИ": ['property_Подтип СКЗИ_1', 'property_Тип СКЗИ_1', 'property_Страна производства_1'],
        "Персона": ['property_Ученое звание_1', 'property_Ученая степень_1', 'property_Специализация персоны_1', 'property_Пол_1'],
        "Организация": ['property_Статус_1', 'property_Тип организации_1']
    }

    data = build_hetero_graph_data(
        entity_folder=entity_folder,
        relation_folder=relation_folder,
        included_node_types=included_nodes,
        included_edge_types=included_edges,
        node_features_names=node_features_names,
        do_expand=True,               # Распаковывать properties -> property_... колонки
        do_one_hot_properties=True    # Делать one-hot для всех колнок, начинающихся с property_
    )

    print(data)

    data = duplicate_nodes_random_features(
        data,
        node_types_to_duplicate=["skzi", "organizatsiya", "persona"],
        duplicates_per_node=5,
        copy_feature_prob=1.0,
        drop_edge_prob=0.0,
    )

    print(data)

    torch.save(data, "./data/pyg/hetero_graph.pth")

# эксперимент с известными дубликатами
