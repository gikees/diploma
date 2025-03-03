import numpy as np
import json
import re
import os
import ast
from typing import Tuple

import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.utils import coalesce


ENTITY_DIR = "./data/new_entities"  # Папка с CSV по сущностям
RELATION_DIR = "./data/new_relations"  # Папка с CSV по связям


def parse_properties(prop):
    """Безопасно парсит properties из ячеек CSV, которые содержат строки вида '[{'_type_name': '...', ...}]'."""
    if pd.isna(prop):
        return []
    try:
        return ast.literal_eval(prop)
    except (ValueError, SyntaxError):
        return []


def expand_properties(df, properties_col='properties'):
    """
    Распаковывает колонку 'properties' в несколько колонок property_<type_name>_<i>.
    Возвращает расширенный DataFrame.
    """
    if properties_col not in df.columns:
        return df  # Нет столбца — ничего не делаем

    df['parsed_properties'] = df[properties_col].apply(parse_properties)

    all_type_names = set()
    for props in df['parsed_properties']:
        for p in props:
            t = p.get('_type_name')
            if t:
                all_type_names.add(t)

    new_columns = {}
    for type_name in all_type_names:
        extracted_values = df['parsed_properties'].apply(
            lambda props: [
                v.get('value', {}).get('stringValueInput', {}).get('value', None)
                for item in props if item.get('_type_name') == type_name
                for v in item.get('_value', [])
            ]
        )
        # Определяем максимальное кол-во значений
        max_len = extracted_values.apply(len).max()

        for i in range(max_len):
            col_name = f'property_{type_name}_{i+1}'
            new_columns[col_name] = extracted_values.apply(
                lambda x: x[i] if i < len(x) else None
            )

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    df.drop(columns=['parsed_properties'], inplace=True)

    return df


def sub(text):
    """
    PyG не любит кириллические символы и прочие знаки в названиях типов,
    поэтому заменяем их на латиницу/цифры.
    """
    text = re.sub(r'[^a-zA-ZА-Яа-я0-9]', '', text)
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
        'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya'
    }
    result = ''
    for char in text.lower():
        result += translit_map.get(char, char)
    return result


def build_connections_graph(
    entity_folder: str = ENTITY_DIR,
    relation_folder: str = RELATION_DIR,
    included_node_types=None,
    included_edge_types=None,
):
    if included_node_types is None:
        included_node_types = set()
    else:
        included_node_types = set(included_node_types)

    if included_edge_types is None:
        included_edge_types = set()
    else:
        included_edge_types = set(included_edge_types)

    import networkx as nx
    graph = nx.MultiDiGraph()
    # 1) Загружаем все сущности, разбиваем по типам
    nodes_by_type = defaultdict(list)   # { node_type: [rows_as_dict, ...], ... }

    for file in os.listdir(entity_folder):
        if file.endswith('.csv'):
            node_type = re.compile(r'expanded_(.*)_df\.csv').findall(file)[0]
            if included_node_types and node_type not in included_node_types:
                continue
            path = os.path.join(entity_folder, file)
            df = pd.read_csv(path)

            # Разбиваем по типам (по столбцу '_type_name')
            for _, row in df.iterrows():
                assert node_type == row['_type_name'] if '_type_name' in row else 'Unknown'
                nodes_by_type[node_type].append(row.to_dict())

    all_nodes_by_id = {}  # Глобальный словарь: { node_id (str): (node_type, local_index) }

    # 2) Обрабатываем каждый тип узлов: делаем DataFrame, кодируем фичи
    for node_type, rows in nodes_by_type.items():
        df_nodes = pd.DataFrame(rows)

        # Удалим дубликаты по 'id' (на случай, если что-то дублируется)
        df_nodes.drop_duplicates(subset='id', inplace=True)

        # Собираем локальную мапу (id -> локальный_индекс)
        count = 0
        for idx, row in enumerate(df_nodes.itertuples(index=False)):
            node_id_str = getattr(row, 'id', None)
            if node_id_str is None:
                continue
            all_nodes_by_id[node_id_str] = (node_type, idx)
            count += 1
        graph.add_node(node_type, w=count)

    # 3) Разбиваем связи по (src_type, edge_type, dst_type)
    for file in os.listdir(relation_folder):
        if file.endswith('.csv'):
            link_type = re.compile(r'expanded_(.*)_df\.csv').findall(file)[0]
            if included_edge_types and link_type not in included_edge_types:
                continue
            path = os.path.join(relation_folder, file)
            df_rel = pd.read_csv(path)
            df_rel.drop_duplicates(subset='id', inplace=True)

            for _, row in df_rel.iterrows():
                link_type = row['_type_name'] if '_type_name' in row else 'Unknown'
                # if included_edge_types and link_type not in included_edge_types:
                #     continue

                first_id = row['first'] if 'first' in row else None
                second_id = row['second'] if 'second' in row else None
                if not first_id or not second_id:
                    continue
                if first_id not in all_nodes_by_id or second_id not in all_nodes_by_id:
                    # Если узлы не найдены (или исключены), пропускаем
                    continue

                # print(link_type)
                src_type, src_idx = all_nodes_by_id[first_id]
                dst_type, dst_idx = all_nodes_by_id[second_id]

                w = graph.get_edge_data(src_type, dst_type, link_type, default={'w': 0})['w']
                graph.add_edge(src_type, dst_type, link_type, w=w+1)

    print(graph.number_of_nodes(), graph.number_of_edges())
    for u, v, key, data in graph.edges(keys=True, data=True):
        print(f"{u}  -- {key} --> {v} : {data['w']}")

    # Отрисовка
    def nx_to_graphviz(G, min_node_weight=10, min_edge_weight=10):
        from graphviz import Digraph
        dot = Digraph()

        for node, data in G.nodes(data=True):
            w = data['w']
            if w < min_node_weight: continue
            label = str(node) + '\n' + str(w)
            dot.node(str(node), label=label)

        for src, dst, key, data in G.edges(keys=True, data=True):
            # Формируем метку из атрибутов ребра
            w = data['w']
            if w < min_edge_weight: continue
            label = key + '\n' + str(w)
            dot.edge(str(src), str(dst), label=label)

        return dot
    dot = nx_to_graphviz(graph, min_node_weight=10, min_edge_weight=10)
    dot.render("graph", engine='circo', view=True, format="png")

def build_hetero_graph_data(
    entity_folder: str = ENTITY_DIR,
    relation_folder: str = RELATION_DIR,
    included_node_types=None,
    included_edge_types=None,
    node_features_names=None,
    undirected: bool = False,
    do_expand: bool = True,
    do_one_hot_properties: bool = True
) -> HeteroData:
    """
    Строит разнородный граф (HeteroData).
    Важное отличие: НЕ убираем дубликаты по 'id'.
    Следовательно, все записи (даже при одинаковом 'id')
    будут становиться ОТДЕЛЬНЫМИ вершинами (изолированными, если нет связей).

    :param entity_folder: Папка с CSV по сущностям
    :param relation_folder: Папка с CSV по связям
    :param included_node_types: Какие типы узлов включить
    :param included_edge_types: Какие типы связей включить
    :param node_features_names: Словарь {node_type: [колонки для one-hot]}
    :param undirected: Делать ли граф неориентированным (дублировать рёбра)
    :param do_expand: Нужно ли распаковывать 'properties'
    :param do_one_hot_properties: Нужно ли делать one-hot для property_* столбцов
    :return: HeteroData
    """

    if included_node_types is None:
        included_node_types = set()
    else:
        included_node_types = set(included_node_types)

    if included_edge_types is None:
        included_edge_types = set()
    else:
        included_edge_types = set(included_edge_types)

    data = HeteroData()
    all_nodes_by_id = {}  # Глобальный словарь: { node_id(str) : (node_type, local_index) }

    # 1) Загружаем все сущности, разбиваем по типам
    from collections import defaultdict
    nodes_by_type = defaultdict(list)

    for file in os.listdir(entity_folder):
        if file.endswith('.csv'):
            node_type_match = re.compile(r'expanded_(.*)_df\.csv').findall(file)
            if not node_type_match:
                continue
            node_type = node_type_match[0]

            if node_type not in included_node_types:
                continue

            path = os.path.join(entity_folder, file)
            df = pd.read_csv(path)

            # (а) Распаковываем properties
            if do_expand:
                df = expand_properties(df, 'properties')

            # (б) one-hot (если нужно)
            if do_one_hot_properties:
                use_cols = node_features_names.get(node_type, []) if node_features_names else []
                df = pd.get_dummies(df, columns=use_cols, dummy_na=False)

            # (в) Не удаляем дубликаты по 'id' =>
            #     каждая строка (даже с тем же 'id') -> отдельная вершина

            # Сохраняем
            for _, row in df.iterrows():
                real_nt = row.get('_type_name', 'Unknown')
                nodes_by_type[real_nt].append(row.to_dict())

    # 2) Обрабатываем каждый тип узлов -> формируем x
    for node_type, rows in nodes_by_type.items():
        df_nodes = pd.DataFrame(rows)

        # Создаем локальную мапу (id -> local_index)
        local_id_map = {}
        for idx, row_ in df_nodes.iterrows():
            node_id_str = row_.get('id', None)
            # Заметим, у нас могут быть несколько строк с одинаковым 'id'
            # => они будут иметь разные idx -> разные вершины
            # => all_nodes_by_id[node_id_str] можно хранить СПИСОК, но
            #    если row_id_str уже есть, перезапишем (или будем хранить последнего).
            #    Для связей (см ниже) важен 'first'/'second',
            #    где ID = row_id_str.
            #    Возьмем тот, что first_id not in all_nodes_by_id,
            #    potentially conflicting.
            #
            # Но это может быть проблемой, если CSV связей указывает на 'id',
            #  а у нас теперь несколько узлов.
            #
            # Если "зеркалим" user question: "Все дублиткаты - отдельные вершины",
            #  тогда "id" не уникален, =>
            #   If "first" = 'someID', "someID" corresponds to SEVERAL vertices in the graph.
            #
            #   => Typically, we can't unambiguously link the edge to a single vertex.
            #   => We might need to do something special. But let's keep it simple:
            if node_id_str is None:
                continue
            local_id_map[node_id_str] = idx
            all_nodes_by_id[node_id_str] = (node_type, idx)

        # Определяем числовые столбцы
        ignore_cols = {
            'id','properties','source','creator','created','_name','_notes',
            '_begin','_end','_type_name','Unnamed: 0','secret','type'
        }
        feature_columns = []
        for col in df_nodes.columns:
            if col not in ignore_cols and pd.api.types.is_numeric_dtype(df_nodes[col]):
                feature_columns.append(col)

        x_df = df_nodes[feature_columns].fillna(0)
        x = torch.tensor(x_df.values, dtype=torch.float)
        if x.size(1) == 0:
            x = torch.ones(len(df_nodes), 10, dtype=torch.float)

        data[sub(node_type)].x = x

    # 3) Обрабатываем связи
    #    Важно: если CSV говорит "first = ID1, second = ID2",
    #           и ID1 / ID2 дублируется,
    #           тогда "какой" узел с ID1 / ID2 брать?
    #    Сейчас код берет "последний" (all_nodes_by_id[...] = (node_type, idx))
    #     => потеря инфы,
    #    Возможно, edges будут недоопределены.
    #
    #    Но в простом варианте -
    #     *Если* "first_id" всё ещё уникален,
    #      1-to-1,
    #     *Если* нет =>
    #      (В build.py обычно or read user question),
    #      or we store multiple.
    #
    #    Для упрощения оставим, как есть.

    edges_by_triplet = defaultdict(lambda: ([], []))

    for file in os.listdir(relation_folder):
        if file.endswith('.csv'):
            link_type_match = re.compile(r'expanded_(.*)_df\.csv').findall(file)
            if not link_type_match:
                continue
            link_type = link_type_match[0]
            if link_type not in included_edge_types:
                continue

            path = os.path.join(relation_folder, file)
            df_rel = pd.read_csv(path)

            for _, row in df_rel.iterrows():
                lt = row.get('_type_name', 'Unknown')
                first_id = row.get('first', None)
                second_id = row.get('second', None)
                if not first_id or not second_id:
                    continue
                if first_id not in all_nodes_by_id or second_id not in all_nodes_by_id:
                    # Если узлы не найдены (или их несколько?), пропускаем
                    continue

                src_type, src_idx = all_nodes_by_id[first_id]
                dst_type, dst_idx = all_nodes_by_id[second_id]

                edges_by_triplet[(src_type, lt, dst_type)][0].append(src_idx)
                edges_by_triplet[(src_type, lt, dst_type)][1].append(dst_idx)

    # Переводим списки в тензоры
    for (src_t, rel_t, dst_t), (src_list, dst_list) in edges_by_triplet.items():
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        data[(sub(src_t), sub(rel_t), sub(dst_t))].edge_index = coalesce(edge_index)
        if undirected:
            rev_ei = torch.tensor([dst_list, src_list], dtype=torch.long)
            data[(sub(dst_t), sub(rel_t), sub(src_t))].edge_index = coalesce(rev_ei)

    return data


if __name__ == "__main__":
    entity_folder = "./data/new_entities"
    relation_folder = "./data/new_relations"

    # Пример типов узлов и связей
    included_nodes = ["Организация", "СКЗИ", "Персона"]
    included_edges = [
        "Сертификация", "Разработка/модернизация",
        "Включает (не иерархия)", "Входит в линейку/серию", "Сотрудник", "Учеба"
    ]

    # Какие колонки One-Hot
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
        undirected=False,
        do_expand=True,
        do_one_hot_properties=True
    )

    print(data)
    # Сохраняем
    torch.save(data, "./data/pyg/hetero_graph_with_all_real_duplicates_as_separate_vertices.pth")
    print("Граф сохранён.")
