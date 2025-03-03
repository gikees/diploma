import yaml
import pandas as pd
import ast
import os


# import torch
# from torch_geometric.data import Data

# Загрузка данных из YAML файлов
def load_dataset(entities_path="entities.yaml", links_path="links.yaml"):
    with open(entities_path, 'r', encoding='utf-8') as f:
        entities = list(yaml.safe_load_all(f))

    with open(links_path, 'r', encoding='utf-8') as f:
        links = list(yaml.safe_load_all(f))

    return entities, links


def divide_by_types(entities: list, links: list):
    """
    Divides ENTITIES and LINKS by types and returns their types
    and dictionaries where keys are types of the nodes, edges.
    """
    entity_types, link_types = set(), set()
    nodes, edges = {}, {}

    for entity in entities:
        type = entity['_type_name']
        entity_types.add(type)
        if type not in nodes.keys():
            nodes[type] = []
        nodes[type].append(entity)

    for link in links:
        type = link['_type_name']
        link_types.add(type)
        if type not in edges.keys():
            edges[type] = []
        edges[type].append(link)

    return entity_types, link_types, nodes, edges


def create_df(obj_dict: dict, type_name: str) -> pd.DataFrame:
    return pd.DataFrame(data=obj_dict[type_name])


def save_dataframe(dataframe, filepath="../data/new_df.csv"):
    dataframe.to_csv(filepath)
    return


def expand_properties(df, properties_col='properties'):
    if properties_col not in df.columns:
        print(f"Столбец '{properties_col}' отсутствует. Пропуск раскрытия свойств.")
        return df  # Возвращаем исходный DataFrame без изменений

    # Функция для извлечения значений из properties
    def parse_properties(prop):
        if pd.isna(prop):
            return []
        try:
            return ast.literal_eval(prop)
        except (ValueError, SyntaxError):
            print(f"Ошибка при разборе свойства: {prop}")
            return []

    # Применяем функцию к столбцу properties
    df['parsed_properties'] = df[properties_col].apply(parse_properties)

    # Создаем список всех уникальных _type_name
    all_type_names = set()
    for props in df['parsed_properties']:
        for p in props:
            type_name = p.get('_type_name')
            if type_name:
                all_type_names.add(type_name)

    # Создаем словарь для новых столбцов
    new_columns = {}

    # Для каждого _type_name создаем отдельные столбцы
    for type_name in all_type_names:
        # Извлекаем все значения для данного type_name
        extracted_values = df['parsed_properties'].apply(
            lambda props: [item for item in props if item.get('_type_name') == type_name]
        )

        # Извлекаем значения из _value, безопасно обрабатывая отсутствующие ключи
        extracted_values = extracted_values.apply(
            lambda items: [
                v.get('value', {}).get('stringValueInput', {}).get('value', 'Null')
                for item in items
                for v in item.get('_value', [])
            ]
        )

        # Определяем максимальное количество значений для данного type_name
        max_len = extracted_values.apply(len).max()

        # Собираем новые столбцы в словарь
        for i in range(max_len):
            column_name = f'property_{type_name}_{i + 1}'
            new_columns[column_name] = extracted_values.apply(
                lambda x: x[i] if i < len(x) else None
            )

    # Объединяем новые столбцы с исходным DataFrame
    if new_columns:
        new_cols_df = pd.DataFrame(new_columns)
        df = pd.concat([df, new_cols_df], axis=1)

    # Удаляем временные столбцы
    df.drop(columns=['parsed_properties'], inplace=True)

    return df


entities, links = load_dataset("../data/entities_sample.yaml", "../data/links_sample.yaml")
# entities, links = load_dataset("../data/entities.yaml", "../data/links.yaml")
entity_types, link_types, nodes, edges = divide_by_types(entities, links)

print(entity_types)
print(len(entity_types))
print(link_types)
print(len(link_types))

node_dfs = {key: None for key in entity_types}
edge_dfs = {key: None for key in link_types}

for key in node_dfs.keys():
    node_dfs[key] = pd.DataFrame(data=nodes[key])
    save_dataframe(node_dfs[key], "./node_data/" + key.replace('/', '_') + "_df.csv")

for key in edge_dfs.keys():
    edge_dfs[key] = pd.DataFrame(data=edges[key])
    save_dataframe(edge_dfs[key], "./edge_data/" + key.replace('/', '_') + "_df.csv")

for df in node_dfs.items():
    print(df)

for df in edge_dfs.items():
    print(df)

# Папки с сущностями и связями
entity_folder = 'node_data'
relation_folder = 'edge_data'
entity_output_folder = '../data/new_entities'
relation_output_folder = '../data/new_relations'

# Создание выходной папки, если ее нет
os.makedirs(entity_output_folder, exist_ok=True)
os.makedirs(relation_output_folder, exist_ok=True)

# Обработка файлов сущностей
for filename in os.listdir(entity_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(entity_folder, filename)
        df = pd.read_csv(filepath, sep=',')  # Убедитесь в правильности разделителя
        print(f"Обрабатывается файл сущности: {filename}")
        df_expanded = expand_properties(df, properties_col='properties')
        output_path = os.path.join(entity_output_folder, f'expanded_{filename}')
        df_expanded.to_csv(output_path, index=False)
        print(f"Сохранен расширенный файл: {output_path}")

# Обработка файлов связей (если требуется аналогично)
for filename in os.listdir(relation_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(relation_folder, filename)
        df = pd.read_csv(filepath, sep=',')  # Убедитесь в правильности разделителя
        print(f"Обрабатывается файл связи: {filename}")
        # Проверяем, содержит ли файл столбец 'properties'
        if 'properties' in df.columns:
            df_expanded = expand_properties(df, properties_col='properties')
            print(f"Столбец 'properties' найден и обработан.")
        else:
            df_expanded = df  # Если столбца нет, оставляем DataFrame без изменений
            print(f"Столбец 'properties' отсутствует. Пропуск раскрытия свойств.")
        output_path = os.path.join(relation_output_folder, f'expanded_{filename}')
        df_expanded.to_csv(output_path, index=False)
        print(f"Сохранен расширенный файл: {output_path}")

# print(entities)
# print(links)

# Создание словаря для преобразования id узлов в числовые индексы
# node_id_map = {entity['id']: idx for idx, entity in enumerate(entities[:5])}
# num_nodes = len(node_id_map)

# print(node_id_map)

# # Создание списка фичей для узлов (например, тип узла или другие свойства)
# node_features = []
# for entity in entities:
#     node_type = entity.get('_type_name', 'Unknown')
#     # Пример: Преобразуем строковое имя типа в числовое значение
#     # (Можно расширить для более сложных фичей)
#     node_feature = [hash(node_type) % 1000]
#     node_features.append(node_feature)

# # Преобразуем node_features в тензор
# x = torch.tensor(node_features, dtype=torch.float)

# # Создание edge_index для представления связей
# edges = []
# for link in links:
#     first_id = link['first']
#     second_id = link['second']
#     # Преобразуем идентификаторы узлов в индексы
#     first_idx = node_id_map[first_id]
#     second_idx = node_id_map[second_id]
#     edges.append([first_idx, second_idx])

# # Преобразуем edges в edge_index тензор
# edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# # Создаем объект Data для PyTorch Geometric
# data = Data(x=x, edge_index=edge_index)

# print(data)
