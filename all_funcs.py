import networkx as nx
from collections import Counter
import numpy as np
import geopandas as gpd
import pandas as pd
import random
import igraph as ig


def get_special_pinyin():
    special_pinyin = {
        '呼和浩特市': 'Hohhot',
        '乌鲁木齐市': 'Urumqi',
        '哈尔滨市': 'Harbin',
        '拉萨市': 'Lhasa',
        '鄂尔多斯市': 'Ordos',
        '伊犁哈萨克自治州': 'YiLi',
        '昌吉回族自治州': 'ChangJi',
        '巴彦淖尔市': 'Bayannur',
        '黔南布依族苗族自治州': 'QianNan',
        '玉溪市': 'YuXi',
        '阿克苏地区': 'Aksu',
        '怒江傈僳族自治州': 'NuJiang',
        '黔西南布依族苗族自治州': 'QianXiNan',
        '河池市': 'HeChi',
        '塔城地区': 'TaCheng',
        '乌兰察布市': 'Ulanqab',
        '甘南藏族自治州': 'GanNan',
        '文山壮族苗族自治州': 'WenShan',
        '临高县': 'LinGao',
        '林芝市': 'LinZhi',
        '博尔塔拉蒙古自治州': 'Bortala',
        '和田地区': 'HeTian',
        '克孜勒苏柯尔克孜自治州': 'Kizilsu',
        '凉山彝族自治州': 'LiangShan',
        '大理白族自治州': 'DaLi',
        '石家庄市': 'ShiJiaZhuang',
        '陵水黎族自治县': 'LingShui',
        '楚雄彝族自治州': 'ChuXiong',
        '喀什地区': 'KaShi',
        '黔东南苗族侗族自治州': 'QianDongNan',
        '锡林郭勒盟': 'Xilingol',
        '西双版纳傣族自治州': 'XiShuangBanNa',
        '澄迈县': 'ChengMai',
        '阿坝藏族羌族自治州': 'ABa',
        '黄石市': 'HuangShi',
        '巴音郭楞蒙古自治州': 'Bayingolin',
        '临夏回族自治州': 'LinXia',
        '保亭黎族苗族自治县': 'BaoTing',
        '黄南藏族自治州': 'HuangNan',
        '阿勒泰地区': 'Altay',
        '屯昌县': 'TunChang',
        '恩施土家族苗族自治州': 'Enshi',
        '石河子市': 'ShiHeZi',
        '克拉玛依市': 'Karamay',
        '延边朝鲜族自治州': 'YanBian',
        '本溪市': 'Benxi',
        '鸡西市': 'Jixi',
        '海西蒙古族藏族自治州': 'HaiXi&Zang',
        '西安市': "Xi'an",
        '马鞍山市': "Ma'anshan",
        '石家庄市': "Shijiazhuang",
    }
    return special_pinyin


def merge_sector():
    from pathlib import Path
    shp_files = [
        {"industry": "制造业",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_制造业.shp"},
        {"industry": "制造业汇总",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_制造业汇总.shp"},
        {"industry": "建筑业",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_建筑业.shp"},
        {"industry": "一般生产服务业",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_一般生产服务业.shp"},
        {"industry": "技术密集生产性服务业",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_技术密集生产性服务业.shp"},
        {"industry": "一般服务业",
            "path": "/Users/xuzzhan/Desktop/Research/产业网络韧性/数据/county_市区_OD_一般服务业.shp"}
    ]

    gdf_list = []
    target_crs = None

    for item in shp_files:
        industry = item["industry"]
        shp_path = Path(item["path"])

        gdf = gpd.read_file(shp_path)

        # 记录来源
        gdf["industry"] = industry
        gdf["src_file"] = shp_path.stem

        # 统一 CRS
        if target_crs is None:
            target_crs = gdf.crs
            if target_crs is None:
                raise ValueError(f"首个文件没有 CRS, 无法统一坐标系: {shp_path}")
        else:
            if gdf.crs is None:
                raise ValueError(f"文件没有 CRS: {shp_path}")
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)

        gdf_list.append(gdf)

    # 按列自动对齐并合并
    merged_gdf = gpd.GeoDataFrame(
        pd.concat(gdf_list, ignore_index=True, sort=False), crs=target_crs)
    temp = merged_gdf.groupby(['city_HQ', 'city_Br'], as_index=False)[
        '网络关'].sum()
    temp = temp[temp['city_HQ'] != temp['city_Br']]
    return temp


def build_graph(data, city_coords={}):
    G = nx.DiGraph()
    pair_weights = Counter()

    # 统一为 branch -> headquarters
    for hq, br, tie in zip(data['city_HQ'], data['city_Br'], data['网络关']):
        if hq != br:
            pair_weights[(br, hq)] += tie

    for (br, hq), weight in pair_weights.items():
        hq_coord = city_coords.get(hq, (None, None))
        br_coord = city_coords.get(br, (None, None))

        if None not in hq_coord:
            G.add_node(hq, pos=hq_coord)
        if None not in br_coord:
            G.add_node(br, pos=br_coord)

        distance = 1.0 / weight if weight > 0 else float('inf')
        G.add_edge(br, hq, weight=weight, distance=distance)
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


# def network_efficiency(G, weight='distance'):
#     total_efficiency = 0.0
#     for source in G.nodes():
#         lengths = nx.single_source_dijkstra_path_length(G, source, weight=weight)
#         for target, dist in lengths.items():
#             if source != target and dist > 0:
#                 total_efficiency += 1 / dist
#     return total_efficiency


def igraph_efficiency(g: ig.Graph, weight="distance", mode="out", normalized=False):
    if g.vcount() > 15000:
        interval = 2000
        batches = [range(k, min(k + interval, g.vcount()))
                   for k in range(0, g.vcount(), interval)]
        global_e = 0.0

        for sources in batches:
            dis_array = np.asarray(g.distances(
                source=sources, weights=weight, mode=mode), dtype=float)
            if weight == "angle":
                dis_array = np.where(dis_array < 0.15, np.inf, dis_array)

            sp = np.zeros_like(dis_array, dtype=float)
            np.divide(1.0, dis_array, out=sp, where=np.isfinite(
                dis_array) & (dis_array > 0))
            global_e += sp.sum()

        N = g.vcount()

    else:
        dis_array = np.asarray(g.distances(
            weights=weight, mode=mode), dtype=float)
        if weight == "angle":
            dis_array = np.where(dis_array < 0.15, np.inf, dis_array)

        sp = np.zeros_like(dis_array, dtype=float)
        np.divide(1.0, dis_array, out=sp, where=np.isfinite(
            dis_array) & (dis_array > 0))
        global_e = sp.sum()
        N = g.vcount()

    if normalized:
        global_e_norm = global_e / (N * (N - 1)) if N > 1 else 0.0
        return global_e_norm
    else:
        return global_e


def initialize_results():
    return {'efficiency': [], 'largest': [], 'second_largest': [],
            'avg_degree': [], 'degree_stddev': [], 'kcore_number': []}


def update_results(G, results, weight='distance'):
    g = ig.Graph.from_networkx(G)

    if weight in g.es.attributes():
        eff = igraph_efficiency(g, weight=weight, mode='out')
    else:
        eff = 0

    degrees = [d for _, d in G.degree()]
    avg_deg = np.mean(degrees) if degrees else 0.0
    std_deg = np.std(degrees) if degrees else 0.0

    core_nums = nx.core_number(nx.Graph(G)) if G.nodes() else {}
    kcore_avg = np.mean(list(core_nums.values())) if core_nums else 0.0

    # 使用有向图对象上的弱连通分量
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    largest = len(comps[0]) if comps else 0
    second = len(comps[1]) if len(comps) > 1 else 0

    results['efficiency'].append(eff)
    results['largest'].append(largest)
    results['second_largest'].append(second)
    results['avg_degree'].append(avg_deg)
    results['degree_stddev'].append(std_deg)
    results['kcore_number'].append(kcore_avg)


# ---------------------------------------------------------------------------- #
# attack

def select_max_node(scores):
    max_score = max(scores.values())
    candidates = [node for node, score in scores.items()
                  if np.isclose(score, max_score)]
    return sorted(candidates, key=str)[0]


def attack_network_once(G, attack_type='betweenness', random_seed=None):
    G_k = G.copy()
    rng = random.Random(random_seed)
    results = initialize_results()
    update_results(G_k, results)
    captured_remaining = [G.copy()]

    while G_k.number_of_nodes() > 0:
        if attack_type == 'random':
            to_remove = rng.choice(list(G_k.nodes()))
        elif attack_type == 'in_degree':
            scores = dict(G_k.in_degree(weight='weight'))
            to_remove = select_max_node(scores)
        elif attack_type == 'betweenness':
            scores = nx.betweenness_centrality(G_k, weight='distance')
            if not scores:
                break
            to_remove = select_max_node(scores)
        elif attack_type == 'closeness':
            scores = nx.closeness_centrality(G_k, distance='distance')
            if not scores:
                break
            to_remove = select_max_node(scores)
        else:
            raise ValueError("Invalid attack_type")

        G_k.remove_node(to_remove)
        if G_k.number_of_nodes() == 0:
            break

        captured_remaining.append(G_k.copy())

        update_results(G_k, results)

    return results, captured_remaining


def _random_attack_worker(G, run, random_seed):
    seed = None if random_seed is None else random_seed + run
    return attack_network_once(G, attack_type='random', random_seed=seed, return_paths=False)


def random_attack_average(G, n_runs=100, random_seed=42, n_jobs=-2, verbose=10):
    from joblib import Parallel, delayed

    all_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
        delayed(_random_attack_worker)(G, run, random_seed) for run in range(n_runs)
    )

    result_keys = all_results[0].keys()

    mean_results = {}
    lower_results = {}
    upper_results = {}

    for key in result_keys:
        values = np.asarray(
            [result[key] for result in all_results],
            dtype=float
        )

        mean_results[key] = np.mean(values, axis=0).tolist()
        lower_results[key] = np.percentile(values, 2.5, axis=0).tolist()
        upper_results[key] = np.percentile(values, 97.5, axis=0).tolist()

    return mean_results, lower_results, upper_results


# ---------------------------------------------------------------------------- #
# cascade
def cascade(G_k, initial_in, mu_0):
    nodes_to_remove = []
    for node in list(G_k.nodes()):
        h0 = initial_in[node]
        if h0 <= 0:
            continue

        ht = G_k.in_degree(node, weight='weight')
        loss_ratio = (h0 - ht) / h0
        if loss_ratio >= mu_0:
            nodes_to_remove.append(node)
    return nodes_to_remove


def attack_with_cascade(G, attack_type='betweenness', mu_0=0.5):
    results = initialize_results()
    update_results(G, results)

    initial_in = {node: G.in_degree(node, weight='weight')
                  for node in G.nodes()}
    G_k = G.copy()
    captured_remaining = [G.copy()]

    while G_k.number_of_nodes() > 0:
        if attack_type == 'in_degree':
            scores = dict(G_k.in_degree(weight='weight'))
            to_remove = select_max_node(scores)
        elif attack_type == 'betweenness':
            g = ig.Graph.from_networkx(G_k)
            if 'distance' in g.es.attributes():
                bet = g.betweenness(weights='distance')
                scores = {i: j for i, j in zip(g.vs['_nx_name'], bet)}
            else:
                scores = {}
            if not scores:
                break
            to_remove = select_max_node(scores)
        elif attack_type == 'closeness':
            g = ig.Graph.from_networkx(G_k)
            if 'distance' in g.es.attributes():
                clo = g.closeness(weights='distance')
                clo = np.nan_to_num(clo, nan=0.0)
                scores = {i: j for i, j in zip(g.vs['_nx_name'], clo)}
            else:
                scores = {}
            if not scores:
                break
            to_remove = select_max_node(scores)
        else:
            raise ValueError("Invalid attack_type")

        G_k.remove_node(to_remove)

        while True:
            nodes_to_remove = cascade(G_k, initial_in, mu_0)
            if not nodes_to_remove:
                break

            G_k.remove_nodes_from(nodes_to_remove)

        if G_k.number_of_nodes() == 0:
            break

        captured_remaining.append(G_k.copy())
        update_results(G_k, results)
    return results, captured_remaining


# ---------------------------------------------------------------------------- #
# reconfigure
def reconfigure(
    G,
    reconfig_type,
    RC_threshold,
    initial_in,
    initial_edges,
    initial_positions,
    reconfigured,
    event_log=None,
    context=None
):
    if reconfig_type not in {'nearest', 'capacity'}:
        raise ValueError("reconfig_type must be 'nearest' or 'capacity'")

    if context is None:
        context = {}

    if reconfig_type == 'capacity':
        capacity = {
            j: G.in_degree(j, weight='weight') + G.out_degree(j, weight='weight') for j in G.nodes()
        }
    else:
        capacity = {}

    for h in list(G.nodes()):
        initial_h_weight = initial_in.get(h, 0)

        if initial_h_weight <= 0:
            continue

        original_in_edges = initial_edges.get(h, {})

        lost_total = sum(
            stored_w
            for i, stored_w in original_in_edges.items()
            if not G.has_edge(i, h)
        )

        loss_ratio = lost_total / initial_h_weight

        if loss_ratio < RC_threshold:  # 损失边权大于等于阈值才重配
            continue

        for i, stored_w in original_in_edges.items():
            if G.has_edge(i, h):
                continue

            already_reconfigured = reconfigured.get((i, h), 0.0)
            lost_w = stored_w - already_reconfigured

            if lost_w <= 0:
                continue

            p_i = initial_positions[i]
            candidates = [j for j in G.nodes() if j != h]

            if not candidates:
                continue

            if reconfig_type == 'nearest':
                candidates = sorted(
                    candidates,
                    key=lambda j: np.hypot(
                        p_i[0] - G.nodes[j]['pos'][0],
                        p_i[1] - G.nodes[j]['pos'][1]
                    )
                )

                targets = candidates[:3]

                allocations = {j: lost_w / len(targets) for j in targets}

            else:
                attractiveness = {}

                for j in candidates:
                    p_j = G.nodes[j]['pos']

                    distance = np.hypot(p_i[0] - p_j[0], p_i[1] - p_j[1])

                    distance = max(distance, 1e-12)
                    attractiveness[j] = capacity[j] / distance

                targets = sorted(
                    candidates,
                    key=lambda j: attractiveness[j],
                    reverse=True
                )[:3]

                total_attractiveness = sum(attractiveness[j] for j in targets)

                if total_attractiveness <= 0:
                    continue

                allocations = {
                    j: attractiveness[j] / total_attractiveness * lost_w
                    for j in targets
                }

            for j, reassigned_w in allocations.items():
                if G.has_edge(j, h):
                    G[j][h]['weight'] += reassigned_w
                else:
                    G.add_edge(j, h, weight=reassigned_w)

                G[j][h]['distance'] = 1.0 / G[j][h]['weight']

                if event_log is not None:
                    event = {
                        'failed_branch': i,
                        'headquarters': h,
                        'receiver': j,
                        'reassigned_weight': reassigned_w,
                        'reconfig_type': reconfig_type
                    }

                    event.update(context)
                    event_log.append(event)

            reconfigured[(i, h)] = already_reconfigured + lost_w

    return G


def attack_with_cascade_reconfigure(G, mu_0=0.2, attack_type='betweenness', reconfig_type='nearest', RC_threshold=0.3):
    G_var = G.copy()

    # 初始入向连接权重，用于计算 μ_i^t 和 σ_h^t
    initial_in = {n: G.in_degree(n, weight='weight') for n in G.nodes()}

    # 保存所有原始 i→h 边
    initial_edges = {h: {i: G[i][h]['weight']
                         for i in G.predecessors(h)} for h in G.nodes()}

    # 保存初始位置，因为失效城市之后会从 G_var 中删除
    initial_positions = {n: G.nodes[n]['pos'] for n in G.nodes()}

    # 记录每条原始边已经完成重配置的权重
    reconfigured = {}

    captured_remaining = [G_var.copy()]

    results = initialize_results()
    update_results(G_var, results)

    while G_var.number_of_nodes() > 0:
        if attack_type == 'in_degree':
            scores = dict(G_var.in_degree(weight='weight'))
            to_remove = select_max_node(scores)
        elif attack_type == 'betweenness':
            g = ig.Graph.from_networkx(G_var)
            if 'distance' in g.es.attributes():
                bet = g.betweenness(weights='distance')
                scores = {i: j for i, j in zip(g.vs['_nx_name'], bet)}
            else:
                scores = {}
            if not scores:
                break
            to_remove = select_max_node(scores)
        elif attack_type == 'closeness':
            g = ig.Graph.from_networkx(G_var)
            if 'distance' in g.es.attributes():
                clo = g.closeness(weights='distance')
                clo = np.nan_to_num(clo, nan=0.0)
                scores = {i: j for i, j in zip(g.vs['_nx_name'], clo)}
            else:
                scores = {}
            if not scores:
                break
            to_remove = select_max_node(scores)
        else:
            raise ValueError("Invalid attack_type")

        G_var.remove_node(to_remove)

        # 级联过程迭代至稳定
        while True:
            nodes_to_remove = cascade(G_var, initial_in, mu_0)

            if not nodes_to_remove:
                break

            G_var.remove_nodes_from(nodes_to_remove)

        # 每个攻击步骤的级联稳定后立即重配置
        if G_var.number_of_nodes() > 0:
            G_var = reconfigure(
                G=G_var,
                reconfig_type=reconfig_type,
                RC_threshold=RC_threshold,
                initial_in=initial_in,
                initial_edges=initial_edges,
                initial_positions=initial_positions,
                reconfigured=reconfigured
            )

        captured_remaining.append(G_var.copy())
        update_results(G_var, results)

    return results, captured_remaining


# ---------------------------------------------------------------------------- #
# city importance
# class CityContribution:

def get_contribution_metrics(G):
    results = initialize_results()
    update_results(G, results)

    return {
        'LCC': results['largest'][-1],
        'NE': results['efficiency'][-1],
        'AD': results['avg_degree'][-1]
    }


def calculate_metric_change(before, after, initial, direction='loss'):
    if direction not in {'loss', 'gain'}:
        raise ValueError("direction must be 'loss' or 'gain'")

    changes = {}

    for metric in ('LCC', 'NE', 'AD'):
        denominator = initial[metric]

        if denominator <= 0:
            changes[metric] = 0.0
        else:
            if direction == 'loss':
                difference = before[metric] - after[metric]
            else:
                difference = after[metric] - before[metric]

            changes[metric] = max(difference / denominator, 0.0)

    return changes


def calculate_metric_losses(before, after, initial):
    """Backward-compatible wrapper for normalised performance losses."""
    return calculate_metric_change(
        before=before,
        after=after,
        initial=initial,
        direction='loss'
    )


def critic_weights(data, columns=('delta_LCC', 'delta_NE', 'delta_AD'), eps=1e-12):
    X = data.loc[:, columns].to_numpy(dtype=float)

    if len(X) < 2:
        return {'LCC': 1 / 3, 'NE': 1 / 3, 'AD': 1 / 3}

    x_min = np.nanmin(X, axis=0)
    x_max = np.nanmax(X, axis=0)
    x_range = x_max - x_min

    Z = np.divide(
        X - x_min,
        x_range,
        out=np.zeros_like(X),
        where=x_range > eps
    )

    std = np.nanstd(Z, axis=0, ddof=0)

    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    information = std * np.sum(1.0 - corr, axis=1)

    if information.sum() <= eps:
        weights = np.full(3, 1 / 3)
    else:
        weights = information / information.sum()

    return {
        'LCC': weights[0],
        'NE': weights[1],
        'AD': weights[2]
    }


def geometric_contribution(row, weights, epsilon=None):
    values = {
        'LCC': row['delta_LCC'],
        'NE': row['delta_NE'],
        'AD': row['delta_AD']
    }

    score = 1.0

    for metric, weight in weights.items():
        if weight <= 0:
            continue

        value = values[metric]

        if epsilon is None:
            if value <= 0:
                return 0.0
        else:
            value = max(value, epsilon)

        score *= value ** weight

    return float(score)


def linear_critic_scores(data, weights, eps=1e-12):
    columns = ['delta_LCC', 'delta_NE', 'delta_AD']
    X = data[columns].astype(float)

    x_min = X.min(axis=0)
    x_range = X.max(axis=0) - x_min

    Z = (X - x_min).divide(
        x_range.where(x_range > eps),
        axis=1
    ).fillna(0.0)

    return (
        weights['LCC'] * Z['delta_LCC']
        + weights['NE'] * Z['delta_NE']
        + weights['AD'] * Z['delta_AD']
    )


def calculate_city_contribution_raw(
    G,
    mu_0=0.7,
    reconfig_type='nearest',
    RC_threshold=0.3,
    importance_mode='incremental'
):
    if importance_mode not in {'incremental', 'cumulative_loss'}:
        raise ValueError(
            "importance_mode must be 'incremental' or 'cumulative_loss'"
        )

    G_var = G.copy()

    initial_metrics = get_contribution_metrics(G)

    initial_in = {n: G.in_degree(n, weight='weight') for n in G.nodes()}

    initial_edges = {
        h: {
            i: G[i][h]['weight']
            for i in G.predecessors(h)
        }
        for h in G.nodes()
    }

    initial_positions = {n: G.nodes[n]['pos'] for n in G.nodes()}

    reconfigured = {}
    records = []
    reconfiguration_events = []
    cascade_events = []
    captured_remaining = [list(G_var.nodes())]

    step = 0

    while G_var.number_of_nodes() > 0:
        step += 1

        metrics_before = get_contribution_metrics(G_var)
        nodes_before = G_var.number_of_nodes()

        g = ig.Graph.from_networkx(G_var)
        if 'distance' in g.es.attributes():
            bet = g.betweenness(weights='distance')
            scores = {i: j for i, j in zip(g.vs['_nx_name'], bet)}
        else:
            scores = {}
        if not scores:
            break

        attacked_node = max(scores, key=scores.get)
        attacked_bc = scores[attacked_node]

        # Attack-only state
        G_var.remove_node(attacked_node)
        metrics_attack = get_contribution_metrics(G_var)
        nodes_after_attack = G_var.number_of_nodes()

        # Cascading state
        cascade_removed_this_step = []

        while G_var.number_of_nodes() > 0:
            nodes_to_remove = cascade(G_var, initial_in, mu_0)

            if not nodes_to_remove:
                break

            for node in nodes_to_remove:
                cascade_events.append({
                    'strategy': 'dynamic',
                    'step': step,
                    'trigger_node': attacked_node,
                    'cascade_node': node
                })

            cascade_removed_this_step.extend(nodes_to_remove)
            G_var.remove_nodes_from(nodes_to_remove)

        metrics_cascade = get_contribution_metrics(G_var)
        nodes_after_cascade = G_var.number_of_nodes()

        # Reconfiguration state
        if G_var.number_of_nodes() > 0:
            G_var = reconfigure(
                G=G_var,
                reconfig_type=reconfig_type,
                RC_threshold=RC_threshold,
                initial_in=initial_in,
                initial_edges=initial_edges,
                initial_positions=initial_positions,
                reconfigured=reconfigured,
                event_log=reconfiguration_events,
                context={
                    'strategy': 'dynamic',
                    'trigger_node': attacked_node,
                    'step': step
                }
            )

        metrics_reconfig = get_contribution_metrics(G_var)
        nodes_after_reconfig = G_var.number_of_nodes()

        if importance_mode == 'incremental':
            stage_information = {
                'Attack': {
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_attack,
                    'nodes_before': nodes_before,
                    'remaining_nodes': nodes_after_attack,
                    'direction': 'loss'
                },
                'Cascade': {
                    'metrics_before': metrics_attack,
                    'metrics_after': metrics_cascade,
                    'nodes_before': nodes_after_attack,
                    'remaining_nodes': nodes_after_cascade,
                    'direction': 'loss'
                },
                'Reconfiguration': {
                    'metrics_before': metrics_cascade,
                    'metrics_after': metrics_reconfig,
                    'nodes_before': nodes_after_cascade,
                    'remaining_nodes': nodes_after_reconfig,
                    'direction': 'gain'
                }
            }
        else:
            stage_information = {
                'Attack': {
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_attack,
                    'nodes_before': nodes_before,
                    'remaining_nodes': nodes_after_attack,
                    'direction': 'loss'
                },
                'Cascade': {
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_cascade,
                    'nodes_before': nodes_before,
                    'remaining_nodes': nodes_after_cascade,
                    'direction': 'loss'
                },
                'Reconfiguration': {
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_reconfig,
                    'nodes_before': nodes_before,
                    'remaining_nodes': nodes_after_reconfig,
                    'direction': 'loss'
                }
            }

        for stage, stage_info in stage_information.items():
            stage_before = stage_info['metrics_before']
            metrics_after = stage_info['metrics_after']
            stage_nodes_before = stage_info['nodes_before']
            remaining_nodes = stage_info['remaining_nodes']
            change_direction = stage_info['direction']

            changes = calculate_metric_change(
                before=stage_before,
                after=metrics_after,
                initial=initial_metrics,
                direction=change_direction
            )

            records.append({
                'strategy': 'dynamic',
                'node': attacked_node,
                'stage': stage,
                'importance_mode': importance_mode,
                'change_direction': change_direction,
                'attack_order': step,
                'betweenness_before_attack': attacked_bc,
                'nodes_before': nodes_before,
                'stage_nodes_before': stage_nodes_before,
                'remaining_nodes': remaining_nodes,
                'cascade_removed_count': len(set(cascade_removed_this_step)),
                'LCC_before': stage_before['LCC'],
                'NE_before': stage_before['NE'],
                'AD_before': stage_before['AD'],
                'LCC_after': metrics_after['LCC'],
                'NE_after': metrics_after['NE'],
                'AD_after': metrics_after['AD'],
                'delta_LCC': changes['LCC'],
                'delta_NE': changes['NE'],
                'delta_AD': changes['AD']
            })

        captured_remaining.append(list(G_var.nodes()))

    return (
        pd.DataFrame(records),
        pd.DataFrame(reconfiguration_events),
        pd.DataFrame(cascade_events),
        captured_remaining
    )


def score_city_contributions(
    raw_data,
    critic_scope='stage',
    epsilon=None
):
    if critic_scope not in {
        'pooled',
        'stage',
    }:
        raise ValueError(
            "critic_scope must be 'pooled' or 'stage'"
        )

    data = raw_data.copy()
    if 'importance_mode' not in data.columns:
        data['importance_mode'] = 'unspecified'

    weight_records = []

    if critic_scope == 'pooled':
        scored_parts = []

        for importance_mode, group in data.groupby(
            'importance_mode',
            sort=False
        ):
            group = group.copy()
            weights = critic_weights(group)

            group['contribution_score'] = group.apply(
                geometric_contribution,
                axis=1,
                weights=weights,
                epsilon=epsilon
            )
        #     group['contribution_score'] = linear_critic_scores(
        #     group,
        #     weights=weights
        # )
            scored_parts.append(group)

            weight_records.append({
                'scope': 'pooled',
                'importance_mode': importance_mode,
                'strategy': 'all',
                'stage': 'all',
                'omega_LCC': weights['LCC'],
                'omega_NE': weights['NE'],
                'omega_AD': weights['AD']
            })

        data = pd.concat(
            scored_parts,
            ignore_index=True
        )

    elif critic_scope == 'stage':
        scored_parts = []

        for (importance_mode, stage), group in data.groupby(
            ['importance_mode', 'stage'],
            sort=False
        ):
            group = group.copy()
            weights = critic_weights(group)

            group['contribution_score'] = group.apply(
                geometric_contribution,
                axis=1,
                weights=weights,
                epsilon=epsilon
            )
        #     group['contribution_score'] = linear_critic_scores(
        #     group,
        #     weights=weights
        # )

            scored_parts.append(group)

            weight_records.append({
                'scope': 'stage',
                'importance_mode': importance_mode,
                'strategy': 'all',
                'stage': stage,
                'omega_LCC': weights['LCC'],
                'omega_NE': weights['NE'],
                'omega_AD': weights['AD']
            })

        data = pd.concat(
            scored_parts,
            ignore_index=True
        )

    contribution_groups = [
        'importance_mode',
        'strategy',
        'stage'
    ]

    group_total = data.groupby(
        contribution_groups
    )['contribution_score'].transform('sum')

    data['contribution_rate'] = np.divide(
        data['contribution_score'],
        group_total,
        out=np.zeros(len(data), dtype=float),
        where=group_total.to_numpy() > 0
    )

    data['rank'] = data.groupby(
        contribution_groups
    )['contribution_score'].rank(
        method='min',
        ascending=False
    ).astype(int)

    data = data.sort_values(
        ['importance_mode', 'strategy', 'stage', 'rank', 'node']
    ).reset_index(drop=True)

    return data, pd.DataFrame(weight_records)


def summarise_receiver_contribution(
    G,
    event_data
):
    output_columns = [
        'strategy',
        'receiver',
        'received_weight',
        'receiver_contribution_rate',
        'event_count',
        'trigger_count',
        'failed_branch_count',
        'headquarters_count',
        'initial_network_scale',
        'absorption_intensity',
        'intensity_contribution_rate',
        'receiver_rank',
        'intensity_rank'
    ]

    if event_data is None or event_data.empty:
        return pd.DataFrame(columns=output_columns)

    initial_scale = {
        node: (
            G.in_degree(node, weight='weight')
            + G.out_degree(node, weight='weight')
        )
        for node in G.nodes()
    }

    summary = (
        event_data
        .groupby(['strategy', 'receiver'])
        .agg(
            received_weight=(
                'reassigned_weight',
                'sum'
            ),
            event_count=(
                'reassigned_weight',
                'size'
            ),
            trigger_count=(
                'trigger_node',
                'nunique'
            ),
            failed_branch_count=(
                'failed_branch',
                'nunique'
            ),
            headquarters_count=(
                'headquarters',
                'nunique'
            )
        )
        .reset_index()
    )

    summary['initial_network_scale'] = summary[
        'receiver'
    ].map(initial_scale).fillna(0.0)

    summary['absorption_intensity'] = np.divide(
        summary['received_weight'],
        summary['initial_network_scale'],
        out=np.zeros(len(summary), dtype=float),
        where=summary['initial_network_scale'].to_numpy() > 0
    )

    received_total = summary.groupby(
        'strategy'
    )['received_weight'].transform('sum')

    summary['receiver_contribution_rate'] = np.divide(
        summary['received_weight'],
        received_total,
        out=np.zeros(len(summary), dtype=float),
        where=received_total.to_numpy() > 0
    )

    intensity_total = summary.groupby(
        'strategy'
    )['absorption_intensity'].transform('sum')

    summary['intensity_contribution_rate'] = np.divide(
        summary['absorption_intensity'],
        intensity_total,
        out=np.zeros(len(summary), dtype=float),
        where=intensity_total.to_numpy() > 0
    )

    summary['receiver_rank'] = summary.groupby(
        'strategy'
    )['received_weight'].rank(
        method='min',
        ascending=False
    ).astype(int)

    summary['intensity_rank'] = summary.groupby(
        'strategy'
    )['absorption_intensity'].rank(
        method='min',
        ascending=False
    ).astype(int)

    return summary.sort_values(
        ['strategy', 'receiver_rank', 'receiver']
    ).reset_index(drop=True)


def calculate_all_city_contributions(
        G,
        mu_0=0.7,
        reconfig_type='nearest',
        RC_threshold=0.3,
        importance_mode='cumulative_loss',
        critic_scope='stage',
        epsilon=None
):

    contribution_raw = calculate_city_contribution_raw(G=G,
                                                    mu_0=mu_0,
                                                    reconfig_type=reconfig_type,
                                                    RC_threshold=RC_threshold,
                                                    importance_mode=importance_mode
                                                    )
    dynamic_raw, dynamic_events, cascade_events, captured_remaining = contribution_raw

    contribution_data, critic_weight_data = score_city_contributions(raw_data=dynamic_raw,
                                                                     critic_scope=critic_scope,
                                                                     epsilon=epsilon
                                                                     )

    receiver_contribution = summarise_receiver_contribution(
        G=G, event_data=dynamic_events)

    return {
        'city_contribution': contribution_data,
        'critic_weights': critic_weight_data,
        'receiver_contribution': receiver_contribution,
        'receiver_events': dynamic_events,
        'cascade_events': cascade_events,
        'captured_remaining': captured_remaining
    }
