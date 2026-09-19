"""
Amazon Robotics Hackathon - Routing API

Team name:
Email address:

Levels 1-3: task assignment, congestion-aware routing, batching and dock clearance.
"""

import heapq
import math
from typing import Optional
from ar_hackathon.models.graph_state import GraphState


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """给指定机器人返回下一步的节点编号；None 表示等待。"""
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    distances, first_steps = _shortest_paths(state, unit.current_node)

    if unit.carrying:
        # 多货架：优先去当前有空位、距离最近的目的站。
        destinations = []
        for pod_id in unit.carrying:
            pod = state.get_pod(pod_id)
            if pod is not None and pod.destination_station in distances:
                station = state.get_node(pod.destination_station)
                full = (station is not None and station.capacity is not None
                        and state.node_occupancy(station.id) >= station.capacity)
                destinations.append((full, distances[pod.destination_station],
                                     pod.entry_time, pod.destination_station))
        target = min(destinations)[3] if destinations else None
    else:
        # 空载：统一匹配所有空载机器人，避免大家都追同一个货架。
        assignments = _assign_pods(state)
        pod = state.get_pod(assignments.get(unit.id))
        target = pod.current_node if pod is not None else None

    if target is None and not unit.carrying:
        # 无任务时预先返回取货区；仅使用地图中的 storage 类型，不读取未来任务。
        storage_options = [(distances.get(node.id, float("inf")), node.id)
                           for node in state.nodes if node.node_type == "storage"]
        if storage_options:
            target = min(storage_options)[1]
    next_node = first_steps.get(target)
    if next_node is None:
        # 送完货后，没有新任务也要离开有限容量的站点，给后来者让位。
        return _clear_dock(state, unit)

    # 最短路线可以包含等待。如果第一段路尚未空出，本轮就等待。
    edge = state.get_edge(unit.current_node, next_node)
    if edge is None or _edge_available_at(state, edge) > 0:
        return None

    node = state.get_node(next_node)
    if node is not None and node.capacity is not None:
        if state.node_occupancy(next_node) >= node.capacity:
            return None
    return next_node


def _assign_pods(state: GraphState):
    """按预计取货耗时，贪心匹配机器人和货架。每次调用重新核对状态。"""
    waiting_pods = []
    for pod in state.active_pods:
        if pod.carried_by is None and pod.current_node is not None:
            waiting_pods.append(pod)

    candidates = []
    for unit in state.drive_units:
        if unit.carrying or not unit.has_capacity:
            continue

        # 空载且正在路上的机器人也参与分工，从它即将到达的位置估算。
        start = unit.current_node
        delay = 0
        if unit.in_transit:
            start = unit.transit_destination
            delay = max(0, math.ceil(unit.transit_remaining_time))
        distances, _ = _shortest_paths(state, start, delay)

        for pod in waiting_pods:
            cost = distances.get(pod.current_node, float("inf"))
            if cost < float("inf"):
                candidates.append((cost, pod.entry_time, unit.id, pod.id))

    # 元组从左到右排序：耗时相同时，优先更早出现的任务，再按 ID。
    candidates.sort()
    assignments = {}
    assigned_pods = set()
    for cost, entry_time, unit_id, pod_id in candidates:
        if unit_id not in assignments and pod_id not in assigned_pods:
            assignments[unit_id] = pod_id
            # 引擎在到达时会按 entry_time、ID 自动装货，直到机器人装满。
            # 一次预留同一位置的整批货架，避免另一台车为同批货白跑。
            unit = state.get_drive_unit(unit_id)
            chosen_pod = state.get_pod(pod_id)
            batch = [pod for pod in waiting_pods
                     if pod.current_node == chosen_pod.current_node
                     and pod.id not in assigned_pods]
            batch.sort(key=lambda pod: (pod.entry_time, pod.id))
            for pod in batch[:unit.capacity - len(unit.carrying)]:
                assigned_pods.add(pod.id)
    return assignments


def _clear_dock(state, unit):
    """无任务时退出有限容量站点，优先选择普通通道或存储位置。"""
    current = state.get_node(unit.current_node)
    if current is None or current.capacity is None:
        return None
    options = []
    for neighbor in state.neighbors(unit.current_node):
        node = state.get_node(neighbor)
        edge = state.get_edge(unit.current_node, neighbor)
        if node is None or edge is None or _edge_available_at(state, edge) > 0:
            continue
        if node.capacity is not None and state.node_occupancy(neighbor) >= node.capacity:
            continue
        options.append((node.capacity is not None, node.node_type == "station",
                        edge.weight, neighbor))
    return min(options)[3] if options else None


def _edge_available_at(state: GraphState, edge):
    """根据当前路上的机器人，估算这条道路几步后有空位。"""
    if edge.capacity is None:
        return 0
    if edge.capacity <= 0:
        return float("inf")

    release_times = []
    for unit in state.drive_units:
        if unit.in_transit and edge.connects(unit.current_node, unit.transit_destination):
            release_times.append(max(0, math.ceil(unit.transit_remaining_time)))

    if len(release_times) < edge.capacity:
        return 0
    release_times.sort()
    # 容量为 1 时，等当前机器人离开；容量更大时，等到至少空一个位置。
    return release_times[len(release_times) - edge.capacity]


def _shortest_paths(state: GraphState, start: int, initial_delay=0):
    """Dijkstra：累计行驶和等待时间，记录通往每个节点的第一步。"""
    distances = {start: initial_delay}
    first_steps = {}
    queue = [(initial_delay, start)]

    # 先建立邻接表，避免在 Dijkstra 循环中反复扫描全部道路。
    adjacency = {}
    for edge in state.edges:
        ready_at = _edge_available_at(state, edge)
        travel_time = max(1, math.ceil(edge.weight))
        adjacency.setdefault(edge.from_node, []).append((edge.to_node, travel_time, ready_at))
        if edge.bidirectional:
            adjacency.setdefault(edge.to_node, []).append((edge.from_node, travel_time, ready_at))

    while queue:
        arrival_time, node = heapq.heappop(queue)
        if arrival_time > distances[node]:
            continue

        for neighbor, travel_time, ready_at in adjacency.get(node, []):
            # 到达路口后，如果道路还没空出，就等到 ready_at 再出发。
            next_arrival = max(arrival_time, ready_at) + travel_time
            if next_arrival < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_arrival
                if node == start:
                    first_steps[neighbor] = neighbor
                else:
                    first_steps[neighbor] = first_steps[node]
                heapq.heappush(queue, (next_arrival, neighbor))

    # 这里只预测当前已在路上的交通，未来的新交通在下次决策时重算。
    return distances, first_steps
