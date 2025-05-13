import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from configs import dify_config
from core.workflow.graph_engine.entities.run_condition import RunCondition
from core.workflow.nodes import NodeType
from core.workflow.nodes.answer.answer_stream_generate_router import AnswerStreamGeneratorRouter
from core.workflow.nodes.answer.entities import AnswerStreamGenerateRoute
from core.workflow.nodes.end.end_stream_generate_router import EndStreamGeneratorRouter
from core.workflow.nodes.end.entities import EndStreamParam


class GraphEdge(BaseModel):
    source_node_id: str = Field(..., description="source node id")
    target_node_id: str = Field(..., description="target node id")
    run_condition: Optional[RunCondition] = None
    """run condition"""


class GraphParallel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="random uuid parallel id")
    start_from_node_id: str = Field(..., description="start from node id")
    parent_parallel_id: Optional[str] = None
    """parent parallel id"""
    parent_parallel_start_node_id: Optional[str] = None
    """parent parallel start node id"""
    end_to_node_id: Optional[str] = None
    """end to node id"""


class Graph(BaseModel):
    root_node_id: str = Field(..., description="root node id of the graph")
    node_ids: list[str] = Field(default_factory=list, description="graph node ids")
    node_id_config_mapping: dict[str, dict] = Field(
        default_factory=dict, description="node configs mapping (node id: node config)"
    )
    edge_mapping: dict[str, list[GraphEdge]] = Field(
        default_factory=dict, description="graph edge mapping (source node id: edges)"
    )
    reverse_edge_mapping: dict[str, list[GraphEdge]] = Field(
        default_factory=dict, description="reverse graph edge mapping (target node id: edges)"
    )
    parallel_mapping: dict[str, GraphParallel] = Field(
        default_factory=dict, description="graph parallel mapping (parallel id: parallel)"
    )
    node_parallel_mapping: dict[str, str] = Field(
        default_factory=dict, description="graph node parallel mapping (node id: parallel id)"
    )
    answer_stream_generate_routes: AnswerStreamGenerateRoute = Field(..., description="answer stream generate routes")
    end_stream_param: EndStreamParam = Field(..., description="end stream param")

    @classmethod
    def init(cls, graph_config: Mapping[str, Any], root_node_id: Optional[str] = None) -> "Graph":
        """
        Init graph

        :param graph_config: graph config
        :param root_node_id: root node id
        :return: graph
        """
        # edge configs
        edge_configs = graph_config.get("edges")
        if edge_configs is None:
            edge_configs = []
        # node configs
        node_configs = graph_config.get("nodes")
        if not node_configs:
            raise ValueError("Graph must have at least one node")

        edge_configs = cast(list, edge_configs)
        node_configs = cast(list, node_configs)

        # reorganize edges mapping
        edge_mapping: dict[str, list[GraphEdge]] = {}
        reverse_edge_mapping: dict[str, list[GraphEdge]] = {}
        target_edge_ids = set()
        fail_branch_source_node_id = [
            node["id"] for node in node_configs if node["data"].get("error_strategy") == "fail-branch"
        ]
        for edge_config in edge_configs:
            source_node_id = edge_config.get("source")
            if not source_node_id:
                continue

            if source_node_id not in edge_mapping:
                edge_mapping[source_node_id] = []

            target_node_id = edge_config.get("target")
            if not target_node_id:
                continue

            if target_node_id not in reverse_edge_mapping:
                reverse_edge_mapping[target_node_id] = []

            target_edge_ids.add(target_node_id)

            # parse run condition
            run_condition = None
            if edge_config.get("sourceHandle"):
                if (
                    edge_config.get("source") in fail_branch_source_node_id
                    and edge_config.get("sourceHandle") != "fail-branch"
                ):
                    run_condition = RunCondition(type="branch_identify", branch_identify="success-branch")
                elif edge_config.get("sourceHandle") != "source":
                    run_condition = RunCondition(
                        type="branch_identify", branch_identify=edge_config.get("sourceHandle")
                    )

            graph_edge = GraphEdge(
                source_node_id=source_node_id, target_node_id=target_node_id, run_condition=run_condition
            )

            edge_mapping[source_node_id].append(graph_edge)
            reverse_edge_mapping[target_node_id].append(graph_edge)

        # fetch nodes that have no predecessor node
        root_node_configs = []
        all_node_id_config_mapping: dict[str, dict] = {}
        for node_config in node_configs:
            node_id = node_config.get("id")
            if not node_id:
                continue

            if node_id not in target_edge_ids:
                root_node_configs.append(node_config)

            all_node_id_config_mapping[node_id] = node_config

        root_node_ids = [node_config.get("id") for node_config in root_node_configs]

        # fetch root node
        if not root_node_id:
            # if no root node id, use the START type node as root node
            root_node_id = next(
                (
                    node_config.get("id")
                    for node_config in root_node_configs
                    if node_config.get("data", {}).get("type", "") == NodeType.START.value
                ),
                None,
            )

        if not root_node_id or root_node_id not in root_node_ids:
            raise ValueError(f"Root node id {root_node_id} not found in the graph")

        # 检查图是否存在环，避免循环执行
        cls._check_connected_to_previous_node(route=[root_node_id], edge_mapping=edge_mapping)

        # 根据edge_mapping，按照DFS添加节点ID
        node_ids = [root_node_id]
        cls._recursively_add_node_ids(node_ids=node_ids, edge_mapping=edge_mapping, node_id=root_node_id)

        # 获取真正有有效(可达)的node_id->node_config的映射，all_node_id_config_mapping是所有的，node_ids是可达的
        node_id_config_mapping = {node_id: all_node_id_config_mapping[node_id] for node_id in node_ids}

        # init parallel mapping
        parallel_mapping: dict[str, GraphParallel] = {}
        node_parallel_mapping: dict[str, str] = {}
        # 最开始初始化的时候是root_node_id, 此时还没有并行, parent_parallel也为None
        cls._recursively_add_parallels(
            edge_mapping=edge_mapping,
            reverse_edge_mapping=reverse_edge_mapping,
            start_node_id=root_node_id,
            parallel_mapping=parallel_mapping,
            node_parallel_mapping=node_parallel_mapping,
        )

        # Check if it exceeds N layers of parallel
        for parallel in parallel_mapping.values():
            if parallel.parent_parallel_id:
                cls._check_exceed_parallel_limit(
                    parallel_mapping=parallel_mapping,
                    level_limit=dify_config.WORKFLOW_PARALLEL_DEPTH_LIMIT,
                    parent_parallel_id=parallel.parent_parallel_id,
                )

        # init answer stream generate routes
        answer_stream_generate_routes = AnswerStreamGeneratorRouter.init(
            node_id_config_mapping=node_id_config_mapping, reverse_edge_mapping=reverse_edge_mapping
        )

        # init end stream param
        end_stream_param = EndStreamGeneratorRouter.init(
            node_id_config_mapping=node_id_config_mapping,
            reverse_edge_mapping=reverse_edge_mapping,
            node_parallel_mapping=node_parallel_mapping,
        )

        # init graph
        graph = cls(
            root_node_id=root_node_id,
            node_ids=node_ids,
            node_id_config_mapping=node_id_config_mapping,
            edge_mapping=edge_mapping,
            reverse_edge_mapping=reverse_edge_mapping,
            parallel_mapping=parallel_mapping,
            node_parallel_mapping=node_parallel_mapping,
            answer_stream_generate_routes=answer_stream_generate_routes,
            end_stream_param=end_stream_param,
        )

        return graph

    def add_extra_edge(
        self, source_node_id: str, target_node_id: str, run_condition: Optional[RunCondition] = None
    ) -> None:
        """
        Add extra edge to the graph

        :param source_node_id: source node id
        :param target_node_id: target node id
        :param run_condition: run condition
        """
        if source_node_id not in self.node_ids or target_node_id not in self.node_ids:
            return

        if source_node_id not in self.edge_mapping:
            self.edge_mapping[source_node_id] = []

        if target_node_id in [graph_edge.target_node_id for graph_edge in self.edge_mapping[source_node_id]]:
            return

        graph_edge = GraphEdge(
            source_node_id=source_node_id, target_node_id=target_node_id, run_condition=run_condition
        )

        self.edge_mapping[source_node_id].append(graph_edge)

    def get_leaf_node_ids(self) -> list[str]:
        """
        Get leaf node ids of the graph

        :return: leaf node ids
        """
        leaf_node_ids = []
        for node_id in self.node_ids:
            if node_id not in self.edge_mapping or (
                len(self.edge_mapping[node_id]) == 1
                and self.edge_mapping[node_id][0].target_node_id == self.root_node_id
            ):
                leaf_node_ids.append(node_id)

        return leaf_node_ids

    @classmethod
    def _recursively_add_node_ids(
        cls, node_ids: list[str], edge_mapping: dict[str, list[GraphEdge]], node_id: str
    ) -> None:
        """
        Recursively add node ids

        :param node_ids: node ids
        :param edge_mapping: edge mapping
        :param node_id: node id
        """
        for graph_edge in edge_mapping.get(node_id, []):
            if graph_edge.target_node_id in node_ids:
                continue

            node_ids.append(graph_edge.target_node_id)
            cls._recursively_add_node_ids(
                node_ids=node_ids, edge_mapping=edge_mapping, node_id=graph_edge.target_node_id
            )

    @classmethod
    def _check_connected_to_previous_node(cls, route: list[str], edge_mapping: dict[str, list[GraphEdge]]) -> None:
        """
        Check whether it is connected to the previous node
        """
        last_node_id = route[-1]

        for graph_edge in edge_mapping.get(last_node_id, []):
            if not graph_edge.target_node_id:
                continue

            if graph_edge.target_node_id in route:
                raise ValueError(
                    f"Node {graph_edge.source_node_id} is connected to the previous node, please check the graph."
                )

            new_route = route.copy()
            new_route.append(graph_edge.target_node_id)
            cls._check_connected_to_previous_node(
                route=new_route,
                edge_mapping=edge_mapping,
            )

    @classmethod
    def _recursively_add_parallels(
        cls,
        edge_mapping: dict[str, list[GraphEdge]],
        reverse_edge_mapping: dict[str, list[GraphEdge]],
        start_node_id: str,
        parallel_mapping: dict[str, GraphParallel],
        node_parallel_mapping: dict[str, str],
        parent_parallel: Optional[GraphParallel] = None,
    ) -> None:
        """
        Recursively add parallel ids
        收集属于并行分支的节点 ID，并将它们分配到适当的并行映射中。
        :param edge_mapping: edge mapping
        :param start_node_id: start from node id
        :param parallel_mapping: parallel mapping
        :param node_parallel_mapping: node parallel mapping
        :param parent_parallel: parent parallel
        """
        target_node_edges = edge_mapping.get(start_node_id, [])  # 获取所有出边
        parallel = None  # 尚未确定并行，设置为None
        if len(target_node_edges) > 1:
            # 出边数>1，可能存在并行
            # fetch all node ids in current parallels
            parallel_branch_node_ids = defaultdict(list)  # 存储并行分支节点id
            condition_edge_mappings = defaultdict(list)  # 存储条件边映射
            for graph_edge in target_node_edges:
                # 根据run_condition分组，没有 run_condition 的边被归类到“default”键下
                if graph_edge.run_condition is None:
                    parallel_branch_node_ids["default"].append(graph_edge.target_node_id)
                else:
                    # 带有 run_condition 的边按其 condition_hash 分组，使该方法能够处理条件分支。
                    condition_hash = graph_edge.run_condition.hash
                    condition_edge_mappings[condition_hash].append(graph_edge)

            for condition_hash, graph_edges in condition_edge_mappings.items():
                # 如果条件边映射中有多个边，可能存在并行分支，将条件hash作为键，存储对应的目标节点id
                if len(graph_edges) > 1:
                    for graph_edge in graph_edges:
                        parallel_branch_node_ids[condition_hash].append(graph_edge.target_node_id)

            condition_parallels = {}
            for condition_hash, condition_parallel_branch_node_ids in parallel_branch_node_ids.items():
                # 遍历所有并行分支(分支id -> 目标节点id列表)
                # any target node id in node_parallel_mapping
                parallel = None
                if condition_parallel_branch_node_ids:
                    parent_parallel_id = parent_parallel.id if parent_parallel else None

                    # uuid, 当前的start node id, 父parallel的id, 父parallel的start node id
                    parallel = GraphParallel(
                        start_from_node_id=start_node_id,
                        parent_parallel_id=parent_parallel.id if parent_parallel else None,
                        parent_parallel_start_node_id=parent_parallel.start_from_node_id if parent_parallel else None,
                    )
                    parallel_mapping[parallel.id] = parallel
                    condition_parallels[condition_hash] = parallel

                    # 检索并行每个分支中的所有节点, 递归遍历每个分支，收集节点 ID。
                    # in_branch_node_ids: {branch_node_id: node_ids(不包含合并节点)}
                    in_branch_node_ids = cls._fetch_all_node_ids_in_parallels(
                        edge_mapping=edge_mapping,
                        reverse_edge_mapping=reverse_edge_mapping,
                        parallel_branch_node_ids=condition_parallel_branch_node_ids,
                    )

                    # collect all branches node ids
                    # 如果节点是父并行的一部分或没有父并行，则将节点包含在当前并行中。
                    parallel_node_ids = []
                    for _, node_ids in in_branch_node_ids.items():
                        for node_id in node_ids:
                            # 检查当前节点是否已分配给父并行（如果有的话），或者是否可以安全地分配给当前并行。
                            in_parent_parallel = True
                            if parent_parallel_id:
                                in_parent_parallel = False
                                for parallel_node_id, parallel_id in node_parallel_mapping.items():
                                    if parallel_id == parent_parallel_id and parallel_node_id == node_id:
                                        # 当前节点是父级并行的一部分，退出循环。
                                        in_parent_parallel = True
                                        break

                            # 如果当前节点没有父并行节点/如果当前节点存在父并行节点，并且在父并行中
                            # 说明可以作为当前并行的节点
                            # 否则如果存在父并行，又不在父并行中，则不作为当前并行的节点
                            if in_parent_parallel:
                                parallel_node_ids.append(node_id)
                                node_parallel_mapping[node_id] = parallel.id

                    outside_parallel_target_node_ids = set()
                    for node_id in parallel_node_ids:
                        if node_id == parallel.start_from_node_id:
                            continue

                        node_edges = edge_mapping.get(node_id)
                        if not node_edges:
                            continue

                        if len(node_edges) > 1:
                            continue

                        target_node_id = node_edges[0].target_node_id
                        if target_node_id in parallel_node_ids:
                            continue

                        if parent_parallel_id:
                            parent_parallel = parallel_mapping.get(parent_parallel_id)
                            if not parent_parallel:
                                continue

                        if (
                            (
                                node_parallel_mapping.get(target_node_id)
                                and node_parallel_mapping.get(target_node_id) == parent_parallel_id
                            )
                            or (
                                parent_parallel
                                and parent_parallel.end_to_node_id
                                and target_node_id == parent_parallel.end_to_node_id
                            )
                            or (not node_parallel_mapping.get(target_node_id) and not parent_parallel)
                        ):
                            outside_parallel_target_node_ids.add(target_node_id)

                    if len(outside_parallel_target_node_ids) == 1:
                        if (
                            parent_parallel
                            and parent_parallel.end_to_node_id
                            and parallel.end_to_node_id == parent_parallel.end_to_node_id
                        ):
                            parallel.end_to_node_id = None
                        else:
                            parallel.end_to_node_id = outside_parallel_target_node_ids.pop()

            if condition_edge_mappings:
                for condition_hash, graph_edges in condition_edge_mappings.items():
                    for graph_edge in graph_edges:
                        current_parallel = cls._get_current_parallel(
                            parallel_mapping=parallel_mapping,
                            graph_edge=graph_edge,
                            parallel=condition_parallels.get(condition_hash),
                            parent_parallel=parent_parallel,
                        )

                        cls._recursively_add_parallels(
                            edge_mapping=edge_mapping,
                            reverse_edge_mapping=reverse_edge_mapping,
                            start_node_id=graph_edge.target_node_id,
                            parallel_mapping=parallel_mapping,
                            node_parallel_mapping=node_parallel_mapping,
                            parent_parallel=current_parallel,
                        )
            else:
                for graph_edge in target_node_edges:
                    current_parallel = cls._get_current_parallel(
                        parallel_mapping=parallel_mapping,
                        graph_edge=graph_edge,
                        parallel=parallel,
                        parent_parallel=parent_parallel,
                    )

                    cls._recursively_add_parallels(
                        edge_mapping=edge_mapping,
                        reverse_edge_mapping=reverse_edge_mapping,
                        start_node_id=graph_edge.target_node_id,
                        parallel_mapping=parallel_mapping,
                        node_parallel_mapping=node_parallel_mapping,
                        parent_parallel=current_parallel,
                    )
        else:
            for graph_edge in target_node_edges:
                current_parallel = cls._get_current_parallel(
                    parallel_mapping=parallel_mapping,
                    graph_edge=graph_edge,
                    parallel=parallel,
                    parent_parallel=parent_parallel,
                )

                cls._recursively_add_parallels(
                    edge_mapping=edge_mapping,
                    reverse_edge_mapping=reverse_edge_mapping,
                    start_node_id=graph_edge.target_node_id,
                    parallel_mapping=parallel_mapping,
                    node_parallel_mapping=node_parallel_mapping,
                    parent_parallel=current_parallel,
                )

    @classmethod
    def _get_current_parallel(
        cls,
        parallel_mapping: dict[str, GraphParallel],
        graph_edge: GraphEdge,
        parallel: Optional[GraphParallel] = None,
        parent_parallel: Optional[GraphParallel] = None,
    ) -> Optional[GraphParallel]:
        """
        Get current parallel
        """
        current_parallel = None
        if parallel:
            current_parallel = parallel
        elif parent_parallel:
            if not parent_parallel.end_to_node_id or (
                parent_parallel.end_to_node_id and graph_edge.target_node_id != parent_parallel.end_to_node_id
            ):
                current_parallel = parent_parallel
            else:
                # fetch parent parallel's parent parallel
                parent_parallel_parent_parallel_id = parent_parallel.parent_parallel_id
                if parent_parallel_parent_parallel_id:
                    parent_parallel_parent_parallel = parallel_mapping.get(parent_parallel_parent_parallel_id)
                    if parent_parallel_parent_parallel and (
                        not parent_parallel_parent_parallel.end_to_node_id
                        or (
                            parent_parallel_parent_parallel.end_to_node_id
                            and graph_edge.target_node_id != parent_parallel_parent_parallel.end_to_node_id
                        )
                    ):
                        current_parallel = parent_parallel_parent_parallel

        return current_parallel

    @classmethod
    def _check_exceed_parallel_limit(
        cls,
        parallel_mapping: dict[str, GraphParallel],
        level_limit: int,
        parent_parallel_id: str,
        current_level: int = 1,
    ) -> None:
        """
        Check if it exceeds N layers of parallel
        """
        parent_parallel = parallel_mapping.get(parent_parallel_id)
        if not parent_parallel:
            return

        current_level += 1
        if current_level > level_limit:
            raise ValueError(f"Exceeds {level_limit} layers of parallel")

        if parent_parallel.parent_parallel_id:
            cls._check_exceed_parallel_limit(
                parallel_mapping=parallel_mapping,
                level_limit=level_limit,
                parent_parallel_id=parent_parallel.parent_parallel_id,
                current_level=current_level,
            )

    @classmethod
    def _recursively_add_parallel_node_ids(
        cls,
        branch_node_ids: list[str],
        edge_mapping: dict[str, list[GraphEdge]],
        merge_node_id: str,
        start_node_id: str,
    ) -> None:
        """
        Recursively add node ids

        :param branch_node_ids: in branch node ids
        :param edge_mapping: edge mapping
        :param merge_node_id: merge node id
        :param start_node_id: start node id
        """
        for graph_edge in edge_mapping.get(start_node_id, []):
            if graph_edge.target_node_id != merge_node_id and graph_edge.target_node_id not in branch_node_ids:
                branch_node_ids.append(graph_edge.target_node_id)
                cls._recursively_add_parallel_node_ids(
                    branch_node_ids=branch_node_ids,
                    edge_mapping=edge_mapping,
                    merge_node_id=merge_node_id,
                    start_node_id=graph_edge.target_node_id,
                )

    @classmethod
    def _fetch_all_node_ids_in_parallels(
        cls,
        edge_mapping: dict[str, list[GraphEdge]],
        reverse_edge_mapping: dict[str, list[GraphEdge]],
        parallel_branch_node_ids: list[str],
    ) -> dict[str, list[str]]:
        """
        Fetch all node ids in parallels
        主要用于识别并行分支中的节点以及分支合并点。
        """

        """
        routes_node_ids : 存储从每个分支起始节点可访问的节点 ID 列表。
        leaf_node_ids : 存储每个分支的叶节点 ID（没有出边的节点）。
        merge_branch_node_ids : 存储分支可能重新加入的潜在合并节点。
        """

        routes_node_ids: dict[str, list[str]] = {}
        for parallel_branch_node_id in parallel_branch_node_ids:
            routes_node_ids[parallel_branch_node_id] = [parallel_branch_node_id]

            # fetch routes node ids
            # routes_node_ids[target_node_id] = [target_node_id, accessable node ids...]
            # 为每个并行分支的起始节点创建一个列表,使用_recursively_fetch_routes收集每个分支中的所有可达节点
            cls._recursively_fetch_routes(
                edge_mapping=edge_mapping,
                start_node_id=parallel_branch_node_id,
                routes_node_ids=routes_node_ids[parallel_branch_node_id],
            )

        # fetch leaf node ids from routes node ids
        leaf_node_ids: dict[str, list[str]] = {}
        merge_branch_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            # 遍历每个分支(分支的起始节点id, 分支上的所有节点id)
            for node_id in node_ids:
                # 识别叶子节点，叶子节点是不在edge_mapping里的，因为没有出边，在reverse_edge_mapping中有
                if node_id not in edge_mapping or len(edge_mapping[node_id]) == 0:
                    # 初始化空列表
                    if branch_node_id not in leaf_node_ids:
                        leaf_node_ids[branch_node_id] = []

                    leaf_node_ids[branch_node_id].append(node_id)

                # branch_node_id是分支1, 遍历除了分支1的所有分支(分支2), 如果分支2的路径上存在分支1的相同节点id, 那么就是潜在的合并节点
                for branch_node_id2, inner_route2 in routes_node_ids.items():
                    if (
                        branch_node_id != branch_node_id2
                        and node_id in inner_route2
                        and len(reverse_edge_mapping.get(node_id, [])) > 1
                        and cls._is_node_in_routes(
                            reverse_edge_mapping=reverse_edge_mapping,
                            start_node_id=node_id,
                            routes_node_ids=routes_node_ids,
                        )
                    ):
                        # 初始化空列表，可能有多个潜在的合并的节点(例如两个分支合并后的路径是一条单路径)
                        if node_id not in merge_branch_node_ids:
                            merge_branch_node_ids[node_id] = []

                        # 为什么将分支的节点的id添加到merge_branch_node_ids?  第一个分支一开始是没添加，但遍历到后面的时候，第一个分支实际上就是分支2，分支2变成了分支1，所以最开始的分支也被添加进来了！！！
                        if branch_node_id2 not in merge_branch_node_ids[node_id]:
                            merge_branch_node_ids[node_id].append(branch_node_id2)

        # sorted merge_branch_node_ids by branch_node_ids length desc
        # 按合并分支数量排序合并节点
        merge_branch_node_ids = dict(sorted(merge_branch_node_ids.items(), key=lambda x: len(x[1]), reverse=True))

        # 初始化重复或重叠的合并节点(后面要删除)
        duplicate_end_node_ids = {}

        # 如果某两个节点对应的合并分支节点id集合完全相同，则认为是重复/重叠节点，添加到duplicate_end_node_ids中
        for node_id, branch_node_ids in merge_branch_node_ids.items():
            for node_id2, branch_node_ids2 in merge_branch_node_ids.items():
                if node_id != node_id2 and set(branch_node_ids) == set(branch_node_ids2):
                    if (node_id, node_id2) not in duplicate_end_node_ids and (
                        node_id2,
                        node_id,
                    ) not in duplicate_end_node_ids:
                        duplicate_end_node_ids[(node_id, node_id2)] = branch_node_ids

        # 删除靠后的节点
        for (node_id, node_id2), branch_node_ids in duplicate_end_node_ids.items():
            # check which node is after
            if cls._is_node2_after_node1(node1_id=node_id, node2_id=node_id2, edge_mapping=edge_mapping):
                if node_id in merge_branch_node_ids and node_id2 in merge_branch_node_ids:
                    del merge_branch_node_ids[node_id2]
            elif cls._is_node2_after_node1(node1_id=node_id2, node2_id=node_id, edge_mapping=edge_mapping):
                if node_id in merge_branch_node_ids and node_id2 in merge_branch_node_ids:
                    del merge_branch_node_ids[node_id]

        # 将节点分配到各自的分支(分支起始节点id -> 合并节点id)
        branches_merge_node_ids: dict[str, str] = {}
        for node_id, branch_node_ids in merge_branch_node_ids.items():
            if len(branch_node_ids) <= 1:
                continue

            for branch_node_id in branch_node_ids:
                if branch_node_id in branches_merge_node_ids:
                    continue

                branches_merge_node_ids[branch_node_id] = node_id

        # 为每个分支编译节点
        # 上面的过程得到了最终的合并节点初始id，以及每个分支的合并节点id、每个分支的路径所有节点的id
        in_branch_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            # 对于每个分支，如果不存在合并点，则包含所有收集的节点， 就是一条单独的路径
            in_branch_node_ids[branch_node_id] = []
            if branch_node_id not in branches_merge_node_ids:
                # all node ids in current branch is in this thread
                in_branch_node_ids[branch_node_id].append(branch_node_id)
                in_branch_node_ids[branch_node_id].extend(node_ids)
            else:
                # 如果存在合并点，从分支起始节点遍历到合并点(不包括)。
                merge_node_id = branches_merge_node_ids[branch_node_id]
                if merge_node_id != branch_node_id:
                    in_branch_node_ids[branch_node_id].append(branch_node_id)

                # fetch all node ids from branch_node_id and merge_node_id
                cls._recursively_add_parallel_node_ids(
                    branch_node_ids=in_branch_node_ids[branch_node_id],
                    edge_mapping=edge_mapping,
                    merge_node_id=merge_node_id,
                    start_node_id=branch_node_id,
                )

        return in_branch_node_ids

    @classmethod
    def _recursively_fetch_routes(
        cls, edge_mapping: dict[str, list[GraphEdge]], start_node_id: str, routes_node_ids: list[str]
    ) -> None:
        """
        Recursively fetch route
        从给定的起始节点开始，收集该节点可到达的所有节点ID。
        """
        if start_node_id not in edge_mapping:
            return

        for graph_edge in edge_mapping[start_node_id]:
            # find next node ids
            if graph_edge.target_node_id not in routes_node_ids:
                routes_node_ids.append(graph_edge.target_node_id)

                cls._recursively_fetch_routes(
                    edge_mapping=edge_mapping, start_node_id=graph_edge.target_node_id, routes_node_ids=routes_node_ids
                )

    @classmethod
    def _is_node_in_routes(
        cls, reverse_edge_mapping: dict[str, list[GraphEdge]], start_node_id: str, routes_node_ids: dict[str, list[str]]
    ) -> bool:
        """
        Recursively check if the node is in the routes
        该方法检查节点是否是所有分支中的公共点，表明这是一个有效的合并点。
        """
        # 如果是StartNode，就不是有效的合并点，因为不在reverse_edge_mapping中，
        if start_node_id not in reverse_edge_mapping:
            return False

        all_routes_node_ids = set()
        parallel_start_node_ids: dict[str, list[str]] = {}
        for branch_node_id, node_ids in routes_node_ids.items():
            all_routes_node_ids.update(node_ids)

            if branch_node_id in reverse_edge_mapping:
                for graph_edge in reverse_edge_mapping[branch_node_id]:
                    if graph_edge.source_node_id not in parallel_start_node_ids:
                        parallel_start_node_ids[graph_edge.source_node_id] = []

                    parallel_start_node_ids[graph_edge.source_node_id].append(branch_node_id)

        for _, branch_node_ids in parallel_start_node_ids.items():
            if set(branch_node_ids) == set(routes_node_ids.keys()):
                return True

        return False

    @classmethod
    def _is_node2_after_node1(cls, node1_id: str, node2_id: str, edge_mapping: dict[str, list[GraphEdge]]) -> bool:
        """
        is node2 after node1
        """
        if node1_id not in edge_mapping:
            return False

        for graph_edge in edge_mapping[node1_id]:
            if graph_edge.target_node_id == node2_id:
                return True

            if cls._is_node2_after_node1(
                node1_id=graph_edge.target_node_id, node2_id=node2_id, edge_mapping=edge_mapping
            ):
                return True

        return False
