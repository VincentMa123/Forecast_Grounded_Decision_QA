from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeclaw.backend.pipeline.scenario_preflight import (
    DATA_FILE_RE,
    DATA_FILE_SUBDIRECTORIES,
    DEFAULT_PIPELINE_DATA_ROOT,
)
from .pipeline_scope import filter_rows_by_named_pipeline


TARGET_STATION_RE = re.compile(
    r"(?:到|通向|至)\s*([一-鿿]{2,16}站)"
)
INBOUND_COUNT_RE = re.compile(
    r"(?:全部|共)?\s*([一-十\d]+)\s*条(?:直接)?入站(?:管道|管线)"
)
EXCLUSIVE_CAUSE_RE = re.compile(
    r"(?:原因)?只能(?:是|在)?上游|只可能(?:是|在)?上游"
)
TEMPORAL_LOAD_RE = re.compile(
    r"夜里(?:一般|通常)?是低负荷|夜里.{0,12}负荷(?:曲线)?(?:在)?下降"
)
DENIES_MULTI_SOURCE_RE = re.compile(
    r"不是.{0,8}多源可达|而非.{0,8}多源可达"
)
QUALIFIED_SHARED_GATEWAY_RE = re.compile(
    r"多源可达|共享(?:网关|上游|入口)|共同(?:网关|上游|入口)"
)
TOPOLOGY_REQUEST_RE = re.compile(
    r"可达|最短路径|路径|反向追|共享网关|单源依赖|多源"
    r"|\b(?:reachable|reachability|shortest path|shared gateway|multi.source|single.source)\b",
    re.IGNORECASE,
)
CHINESE_NUMBERS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

CANONICAL_STATION_ALIASES = {
    "通州南分输站": "通州南站",
    "湘潭分输站": "湘潭站",
    "乌鲁木齐压气站": "乌鲁木齐站",
}


def build_topology_evidence_result(
    text: str,
    *,
    pipeline_data_root: Path | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build topology evidence and retain a specific, machine-readable failure."""
    root = Path(pipeline_data_root or DEFAULT_PIPELINE_DATA_ROOT).resolve()
    referenced: Dict[str, str] = {}
    for match in DATA_FILE_RE.finditer(text):
        referenced.setdefault(match.group("kind").casefold(), match.group("filename"))
    target_match = TARGET_STATION_RE.search(text)
    if not target_match:
        return {}, _topology_error(
            "missing_target_station",
            "No target station could be parsed from the topology request.",
        )
    missing_references = sorted({"node", "pipeline"} - set(referenced))
    if missing_references:
        return {}, _topology_error(
            "missing_topology_file_reference",
            "A topology request must name one daily node CSV and one daily pipeline CSV.",
            missing_kinds=missing_references,
        )

    node_path = root / DATA_FILE_SUBDIRECTORIES["node"] / referenced["node"]
    pipeline_path = root / DATA_FILE_SUBDIRECTORIES["pipeline"] / referenced["pipeline"]
    missing_files = [
        path.name for path in (node_path, pipeline_path) if not path.is_file()
    ]
    if missing_files:
        return {}, _topology_error(
            "topology_file_not_found",
            "One or more requested topology fixtures do not exist.",
            missing_files=missing_files,
        )

    station_types: Dict[str, set[str]] = defaultdict(set)
    source_instances: List[Dict[str, str]] = []
    with node_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            station = str(row.get("站名") or row.get("station") or "").strip()
            kind = str(row.get("类型") or row.get("type") or "").strip()
            pipeline = str(
                row.get("管道划分") or row.get("pipeline") or ""
            ).strip()
            if station:
                station_types[station].add(kind)
                if "气源" in kind or kind.casefold() == "source":
                    source_instances.append({"station": station, "pipeline": pipeline})

    pipeline_rows: List[Dict[str, str]] = []
    with pipeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = str(
                row.get("起点站名") or row.get("source") or ""
            ).strip()
            target = str(
                row.get("终点站名") or row.get("target") or ""
            ).strip()
            if not source or not target:
                continue
            pipeline_rows.append(
                {
                    "source": source,
                    "target": target,
                    "pipeline": str(
                        row.get("管道划分") or row.get("pipeline") or ""
                    ).strip(),
                    "flow": str(
                        row.get("管道流量") or row.get("flow") or ""
                    ).strip(),
                }
            )

    pipeline_rows, pipeline_scope = filter_rows_by_named_pipeline(pipeline_rows, text)
    if not pipeline_rows:
        return {}, _topology_error(
            "empty_topology_scope",
            "No pipeline rows remain after applying the requested pipeline scope.",
            pipeline_scope=pipeline_scope,
        )
    reverse: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in pipeline_rows:
        reverse[row["target"]].append(row)

    requested_target = target_match.group(1)
    target = _canonical_target_station(requested_target, root)
    topology_stations = {
        station for row in pipeline_rows for station in (row["source"], row["target"])
    }
    topology_stations.update(station_types)
    if target not in topology_stations:
        return {}, _topology_error(
            "target_station_not_in_topology",
            "The requested target does not map to a station in the selected topology files.",
            requested_target_station=requested_target,
            canonical_target_station=target,
        )
    distance = {target: 0}
    path_edges: Dict[str, Dict[str, str]] = {}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for edge in reverse.get(current, []):
            upstream = edge["source"]
            if upstream not in distance:
                distance[upstream] = distance[current] + 1
                path_edges[upstream] = edge
                queue.append(upstream)

    reachable_sources = sorted(
        (
            {"station": station, "hops": hops}
            for station, hops in distance.items()
            if "气源" in station_types.get(station, set())
            or "source"
            in {value.casefold() for value in station_types.get(station, set())}
        ),
        key=lambda item: (item["hops"], item["station"]),
    )
    pipeline_distances: Dict[str, Dict[str, int]] = {}
    for pipeline in {row["pipeline"] for row in pipeline_rows if row["pipeline"]}:
        pipeline_reverse: Dict[str, List[str]] = defaultdict(list)
        for row in pipeline_rows:
            if row["pipeline"] == pipeline:
                pipeline_reverse[row["target"]].append(row["source"])
        distances = {target: 0}
        pipeline_queue = deque([target])
        while pipeline_queue:
            current = pipeline_queue.popleft()
            for upstream_station in pipeline_reverse.get(current, []):
                if upstream_station not in distances:
                    distances[upstream_station] = distances[current] + 1
                    pipeline_queue.append(upstream_station)
        pipeline_distances[pipeline] = distances
    reachable_source_instances = sorted(
        (
            {
                "station": item["station"],
                "pipeline": item["pipeline"],
                "hops": pipeline_distances[item["pipeline"]][item["station"]],
            }
            for item in source_instances
            if item["pipeline"] in pipeline_distances
            and item["station"] in pipeline_distances[item["pipeline"]]
        ),
        key=lambda item: (item["hops"], item["station"], item["pipeline"]),
    )
    inbound = reverse.get(target, [])
    upstream = sorted({edge["source"] for edge in inbound})
    nearest_source = reachable_sources[0]["station"] if reachable_sources else None
    example_shortest_path = []
    cursor = nearest_source
    while cursor and cursor != target and cursor in path_edges:
        edge = path_edges[cursor]
        example_shortest_path.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "pipeline": edge["pipeline"],
            }
        )
        cursor = edge["target"]
    summary = {
        "source_files": [referenced["node"], referenced["pipeline"]],
        "target_station": target,
        "reachable_sources": reachable_sources,
        "reachable_source_count": len(reachable_sources),
        "nearest_source": nearest_source,
        "example_shortest_path": example_shortest_path,
        "reachable_source_instances": reachable_source_instances,
        "reachable_source_instance_count": len(reachable_source_instances),
        "direct_inbound_segment_count": len(inbound),
        "direct_upstream_stations": upstream,
        "direct_inbound_segments": inbound,
        "multi_source_reachable": len(reachable_sources) > 1,
        "shared_gateway_dependency": bool(inbound) and len(upstream) == 1,
    }
    if requested_target != target:
        summary["requested_target_station"] = requested_target
        summary["target_normalization"] = {
            "requested": requested_target,
            "canonical": target,
        }
    if pipeline_scope:
        summary["pipeline_scope"] = pipeline_scope
    return summary, {}


def _canonical_target_station(target: str, root: Path) -> str:
    mappings = dict(CANONICAL_STATION_ALIASES)
    mapping_path = root / "consumer_station.csv"
    if mapping_path.is_file():
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                supply_point = str(row.get("供气点") or "").strip()
                station = str(row.get("匹配站名") or "").strip()
                if supply_point and station:
                    mappings[supply_point] = station
    return mappings.get(target, target)


def _topology_error(code: str, message: str, **details: Any) -> Dict[str, Any]:
    return {
        "error_code": code,
        "message": message,
        **{key: value for key, value in details.items() if value not in (None, [], {})},
    }


def topology_tool_required(text: str) -> bool:
    kinds = {match.group("kind").casefold() for match in DATA_FILE_RE.finditer(text)}
    return bool(
        {"node", "pipeline"} <= kinds
        and TARGET_STATION_RE.search(text)
        and TOPOLOGY_REQUEST_RE.search(text)
    )


def topology_summary_from_tool_outputs(
    tool_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    for item in reversed(tool_outputs):
        name = str(item.get("name") or item.get("tool_name") or "").casefold()
        if name != "analyze_pipeline_topology":
            continue
        output = item.get("output") if "output" in item else item.get("result")
        if not isinstance(output, dict) or output.get("success") is not True:
            continue
        summary = output.get("topology_summary")
        if isinstance(summary, dict) and summary:
            return dict(summary)
    return {}


def topology_quality_issues(answer: str, summary: Dict[str, Any]) -> List[str]:
    if not summary:
        return []
    issues = []
    expected_count = summary.get("direct_inbound_segment_count")
    for match in INBOUND_COUNT_RE.finditer(answer):
        claimed = _parse_count(match.group(1))
        if (
            expected_count is not None
            and claimed is not None
            and claimed != expected_count
        ):
            issues.append("topology_inbound_count_mismatch")
            break
    if summary.get("multi_source_reachable"):
        denies_multi = bool(DENIES_MULTI_SOURCE_RE.search(answer))
        unqualified_single = (
            "单源依赖" in answer
            and not QUALIFIED_SHARED_GATEWAY_RE.search(answer)
        )
        if denies_multi or unqualified_single:
            issues.append("topology_source_classification_mismatch")
    if EXCLUSIVE_CAUSE_RE.search(answer):
        issues.append("unsupported_exclusive_causal_claim")
    if TEMPORAL_LOAD_RE.search(answer):
        issues.append("unsupported_temporal_load_claim")
    return issues


def _parse_count(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value in CHINESE_NUMBERS:
        return CHINESE_NUMBERS[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + CHINESE_NUMBERS.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return CHINESE_NUMBERS.get(value[0], 0) * 10
    return None
