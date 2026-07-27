from __future__ import annotations

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.scenario_preflight import (
    DATA_FILE_RE,
    DATA_FILE_SUBDIRECTORIES,
    DEFAULT_PIPELINE_DATA_ROOT,
)
from .pipeline_scope import filter_rows_by_named_pipeline


TARGET_STATION_RE = re.compile(r"(?:\u5230|\u901a\u5411|\u81f3)\s*([\u4e00-\u9fff]{2,16}\u7ad9)")
INBOUND_COUNT_RE = re.compile(
    r"(?:\u5168\u90e8|\u5171)?\s*([\u4e00-\u5341\d]+)\s*\u6761(?:\u76f4\u63a5)?\u5165\u7ad9(?:\u7ba1\u9053|\u7ba1\u7ebf)"
)
EXCLUSIVE_CAUSE_RE = re.compile(r"(?:\u539f\u56e0)?\u53ea\u80fd(?:\u662f|\u5728)?\u4e0a\u6e38|\u53ea\u53ef\u80fd(?:\u662f|\u5728)?\u4e0a\u6e38")
TEMPORAL_LOAD_RE = re.compile(r"\u591c\u91cc(?:\u4e00\u822c|\u901a\u5e38)?\u662f\u4f4e\u8d1f\u8377|\u591c\u91cc.{0,12}\u8d1f\u8377(?:\u66f2\u7ebf)?(?:\u5728)?\u4e0b\u964d")
DENIES_MULTI_SOURCE_RE = re.compile(r"\u4e0d\u662f.{0,8}\u591a\u6e90\u53ef\u8fbe|\u800c\u975e.{0,8}\u591a\u6e90\u53ef\u8fbe")
QUALIFIED_SHARED_GATEWAY_RE = re.compile(r"\u591a\u6e90\u53ef\u8fbe|\u5171\u4eab(?:\u7f51\u5173|\u4e0a\u6e38|\u5165\u53e3)|\u5171\u540c(?:\u7f51\u5173|\u4e0a\u6e38|\u5165\u53e3)")
TOPOLOGY_REQUEST_RE = re.compile(
    r"\u53ef\u8fbe|\u6700\u77ed\u8def\u5f84|\u8def\u5f84|\u53cd\u5411\u8ffd|\u5171\u4eab\u7f51\u5173|\u5355\u6e90\u4f9d\u8d56|\u591a\u6e90"
    r"|\b(?:reachable|reachability|shortest path|shared gateway|multi.source|single.source)\b",
    re.IGNORECASE,
)
CHINESE_NUMBERS = {
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
    "\u5341": 10,
}

CANONICAL_STATION_ALIASES = {
    "通州南分输站": "通州南站",
    "湘潭分输站": "湘潭站",
    "乌鲁木齐压气站": "乌鲁木齐站",
}


def build_topology_evidence(
    text: str,
    *,
    pipeline_data_root: Path | None = None,
) -> Dict[str, Any]:
    """Derive compact graph facts from explicitly referenced daily CSV files."""
    summary, _ = build_topology_evidence_result(
        text,
        pipeline_data_root=pipeline_data_root,
    )
    return summary


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
            station = str(row.get("\u7ad9\u540d") or row.get("station") or "").strip()
            kind = str(row.get("\u7c7b\u578b") or row.get("type") or "").strip()
            pipeline = str(row.get("\u7ba1\u9053\u5212\u5206") or row.get("pipeline") or "").strip()
            if station:
                station_types[station].add(kind)
                if "\u6c14\u6e90" in kind or kind.casefold() == "source":
                    source_instances.append({"station": station, "pipeline": pipeline})

    pipeline_rows: List[Dict[str, str]] = []
    with pipeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = str(row.get("\u8d77\u70b9\u7ad9\u540d") or row.get("source") or "").strip()
            target = str(row.get("\u7ec8\u70b9\u7ad9\u540d") or row.get("target") or "").strip()
            if not source or not target:
                continue
            pipeline_rows.append({
                "source": source,
                "target": target,
                "pipeline": str(row.get("\u7ba1\u9053\u5212\u5206") or row.get("pipeline") or "").strip(),
                "flow": str(row.get("\u7ba1\u9053\u6d41\u91cf") or row.get("flow") or "").strip(),
            })

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
        station
        for row in pipeline_rows
        for station in (row["source"], row["target"])
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
            if "\u6c14\u6e90" in station_types.get(station, set())
            or "source" in {value.casefold() for value in station_types.get(station, set())}
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
        example_shortest_path.append({
            "source": edge["source"],
            "target": edge["target"],
            "pipeline": edge["pipeline"],
        })
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


def topology_summary_from_tool_outputs(tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        if expected_count is not None and claimed is not None and claimed != expected_count:
            issues.append("topology_inbound_count_mismatch")
            break
    if summary.get("multi_source_reachable"):
        denies_multi = bool(DENIES_MULTI_SOURCE_RE.search(answer))
        unqualified_single = "\u5355\u6e90\u4f9d\u8d56" in answer and not QUALIFIED_SHARED_GATEWAY_RE.search(answer)
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
    if value.startswith("\u5341") and len(value) == 2:
        return 10 + CHINESE_NUMBERS.get(value[1], 0)
    if value.endswith("\u5341") and len(value) == 2:
        return CHINESE_NUMBERS.get(value[0], 0) * 10
    return None
