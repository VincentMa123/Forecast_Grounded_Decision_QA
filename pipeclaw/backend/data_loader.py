"""
数据加载工具模块
用于从CSV文件加载和处理各类数据
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .models import (
    ConsumerFlowData,
    ConsumerFlowRecord,
    NodeFlowData,
    NodeFlowRecord,
    PipelineFlowData,
    PipelineFlowRecord,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""

    def __init__(self, base_dir: Path):
        """
        初始化数据加载器
        Args:
            base_dir: backend/pipeline_data path
        """
        self.base_dir = Path(base_dir)
        self.node_flow_dir = self.base_dir / "node_flow"
        self.pipeline_flow_dir = self.base_dir / "pipeline_flow"
        self.consumer_flow_dir = self.base_dir / "consumer_flow"
        self._available_dates_cache: Dict[str, List[str]] = {}
        self._supply_point_to_station = self._load_supply_point_mapping()

    def _load_supply_point_mapping(self) -> Dict[str, str]:
        mapping_file = self.base_dir / "consumer_station.csv"
        if not mapping_file.exists():
            logger.warning("consumer_station.csv not found at %s", mapping_file)
            return {}

        try:
            df = pd.read_csv(mapping_file, encoding="utf-8")
            mapping: Dict[str, str] = {}
            for _, row in df.iterrows():
                supply_point = str(row["供气点"])
                station_name = str(row["匹配站名"])
                mapping[supply_point] = station_name
            return mapping
        except Exception as exc:
            logger.exception("Error loading supply point mapping from %s: %s", mapping_file, exc)
            return {}

    def get_available_dates(self, data_type: str) -> List[str]:
        if data_type in self._available_dates_cache:
            return self._available_dates_cache[data_type]

        if data_type == "node_flow":
            directory = self.node_flow_dir
            pattern = "*_node.csv"
        elif data_type == "pipeline_flow":
            directory = self.pipeline_flow_dir
            pattern = "*_pipeline.csv"
        elif data_type == "consumer_flow":
            directory = self.consumer_flow_dir
            pattern = "*_consumer.csv"
        else:
            return []

        dates = set()
        for file_path in directory.glob(pattern):
            date_str = file_path.stem.split("_")[0]
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                dates.add(dt.strftime("%Y-%m-%d"))
            except ValueError:
                continue

        result = sorted(dates)
        self._available_dates_cache[data_type] = result
        return result

    def load_node_flow(self, query_date: date) -> NodeFlowData:
        date_str = query_date.strftime("%Y%m%d")
        file_path = self.node_flow_dir / f"{date_str}_node.csv"

        if not file_path.exists():
            logger.warning("Node flow file not found: %s", file_path)
            return NodeFlowData(date=query_date, records=[], total_records=0)

        try:
            logger.info("开始加载节点流量文件: %s", file_path)
            df = pd.read_csv(file_path, encoding="utf-8")
            records = [
                NodeFlowRecord(
                    pipeline_division=str(row["管道划分"]),
                    station_name=str(row["站名"]),
                    lon=float(row["lon"]),
                    lat=float(row["lat"]),
                    node_type=str(row["类型"]),
                    control_type=str(row["控制类型"]),
                    input_flow=float(row["输入流量"]) if pd.notna(row["输入流量"]) else None,
                    calculated_flow=float(row["计算流量"]),
                )
                for _, row in df.iterrows()
            ]
            logger.info("已生成 %s 条 NodeFlowRecord 记录", len(records))
            return NodeFlowData(date=query_date, records=records, total_records=len(records))
        except Exception as exc:
            logger.exception("加载节点流量文件 %s 失败: %s", file_path, exc)
            return NodeFlowData(date=query_date, records=[], total_records=0)

    def load_pipeline_flow(self, query_date: date) -> PipelineFlowData:
        date_str = query_date.strftime("%Y%m%d")
        file_path = self.pipeline_flow_dir / f"{date_str}_pipeline.csv"

        if not file_path.exists():
            logger.warning("Pipeline flow file not found: %s", file_path)
            return PipelineFlowData(date=query_date, records=[], total_records=0)

        try:
            logger.info("开始加载管段流量文件: %s", file_path)
            df = pd.read_csv(file_path, encoding="utf-8")
            records = [
                PipelineFlowRecord(
                    start_station=str(row["起点站名"]),
                    end_station=str(row["终点站名"]),
                    pipeline_type=str(row["类型"]),
                    pipeline_division=str(row["管道划分"]),
                    pipeline_flow=float(row["管道流量"]),
                )
                for _, row in df.iterrows()
            ]
            logger.info("已生成 %s 条 PipelineFlowRecord 记录", len(records))
            return PipelineFlowData(date=query_date, records=records, total_records=len(records))
        except Exception as exc:
            logger.exception("加载管段流量文件 %s 失败: %s", file_path, exc)
            return PipelineFlowData(date=query_date, records=[], total_records=0)

    def _date_range(self, start_date: date, end_date: date) -> List[date]:
        if end_date < start_date:
            raise ValueError(f"end_date {end_date} is earlier than start_date {start_date}")
        current = start_date
        dates: List[date] = []
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    def load_node_flow_range(self, start_date: date, end_date: date) -> NodeFlowData:
        records: List[NodeFlowRecord] = []
        for current_date in self._date_range(start_date, end_date):
            payload = self.load_node_flow(current_date)
            records.extend(payload.records)
        return NodeFlowData(date=start_date, records=records, total_records=len(records))

    def load_pipeline_flow_range(self, start_date: date, end_date: date) -> PipelineFlowData:
        records: List[PipelineFlowRecord] = []
        for current_date in self._date_range(start_date, end_date):
            payload = self.load_pipeline_flow(current_date)
            records.extend(payload.records)
        return PipelineFlowData(date=start_date, records=records, total_records=len(records))

    def load_consumer_flow_range(self, start_date: date, end_date: date) -> ConsumerFlowData:
        records: List[ConsumerFlowRecord] = []
        for current_date in self._date_range(start_date, end_date):
            payload = self.load_consumer_flow(current_date)
            records.extend(payload.records)
        return ConsumerFlowData(date=start_date, records=records, total_records=len(records))

    def load_consumer_flow(self, query_date: date) -> ConsumerFlowData:
        date_str = query_date.strftime("%Y%m%d")
        file_path = self.consumer_flow_dir / f"{date_str}_consumer.csv"

        if not file_path.exists():
            logger.warning("Consumer flow file not found: %s", file_path)
            return ConsumerFlowData(date=query_date, records=[], total_records=0)

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            records = []
            for _, row in df.iterrows():
                supply_point = str(row["供气点"])
                station_name = self._supply_point_to_station.get(supply_point, supply_point)
                records.append(
                    ConsumerFlowRecord(
                        pipeline=str(row["管线"]) if pd.notna(row["管线"]) else "",
                        location=str(row["所属地"]) if pd.notna(row["所属地"]) else "",
                        supply_point=supply_point,
                        station_name=station_name,
                        consumer=str(row["用户"]),
                        consumption=float(row["消耗量"]),
                    )
                )
            return ConsumerFlowData(date=query_date, records=records, total_records=len(records))
        except Exception as exc:
            logger.exception("Error loading consumer flow data from %s: %s", file_path, exc)
            return ConsumerFlowData(date=query_date, records=[], total_records=0)

    def resolve_smoke_window(self, start_date: date, end_date: date, data_type: str) -> tuple[date, date]:
        available_dates = self.get_available_dates(data_type)
        if not available_dates:
            raise ValueError(f"No available dates for data_type={data_type}")
        available = [datetime.strptime(item, "%Y-%m-%d").date() for item in available_dates]
        span_days = max(0, (end_date - start_date).days)
        if len(available) <= span_days:
            return available[0], available[-1]
        target_ordinal = start_date.toordinal()
        best_index = 0
        best_delta = None
        for index, candidate in enumerate(available[: len(available) - span_days]):
            delta = abs(candidate.toordinal() - target_ordinal)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index
        resolved_start = available[best_index]
        resolved_end = available[best_index + span_days]
        return resolved_start, resolved_end

    def build_node_station_daily_totals(self, start_date: date, end_date: date) -> Dict[str, float]:
        totals: Dict[str, float] = defaultdict(float)
        for record in self.load_node_flow_range(start_date, end_date).records:
            if record.node_type == "气源":
                continue
            totals[record.station_name] += abs(float(record.calculated_flow or 0.0))
        return dict(totals)
