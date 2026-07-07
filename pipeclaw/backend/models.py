"""
Pydantic数据模型定义
用于描述GIS管网系统中的各类数据结构
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import datetime


# ===================== Node Flow 数据模型 =====================

class NodeFlowRecord(BaseModel):
    """节点流量记录 - 对应node_flow目录中的每日CSV文件"""
    pipeline_division: str = Field(..., description="管道划分/系统名称")
    station_name: str = Field(..., description="站点名称")
    lon: float = Field(..., description="经度")
    lat: float = Field(..., description="纬度")
    node_type: str = Field(..., description="节点类型,如'气源'")
    control_type: str = Field(..., description="控制类型,如'定流量'")
    input_flow: Optional[float] = Field(None, description="输入流量值")
    calculated_flow: float = Field(..., description="计算流量值")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_division": "中俄东线",
                "station_name": "临沂站",
                "lon": 105.959,
                "lat": 32.3316,
                "node_type": "气源",
                "control_type": "定流量",
                "input_flow": None,
                "calculated_flow": -2917.277
            }
        }
    }


class NodeFlowData(BaseModel):
    """特定日期的节点流量数据集合"""
    date: datetime.date = Field(..., description="数据日期")
    records: List[NodeFlowRecord] = Field(default_factory=list, description="节点流量记录列表")
    total_records: int = Field(0, description="记录总数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2019-01-01",
                "records": [],
                "total_records": 0
            }
        }
    }


# ===================== Pipeline Flow 数据模型 =====================

class PipelineFlowRecord(BaseModel):
    """管段流量记录 - 对应pipeline_flow目录中的每日CSV文件"""
    start_station: str = Field(..., description="管段起点站名")
    end_station: str = Field(..., description="管段终点站名")
    pipeline_type: str = Field(..., description="管段类型,如'管段'")
    pipeline_division: str = Field(..., description="所属管道划分/系统")
    pipeline_flow: float = Field(..., description="管道流量值")

    model_config = {
        "json_schema_extra": {
            "example": {
                "start_station": "临沂站",
                "end_station": "连云港站",
                "pipeline_type": "管段",
                "pipeline_division": "中俄东线",
                "pipeline_flow": 759.13
            }
        }
    }


class PipelineFlowData(BaseModel):
    """特定日期的管段流量数据集合"""
    date: datetime.date = Field(..., description="数据日期")
    records: List[PipelineFlowRecord] = Field(default_factory=list, description="管段流量记录列表")
    total_records: int = Field(0, description="记录总数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2019-01-01",
                "records": [],
                "total_records": 0
            }
        }
    }


# ===================== Consumer Flow 数据模型 =====================

class ConsumerFlowRecord(BaseModel):
    """消耗量流量记录 - 对应consumer_flow目录中的每日CSV文件"""
    pipeline: str = Field(..., description="管线系统名称")
    location: str = Field(..., description="所属地名称")
    supply_point: str = Field(..., description="供气点名称")
    station_name: str = Field(..., description="匹配的站点名称")
    consumer: str = Field(..., description="用户/公司名称")
    consumption: float = Field(..., description="消耗量值")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline": "西二、三线西段",
                "location": "新疆",
                "supply_point": "霍尔果斯首站",
                "station_name": "霍尔果斯站",
                "consumer": "伊宁新捷",
                "consumption": 39.3166
            }
        }
    }


class ConsumerFlowData(BaseModel):
    """特定日期的消耗量数据集合"""
    date: datetime.date = Field(..., description="数据日期")
    records: List[ConsumerFlowRecord] = Field(default_factory=list, description="消耗量记录列表")
    total_records: int = Field(0, description="记录总数")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2019-01-01",
                "records": [],
                "total_records": 0
            }
        }
    }


# ===================== 查询参数模型 =====================

class DateRangeQuery(BaseModel):
    """日期范围查询参数"""
    start_date: datetime.date = Field(..., description="开始日期")
    end_date: datetime.date = Field(..., description="结束日期")

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2019-01-01",
                "end_date": "2019-01-31"
            }
        }


class AvailableDates(BaseModel):
    """可用日期列表响应"""
    data_type: str = Field(..., description="数据类型: node_flow, pipeline_flow, node_consumer")
    dates: List[str] = Field(default_factory=list, description="可用日期列表(YYYY-MM-DD格式)")
    total_count: int = Field(0, description="总数量")
    date_range: Dict[str, str] = Field(
        default_factory=dict,
        description="日期范围,包含start和end"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data_type": "node_flow",
                "dates": ["2019-01-01", "2019-01-02"],
                "total_count": 2,
                "date_range": {
                    "start": "2019-01-01",
                    "end": "2019-12-31"
                }
            }
        }


# ===================== API响应模型 =====================

class ApiResponse(BaseModel):
    """通用API响应模型"""
    success: bool = Field(True, description="请求是否成功")
    message: str = Field("", description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "数据获取成功",
                "data": {}
            }
        }
