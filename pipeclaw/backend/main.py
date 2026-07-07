"""
FastAPI backend for GIS Pipeline Visualization.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from agent.orchestrator import init_orchestrator
from agent.router import router as agent_router
from data_loader import DataLoader
from models import ApiResponse, AvailableDates, ConsumerFlowData, NodeFlowData, PipelineFlowData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("backend.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).with_name(".env"), override=True)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    logger.info("✓ OPENAI_API_KEY 已加载: %s...%s", api_key[:10], api_key[-4:])
else:
    logger.warning("✗ 未找到 OPENAI_API_KEY")
logger.info("OPENAI_API_BASE 当前值: %s", os.getenv("OPENAI_API_BASE", "未设置"))
logger.info("OPENAI_MODEL 当前值: %s", os.getenv("OPENAI_MODEL", "未设置"))

app = FastAPI(title="GIS Pipeline API", description="API for gas pipeline network visualization and dispatch", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BACKEND_DIR = Path(__file__).parent
WORKSPACE_DIR = BACKEND_DIR / "pipeline_data"
PUBLIC_DIR = BACKEND_DIR / "public"
IMG_DIR = PUBLIC_DIR / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=PUBLIC_DIR), name="assets")

logger.info("Using data workspace: %s", WORKSPACE_DIR)
data_loader = DataLoader(WORKSPACE_DIR)
init_orchestrator(data_loader, agent_id="default")
app.include_router(agent_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "GIS Pipeline API is running",
        "data_workspace": WORKSPACE_DIR.as_posix(),
        "endpoints": {
            "node_flow": "/api/flow/nodes",
            "pipeline_flow": "/api/flow/pipelines",
            "consumer_flow": "/api/flow/consumers",
            "available_dates": "/api/dates",
            "agent_create": "/api/agent/create",
            "agent_chat": "/api/agent/chat",
            "agent_trace": "/api/agent/trace/{agent_id}/{session_id}",
            "agent_memory": "/api/agent/memory/{agent_id}/summary",
            "agent_suggestions": "/api/agent/suggestions",
            "agent_health": "/api/agent/health",
        },
    }


@app.get("/api/flow/nodes", response_model=NodeFlowData)
async def get_node_flow(query_date: str = Query(..., description="查询日期 (YYYY-MM-DD格式)")):
    try:
        dt = datetime.strptime(query_date, "%Y-%m-%d").date()
        return data_loader.load_node_flow(dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误,请使用YYYY-MM-DD格式")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载数据失败: {str(exc)}")


@app.get("/api/flow/pipelines", response_model=PipelineFlowData)
async def get_pipeline_flow(query_date: str = Query(..., description="查询日期 (YYYY-MM-DD格式)")):
    try:
        dt = datetime.strptime(query_date, "%Y-%m-%d").date()
        return data_loader.load_pipeline_flow(dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误,请使用YYYY-MM-DD格式")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载数据失败: {str(exc)}")


@app.get("/api/flow/consumers", response_model=ConsumerFlowData)
async def get_consumer_flow(query_date: str = Query(..., description="查询日期 (YYYY-MM-DD格式)")):
    try:
        dt = datetime.strptime(query_date, "%Y-%m-%d").date()
        return data_loader.load_consumer_flow(dt)
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误,请使用YYYY-MM-DD格式")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载数据失败: {str(exc)}")


@app.get("/api/dates", response_model=AvailableDates)
async def get_available_dates(data_type: str = Query(..., description="数据类型: node_flow, pipeline_flow, 或 consumer_flow")):
    try:
        if data_type not in ["node_flow", "pipeline_flow", "consumer_flow"]:
            raise HTTPException(status_code=400, detail="无效的data_type,请使用: node_flow, pipeline_flow, 或 consumer_flow")
        dates = data_loader.get_available_dates(data_type)
        date_range = {"start": dates[0], "end": dates[-1]} if dates else {}
        return AvailableDates(data_type=data_type, dates=dates, total_count=len(dates), date_range=date_range)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取日期列表失败: {str(exc)}")


@app.get("/api/flow/consumers/by-node", response_model=ConsumerFlowData)
async def get_consumers_by_node(station_name: str = Query(..., description="站名"), query_date: str = Query(..., description="查询日期 (YYYY-MM-DD格式)")):
    try:
        dt = datetime.strptime(query_date, "%Y-%m-%d").date()
        full_data = data_loader.load_consumer_flow(dt)
        filtered_records = [record for record in full_data.records if record.station_name == station_name]
        return ConsumerFlowData(date=full_data.date, records=filtered_records, total_records=len(filtered_records))
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误,请使用YYYY-MM-DD格式")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"加载数据失败: {str(exc)}")


@app.get("/api/dates/range")
async def get_date_range_summary():
    try:
        summary = {"node_flow": {}, "pipeline_flow": {}, "consumer_flow": {}}
        for key in summary.keys():
            dates = data_loader.get_available_dates(key)
            if dates:
                summary[key] = {"start": dates[0], "end": dates[-1], "count": len(dates)}
        return ApiResponse(success=True, message="获取日期范围成功", data=summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取日期范围失败: {str(exc)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
