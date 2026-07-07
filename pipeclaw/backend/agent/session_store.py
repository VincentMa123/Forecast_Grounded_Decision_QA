"""
Session 持久化存储
使用 JSON 文件保存会话历史到 backend/workspace/history 目录
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionStore:
    """基于 JSON 文件的 Session 存储"""

    def __init__(self, storage_dir: str = "workspace/history"):
        """
        初始化 Session 存储

        Args:
            storage_dir: 存储目录路径（相对于项目根目录）
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Dict[str, Any]] = {}

        logger.info(f"SessionStore 初始化完成，存储目录: {self.storage_dir.absolute()}")
        self._load_all_sessions()
        logger.info(f"已从存储中加载 {len(self._sessions)} 个会话")

    def _get_session_file_path(self, session_id: str) -> Path:
        """获取 session 文件路径"""
        return self.storage_dir / f"{session_id}.json"

    def _load_all_sessions(self):
        """启动时加载所有已存在的 session 文件到内存"""
        if not self.storage_dir.exists():
            logger.info(f"存储目录尚不存在: {self.storage_dir.absolute()}")
            return

        try:
            json_files = list(self.storage_dir.glob("*.json"))
            logger.info(f"在 {self.storage_dir} 中找到 {len(json_files)} 个会话文件")

            for file_path in json_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                        session_id = session_data.get("session_id")
                        if session_id:
                            self._sessions[session_id] = session_data
                            logger.debug(f"已加载会话: {session_id}")
                        else:
                            logger.warning(f"会话文件 {file_path} 缺少 session_id")
                except Exception as e:
                    logger.warning(f"读取会话文件 {file_path} 失败: {e}")
        except Exception as e:
            logger.error(f"从 {self.storage_dir} 加载会话失败: {e}")

    def _save_session_to_file(self, session_id: str):
        """保存单个 session 到文件"""
        if session_id not in self._sessions:
            return

        file_path = self._get_session_file_path(session_id)
        try:
            # 确保父目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._sessions[session_id], f, ensure_ascii=False, indent=2)
            logger.info(f"已将会话 {session_id} 保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存会话 {session_id} 到文件失败: {e}")

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取 session 数据"""
        return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        """检查 session 是否存在"""
        return session_id in self._sessions

    def create(self, session_id: str, initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建新 session

        Args:
            session_id: Session ID
            initial_data: 初始数据（可选）

        Returns:
            创建的 session 数据
        """
        now = datetime.now().isoformat()
        session_data = initial_data or {}
        session_data.update({
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
        })

        # 确保必要的字段存在
        if "title" not in session_data:
            session_data["title"] = "自动创建的会话"
        if "context" not in session_data:
            session_data["context"] = {}
        if "messages" not in session_data:
            session_data["messages"] = []

        self._sessions[session_id] = session_data
        self._save_session_to_file(session_id)
        logger.info(f"已创建会话: {session_id}")
        return session_data

    def update(self, session_id: str, updates: Dict[str, Any]):
        """
        更新 session 数据

        Args:
            session_id: Session ID
            updates: 要更新的字段
        """
        if session_id not in self._sessions:
            logger.warning(f"尝试更新不存在的会话: {session_id}")
            return

        self._sessions[session_id].update(updates)
        self._sessions[session_id]["updated_at"] = datetime.now().isoformat()
        self._save_session_to_file(session_id)

    def delete(self, session_id: str) -> bool:
        """
        删除 session

        Args:
            session_id: Session ID

        Returns:
            是否成功删除
        """
        existed = session_id in self._sessions

        if existed:
            # 从内存中删除
            del self._sessions[session_id]

            # 从文件系统中删除
            file_path = self._get_session_file_path(session_id)
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"已删除会话文件: {file_path}")
            except Exception as e:
                logger.error(f"删除会话文件 {file_path} 失败: {e}")

        return existed

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """获取所有 session"""
        return self._sessions.copy()

    def cleanup_old_sessions(self, max_age_days: int = 30):
        """
        清理超过指定天数的旧 session

        Args:
            max_age_days: 最大保留天数
        """
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        sessions_to_delete = []

        for session_id, session_data in self._sessions.items():
            updated_at_str = session_data.get("updated_at")
            if updated_at_str:
                try:
                    updated_at = datetime.fromisoformat(updated_at_str)
                    if updated_at < cutoff_time:
                        sessions_to_delete.append(session_id)
                except Exception as e:
                    logger.warning(f"解析会话 {session_id} 的 updated_at 失败: {e}")

        for session_id in sessions_to_delete:
            self.delete(session_id)
            logger.info(f"已清理过期会话: {session_id}")


# 全局单例
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """获取全局 SessionStore 实例"""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
