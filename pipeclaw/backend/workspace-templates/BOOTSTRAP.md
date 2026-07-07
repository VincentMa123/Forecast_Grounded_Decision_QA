# BOOTSTRAP

启动时约定：

1. 初始化或加载 `workspace-{agent_id}`
2. 确保 `memory/`、`assets/`、`context_trace/`、`temporary_dir/`、`reports/` 存在
3. 删除旧的 `plan.md`
4. 创建或打开 `context_trace/<session_id>.json`
5. 扫描控制面文件、全部 memory 文件与技能索引
6. 组装 system prompt 后执行当前 turn