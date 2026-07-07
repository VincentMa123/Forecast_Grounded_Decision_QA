# TOOLS

可用工具：

- `write_file(path, content)`
- `edit_file(path, old_string, new_string, replace_all=false)`
- `read_file(path, offset?, limit?)`
- `run_command(cmd, timeout_s=30, cwd?)`

要求：

- 先读再改
- 修改 `plan.md` 前不需要保留历史版本
- 所有写入应落在 agent workspace 内
- 大部分中间结果、临时文件、调试产物统一写入 `temporary_dir/`
- 报告、导出结果、最终交付物统一写入 `reports/`
- 除控制面文件外，不要把新生成文件直接散落在 workspace 根目录