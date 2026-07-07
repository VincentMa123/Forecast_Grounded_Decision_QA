import { useEffect, useMemo, useState } from 'react';
import {
  agentApi,
  AGENT_STORAGE_KEY,
  AgentChatResponse,
  DEFAULT_AGENT_ID,
  MemorySummaryResponse,
  SESSION_STORAGE_KEY,
  TraceSummary,
  WorkspaceFileResponse,
  WorkspaceTreeNode,
  WorkspaceTreeResponse,
} from '../api/agent';
import { ChatPanel } from './ChatPanel';
import './RightDock.css';

type DockTab = 'chat' | 'workspace' | 'memory';

interface RightDockProps {
  currentDate?: string;
  selectedItem?: {
    type: 'system' | 'node' | 'edge';
    id: string;
  };
}

const TAB_LABELS: Array<{ key: DockTab; label: string }> = [
  { key: 'chat', label: 'Chat' },
  { key: 'workspace', label: 'Workspace' },
  { key: 'memory', label: 'Memory' },
];

const WORKSPACE_TREE_DEPTH = 6;
const ROOT_TREE_PATH = '.';

function formatBytes(size?: number | null): string {
  if (!size) return '0 B';
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function collectDirectoryPaths(node: WorkspaceTreeNode): string[] {
  if (node.node_type === 'file') return [];
  return [node.path, ...node.children.flatMap((child) => collectDirectoryPaths(child))];
}

export function RightDock({ currentDate, selectedItem }: RightDockProps) {
  const [activeTab, setActiveTab] = useState<DockTab>('chat');
  const [agentId, setAgentId] = useState<string>(() => {
    if (typeof window === 'undefined') return DEFAULT_AGENT_ID;
    return localStorage.getItem(AGENT_STORAGE_KEY) || DEFAULT_AGENT_ID;
  });
  const [sessionId, setSessionId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(SESSION_STORAGE_KEY);
  });
  const [memorySummary, setMemorySummary] = useState<MemorySummaryResponse | null>(null);
  const [traceSummary, setTraceSummary] = useState<TraceSummary | null>(null);
  const [loadingMemory, setLoadingMemory] = useState(false);
  const [workspaceTree, setWorkspaceTree] = useState<WorkspaceTreeResponse | null>(null);
  const [workspaceFile, setWorkspaceFile] = useState<WorkspaceFileResponse | null>(null);
  const [loadingTree, setLoadingTree] = useState(false);
  const [loadingFile, setLoadingFile] = useState(false);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set([ROOT_TREE_PATH]));

  const refreshMemory = async (targetAgentId: string, targetSessionId?: string | null) => {
    setLoadingMemory(true);
    try {
      const summary = await agentApi.getMemorySummary(targetAgentId, targetSessionId || undefined);
      setMemorySummary(summary);
      setTraceSummary(summary.latest_trace_summary || null);
    } catch (error) {
      console.error('加载 memory summary 失败:', error);
    } finally {
      setLoadingMemory(false);
    }
  };

  const refreshWorkspaceTree = async (targetAgentId: string) => {
    setLoadingTree(true);
    setTreeError(null);
    try {
      const tree = await agentApi.getWorkspaceTree(targetAgentId, ROOT_TREE_PATH, WORKSPACE_TREE_DEPTH);
      setWorkspaceTree(tree);
      const directoryPaths = new Set(collectDirectoryPaths(tree.tree));
      setExpandedPaths((prev) => {
        const next = new Set<string>([ROOT_TREE_PATH]);
        for (const path of prev) {
          if (directoryPaths.has(path)) next.add(path);
        }
        return next;
      });
    } catch (error) {
      console.error('加载 workspace tree 失败:', error);
      setTreeError('加载文件树失败');
    } finally {
      setLoadingTree(false);
    }
  };

  const openWorkspaceFile = async (path: string) => {
    setLoadingFile(true);
    setFileError(null);
    try {
      const file = await agentApi.getWorkspaceFile(agentId, path);
      setWorkspaceFile(file);
      setActiveTab('workspace');
    } catch (error) {
      console.error('打开 workspace 文件失败:', error);
      setFileError('打开文件失败');
    } finally {
      setLoadingFile(false);
    }
  };

  useEffect(() => {
    refreshMemory(agentId, sessionId);
    refreshWorkspaceTree(agentId);
  }, [agentId, sessionId]);

  const handleAgentResponse = async (response: AgentChatResponse) => {
    const nextAgentId = response.agent_id || 'default';
    setAgentId(nextAgentId);
    setSessionId(response.session_id);
    setTraceSummary(response.trace_summary);
    setExpandedPaths(new Set([ROOT_TREE_PATH]));
    await Promise.all([
      refreshMemory(nextAgentId, response.session_id),
      refreshWorkspaceTree(nextAgentId),
    ]);
  };

  const selectionSummary = useMemo(() => {
    if (!selectedItem) return '当前未选中地图元素';
    return `当前选中：${selectedItem.type} / ${selectedItem.id}`;
  }, [selectedItem]);

  const toggleDirectory = (path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      next.add(ROOT_TREE_PATH);
      return next;
    });
  };

  const renderTree = (node: WorkspaceTreeNode, depth = 0): JSX.Element => {
    const isFile = node.node_type === 'file';
    const isExpanded = isFile ? false : expandedPaths.has(node.path);
    const isRoot = node.path === ROOT_TREE_PATH;
    const showChildren = !isFile && (isRoot || isExpanded);
    return (
      <div key={node.path} className="workspace-tree-node">
        <button
          type="button"
          className={`workspace-tree-row ${isFile ? 'is-file' : 'is-directory'} ${workspaceFile?.path === node.path ? 'active' : ''}`}
          style={{ paddingLeft: 10 + depth * 14 }}
          onClick={() => {
            if (isFile) {
              void openWorkspaceFile(node.path);
            } else {
              toggleDirectory(node.path);
            }
          }}
        >
          <span className={`workspace-tree-chevron ${showChildren ? 'expanded' : ''} ${!isFile && node.children?.length > 0 ? '' : 'placeholder'}`}>
            {!isFile && node.children?.length > 0 ? '▶' : ''}
          </span>
          <span className="workspace-tree-icon">{isFile ? '📄' : '📁'}</span>
          <span className="workspace-tree-name">{isRoot ? 'workspace' : (node.name || node.path)}</span>
          {isFile && <span className="workspace-tree-size">{formatBytes(node.size_bytes)}</span>}
        </button>
        {showChildren && node.children?.length > 0 && (
          <div className="workspace-tree-children">
            {node.children.map((child) => renderTree(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="right-dock">
      <div className="right-dock-tabs">
        {TAB_LABELS.map((tab) => (
          <button key={tab.key} className={`right-dock-tab ${activeTab === tab.key ? 'active' : ''}`} onClick={() => setActiveTab(tab.key)}>
            {tab.label}
          </button>
        ))}
      </div>

      <div className="right-dock-body">
        <div className={`dock-panel ${activeTab === 'chat' ? 'active' : 'hidden'}`}>
          <ChatPanel currentDate={currentDate} selectedItem={selectedItem} onAgentChange={setAgentId} onSessionChange={setSessionId} onAgentResponse={handleAgentResponse} />
        </div>

        <div className={`dock-panel ${activeTab === 'workspace' ? 'active' : 'hidden'}`}>
          <div className="dock-section workspace-shell">
            <div className="dock-section-toolbar">
              <div>
                <h4>Workspace Browser</h4>
                <p className="dock-caption">{selectionSummary}</p>
              </div>
              <div className="dock-toolbar-actions">
                <button onClick={() => refreshWorkspaceTree(agentId)} disabled={loadingTree}>刷新</button>
              </div>
            </div>

            <div className="dock-card workspace-overview-card">
              <div>
                <strong>Current Workspace</strong>
                <p>agent_id: {agentId}</p>
                <p>session_id: {sessionId || '暂无'}</p>
                <p>{workspaceTree?.workspace_root || memorySummary?.workspace_root || '尚未初始化'}</p>
              </div>
              <div className="workspace-overview-metrics">
                <span>{workspaceTree?.total_directories ?? 0} dirs</span>
                <span>{workspaceTree?.total_files ?? 0} files</span>
              </div>
            </div>

            <div className="workspace-grid">
              <div className="dock-card workspace-tree-card">
                <div className="workspace-card-header">
                  <strong>文件树</strong>
                  <span>{workspaceTree?.requested_path || ROOT_TREE_PATH}</span>
                </div>
                {treeError ? <p className="dock-empty">{treeError}</p> : null}
                {loadingTree ? <p className="dock-empty">正在加载文件树...</p> : null}
                {!loadingTree && workspaceTree ? (
                  <div className="workspace-tree">{renderTree(workspaceTree.tree)}</div>
                ) : null}
              </div>

              <div className="dock-card workspace-preview-card">
                <div className="workspace-card-header">
                  <strong>文件预览</strong>
                  <span>{workspaceFile?.path || '尚未选择文件'}</span>
                </div>
                {fileError ? <p className="dock-empty">{fileError}</p> : null}
                {loadingFile ? <p className="dock-empty">正在读取文件...</p> : null}
                {!loadingFile && workspaceFile ? (
                  <>
                    <div className="workspace-file-meta">
                      <span>{formatBytes(workspaceFile.size_bytes)}</span>
                      <span>{workspaceFile.is_text ? 'text' : 'binary'}</span>
                      {workspaceFile.truncated ? <span>truncated</span> : null}
                    </div>
                    {workspaceFile.is_text ? (
                      <pre className="workspace-file-content">{workspaceFile.content || '文件为空'}</pre>
                    ) : (
                      <div className="workspace-binary-empty">该文件类型暂不支持文本预览</div>
                    )}
                  </>
                ) : null}
                {!loadingFile && !workspaceFile ? <p className="dock-empty">点击左侧文件即可查看内容。</p> : null}
              </div>
            </div>
          </div>
        </div>

        <div className={`dock-panel ${activeTab === 'memory' ? 'active' : 'hidden'}`}>
          <div className="dock-section">
            <div className="dock-section-toolbar">
              <div>
                <h4>Trace & Memory</h4>
                <p className="dock-caption">围绕当前 Agent 会话的运行审计与 timeline 记忆。</p>
              </div>
              <button onClick={() => refreshMemory(agentId, sessionId)} disabled={loadingMemory}>刷新</button>
            </div>

            <div className="dock-card">
              <strong>Current Workspace</strong>
              <p>agent_id: {agentId}</p>
              <p>session_id: {sessionId || '暂无'}</p>
              <p>{memorySummary?.workspace_root || '尚未初始化'}</p>
            </div>

            <div className="dock-card">
              <strong>Latest Trace</strong>
              {traceSummary ? (
                <>
                  <p>status: {traceSummary.status}</p>
                  <p>updated_at: {traceSummary.updated_at || '暂无'}</p>
                  <p>messages: {traceSummary.messages_count}</p>
                  <p>tool_calls: {traceSummary.tool_calls_count}</p>
                  <p>artifacts: {traceSummary.artifacts_count}</p>
                  <p>{traceSummary.trace_path}</p>
                </>
              ) : (
                <p className="dock-empty">暂无 trace。</p>
              )}
            </div>

            <div className="dock-card">
              <strong>Timeline Memory</strong>
              {memorySummary?.timeline_files?.length ? (
                <ul>
                  {memorySummary.timeline_files.map((item, idx) => <li key={idx}>{item}</li>)}
                </ul>
              ) : (
                <p>暂无 timeline 文件</p>
              )}
            </div>

            <div className="dock-card">
              <strong>Latest Timeline Excerpt</strong>
              <p>{memorySummary?.latest_timeline_path || '暂无'}</p>
              <pre className="dock-pre">{memorySummary?.latest_timeline_excerpt || '暂无内容'}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
