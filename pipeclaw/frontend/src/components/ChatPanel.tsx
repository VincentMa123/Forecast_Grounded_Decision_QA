import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  agentApi,
  AGENT_STORAGE_KEY,
  AgentChatRequest,
  AgentChatResponse,
  CreateAgentResponse,
  DEFAULT_AGENT_ID,
  QuestionSuggestion,
  SessionHistoryItem,
  SESSION_STORAGE_KEY,
  SKILL_STORAGE_KEY,
  TraceData,
} from '../api/agent';
import './ChatPanel.css';

interface ChatPanelProps {
  currentDate?: string;
  selectedItem?: {
    type: 'system' | 'node' | 'edge';
    id: string;
  };
  onSessionChange?: (sessionId: string | null) => void;
  onAgentChange?: (agentId: string) => void;
  onAgentResponse?: (response: AgentChatResponse) => void;
}

interface ToolCallItem {
  id: string;
  tool: string;
  input?: unknown;
  output?: string;
  status: 'running' | 'completed' | 'error';
  startedAt?: string;
  endedAt?: string;
  collapsed: boolean;
}

interface UserTimelineItem {
  id: string;
  kind: 'user';
  content: string;
  timestamp: string;
}

interface AssistantTurnItem {
  id: string;
  kind: 'assistant_turn';
  timestamp: string;
  status: 'streaming' | 'completed' | 'error';
  toolCalls: ToolCallItem[];
  toolsCollapsed: boolean;
  finalReply: string;
  finalReplyTimestamp: string;
}

type TimelineItem = UserTimelineItem | AssistantTurnItem;

function createTimelineId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function createSessionId(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, '0');
  const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  return `${timestamp}_${Math.random().toString(36).slice(2, 8)}`;
}

function getToolStatusLabel(status: ToolCallItem['status']): string {
  if (status === 'running') return '运行中';
  if (status === 'error') return '失败';
  return '完成';
}

function formatTime(value?: string | null): string {
  if (!value) return '--';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function getSessionDisplayTitle(session: SessionHistoryItem): string {
  return session.first_user_message?.trim() || '未命名对话';
}

export const ChatPanel: React.FC<ChatPanelProps> = ({ currentDate, selectedItem, onSessionChange, onAgentChange, onAgentResponse }) => {
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [input, setInput] = useState('');
  const [newAgentIdInput, setNewAgentIdInput] = useState('');
  const [createAgentError, setCreateAgentError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [creatingAgent, setCreatingAgent] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(() => {
    if (typeof window === 'undefined') return null;
    const url = new URL(window.location.href);
    return url.searchParams.get('session_id') || localStorage.getItem(SESSION_STORAGE_KEY);
  });
  const [agentId, setAgentId] = useState<string>(() => {
    if (typeof window === 'undefined') return DEFAULT_AGENT_ID;
    return localStorage.getItem(AGENT_STORAGE_KEY) || DEFAULT_AGENT_ID;
  });
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryItem[]>([]);
  const [sessionHistoryError, setSessionHistoryError] = useState<string | null>(null);
  const [loadingSessionHistory, setLoadingSessionHistory] = useState(false);
  const [allSuggestions, setAllSuggestions] = useState<QuestionSuggestion[]>([]);
  const [displayedSuggestions, setDisplayedSuggestions] = useState<QuestionSuggestion[]>([]);
  const [suggestionStartIndex, setSuggestionStartIndex] = useState(0);
  const [suggestionsVisible, setSuggestionsVisible] = useState(true);
  const [agentMode, setAgentMode] = useState<'mock' | 'production'>('mock');
  const [enableSkills, setEnableSkills] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(SKILL_STORAGE_KEY) === '1';
  });

  const SUGGESTIONS_PER_PAGE = 4;
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const normalizeMessageContent = (content: unknown): string => {
    if (typeof content === 'string') return content;
    if (!content) return '';
    try {
      return JSON.stringify(content, null, 2);
    } catch {
      return String(content);
    }
  };

  const extractErrorMessage = (error: unknown, fallback: string): string => {
    if (axios.isAxiosError(error)) {
      return (error.response?.data as { detail?: string } | undefined)?.detail || error.message || fallback;
    }
    if (error instanceof Error) return error.message;
    return fallback;
  };

  const updateUrlSession = (nextSessionId: string | null) => {
    if (typeof window === 'undefined') return;
    const url = new URL(window.location.href);
    if (nextSessionId) url.searchParams.set('session_id', nextSessionId);
    else url.searchParams.delete('session_id');
    window.history.replaceState(null, '', url.toString());
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  };

  const buildTimelineFromTrace = (trace: TraceData): TimelineItem[] => {
    const parseTimestampMs = (value: unknown): number | null => {
      if (typeof value !== 'string' || !value) return null;
      const parsed = new Date(value).getTime();
      return Number.isNaN(parsed) ? null : parsed;
    };

    const buildToolCallFromTrace = (record: Record<string, unknown>): ToolCallItem & { traceTimestampMs: number | null } => {
      const rawSummary = record.result_summary;
      let status: ToolCallItem['status'] = 'completed';
      if (typeof rawSummary === 'string') {
        try {
          const parsed = JSON.parse(rawSummary) as { success?: boolean; error?: unknown };
          if (parsed?.success === false || Boolean(parsed?.error)) {
            status = 'error';
          }
        } catch {
          status = 'completed';
        }
      }
      return {
        id: typeof record.tool_call_id === 'string' ? record.tool_call_id : createTimelineId('tool'),
        tool: typeof record.tool_name === 'string' ? record.tool_name : 'tool',
        input: record.args,
        output: normalizeMessageContent(rawSummary),
        status,
        startedAt: typeof record.timestamp === 'string' ? record.timestamp : undefined,
        endedAt: typeof record.timestamp === 'string' ? record.timestamp : undefined,
        collapsed: false,
        traceTimestampMs: parseTimestampMs(record.timestamp),
      };
    };

    const orderedToolCalls = (trace.tool_calls || [])
      .map((item) => buildToolCallFromTrace(item as Record<string, unknown>))
      .sort((left, right) => {
        const leftTime = left.traceTimestampMs ?? Number.MAX_SAFE_INTEGER;
        const rightTime = right.traceTimestampMs ?? Number.MAX_SAFE_INTEGER;
        return leftTime - rightTime;
      });

    const restoredTimeline: TimelineItem[] = [];
    let currentUserTimestampMs: number | null = null;
    let nextToolIndex = 0;

    for (const item of trace.messages || []) {
      if (!item || (item.role !== 'user' && item.role !== 'assistant')) {
        continue;
      }

      const timestamp = item.timestamp || new Date().toISOString();
      const timestampMs = parseTimestampMs(timestamp);

      if (item.role === 'user') {
        currentUserTimestampMs = timestampMs;
        restoredTimeline.push({
          id: createTimelineId('user'),
          kind: 'user',
          content: normalizeMessageContent(item.content),
          timestamp,
        });
        continue;
      }

      const toolCalls: ToolCallItem[] = [];
      while (nextToolIndex < orderedToolCalls.length) {
        const candidate = orderedToolCalls[nextToolIndex];
        const candidateMs = candidate.traceTimestampMs;

        if (candidateMs !== null && currentUserTimestampMs !== null && candidateMs < currentUserTimestampMs) {
          nextToolIndex += 1;
          continue;
        }

        if (candidateMs !== null && timestampMs !== null && candidateMs > timestampMs) {
          break;
        }

        const { traceTimestampMs: _traceTimestampMs, ...toolCall } = candidate;
        toolCalls.push(toolCall);
        nextToolIndex += 1;
      }

      restoredTimeline.push({
        id: createTimelineId('assistant'),
        kind: 'assistant_turn',
        timestamp,
        status: toolCalls.some((toolCall) => toolCall.status === 'error') ? 'error' : 'completed',
        toolCalls,
        toolsCollapsed: false,
        finalReply: normalizeMessageContent(item.content),
        finalReplyTimestamp: timestamp,
      });
      currentUserTimestampMs = null;
    }

    for (let index = restoredTimeline.length - 1; index >= 0; index -= 1) {
      const item = restoredTimeline[index];
      if (item.kind === 'assistant_turn') {
        if (trace.status === 'error') {
          item.status = 'error';
        }
        break;
      }
    }
    const remainingToolCalls = orderedToolCalls.slice(nextToolIndex).map(({ traceTimestampMs: _traceTimestampMs, ...toolCall }) => toolCall);
    if (remainingToolCalls.length > 0) {
      for (let index = restoredTimeline.length - 1; index >= 0; index -= 1) {
        const item = restoredTimeline[index];
        if (item.kind === 'assistant_turn') {
          item.toolCalls = [...item.toolCalls, ...remainingToolCalls];
          item.status = item.status === 'error' || remainingToolCalls.some((toolCall) => toolCall.status === 'error') ? 'error' : item.status;
          return restoredTimeline;
        }
      }

      restoredTimeline.push({
        id: createTimelineId('assistant'),
        kind: 'assistant_turn',
        timestamp: trace.updated_at || new Date().toISOString(),
        status: remainingToolCalls.some((toolCall) => toolCall.status === 'error') || trace.status === 'error' ? 'error' : 'completed',
        toolCalls: remainingToolCalls,
        toolsCollapsed: false,
        finalReply: '',
        finalReplyTimestamp: trace.updated_at || new Date().toISOString(),
      });
    }

    return restoredTimeline;
  };

  const refreshSessionHistory = async (targetAgentId: string) => {
    setLoadingSessionHistory(true);
    setSessionHistoryError(null);
    try {
      const response = await agentApi.getSessions(targetAgentId);
      setSessionHistory(response.sessions || []);
    } catch (error) {
      console.error('[ChatPanel] 加载会话历史失败', error);
      setSessionHistory([]);
      setSessionHistoryError(extractErrorMessage(error, '加载历史对话失败'));
    } finally {
      setLoadingSessionHistory(false);
    }
  };

  const restoreFromTrace = async (storedAgentId: string, storedSessionId: string) => {
    const trace = await agentApi.getTrace(storedAgentId, storedSessionId);
    const restoredTimeline = buildTimelineFromTrace(trace);
    const restoredAgentId = trace.agent_id || storedAgentId;
    const restoredSessionId = trace.session_id || storedSessionId;
    localStorage.setItem(AGENT_STORAGE_KEY, restoredAgentId);
    localStorage.setItem(SESSION_STORAGE_KEY, restoredSessionId);
    updateUrlSession(restoredSessionId);
    setAgentId(restoredAgentId);
    setSessionId(restoredSessionId);
    setTimeline(restoredTimeline);
  };

  const updateDisplayedSuggestions = (allSuggs: QuestionSuggestion[], startIdx: number) => {
    if (allSuggs.length === 0) {
      setDisplayedSuggestions([]);
      return;
    }
    setDisplayedSuggestions(allSuggs.slice(startIdx, startIdx + SUGGESTIONS_PER_PAGE));
  };

  const loadSuggestions = async () => {
    try {
      const response = await agentApi.getSuggestions(currentDate, selectedItem?.type, selectedItem?.id);
      const suggestions = response.suggestions || [];
      setAllSuggestions(suggestions);
      setSuggestionStartIndex(0);
      updateDisplayedSuggestions(suggestions, 0);
    } catch (error) {
      console.error('[ChatPanel] 获取建议失败', error);
    }
  };

  useEffect(() => {
    (window as unknown as { __chatPanelDebug?: unknown }).__chatPanelDebug = {
      sessionId,
      agentId,
      itemsCount: timeline.length,
      loading,
      restoring,
      agentMode,
      getSessionId: () => sessionId,
      getAgentId: () => agentId,
      getMessages: () => timeline,
    };
  }, [sessionId, agentId, timeline, loading, restoring, agentMode]);

  useEffect(() => {
    scrollToBottom();
  }, [timeline]);

  useEffect(() => {
    if (sessionId) localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    else localStorage.removeItem(SESSION_STORAGE_KEY);
    localStorage.setItem(AGENT_STORAGE_KEY, agentId);
    onSessionChange?.(sessionId);
    onAgentChange?.(agentId);
  }, [sessionId, agentId, onSessionChange, onAgentChange]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(SKILL_STORAGE_KEY, enableSkills ? '1' : '0');
    }
  }, [enableSkills]);

  useEffect(() => {
    adjustTextareaHeight();
  }, [input]);

  useEffect(() => {
    const init = async () => {
      try {
        const health = await agentApi.healthCheck();
        setAgentMode(health.mode);
      } catch (error) {
        console.error('[ChatPanel] 初始化 health 失败', error);
      }

      try {
        const url = new URL(window.location.href);
        const urlSessionId = url.searchParams.get('session_id');
        const storedSessionId = urlSessionId || localStorage.getItem(SESSION_STORAGE_KEY);
        const storedAgentId = localStorage.getItem(AGENT_STORAGE_KEY) || DEFAULT_AGENT_ID;
        if (storedSessionId) {
          setRestoring(true);
          try {
            await restoreFromTrace(storedAgentId, storedSessionId);
          } catch (error) {
            console.error('[ChatPanel] 恢复 trace 失败，将回到新对话状态', error);
            setSessionId(null);
            setTimeline([]);
            localStorage.removeItem(SESSION_STORAGE_KEY);
            updateUrlSession(null);
          } finally {
            setRestoring(false);
          }
        }
      } catch (error) {
        console.error('[ChatPanel] 初始化恢复失败', error);
        setRestoring(false);
      }
    };

    void init();
  }, []);

  useEffect(() => {
    void refreshSessionHistory(agentId);
  }, [agentId]);

  useEffect(() => {
    void loadSuggestions();
  }, [currentDate, selectedItem]);

  const handleRefreshSuggestions = () => {
    if (allSuggestions.length <= SUGGESTIONS_PER_PAGE) return;
    const nextIndex = suggestionStartIndex + SUGGESTIONS_PER_PAGE;
    const newIndex = nextIndex >= allSuggestions.length ? 0 : nextIndex;
    setSuggestionStartIndex(newIndex);
    updateDisplayedSuggestions(allSuggestions, newIndex);
  };

  const clearChat = async () => {
    if (loading || creatingAgent) return;
    setTimeline([]);
    setInput('');
    setSessionId(null);
    localStorage.removeItem(SESSION_STORAGE_KEY);
    updateUrlSession(null);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    onSessionChange?.(null);
    await loadSuggestions();
  };

  const createAgent = async () => {
    if (loading || restoring || creatingAgent) return;
    setCreatingAgent(true);
    setCreateAgentError(null);
    try {
      const rawAgentId = newAgentIdInput.trim();
      const response: CreateAgentResponse = await agentApi.createAgent(rawAgentId ? { agent_id: rawAgentId } : undefined);
      setAgentId(response.agent_id);
      setSessionId(null);
      setTimeline([]);
      setInput('');
      setNewAgentIdInput('');
      setSessionHistory([]);
      localStorage.removeItem(SESSION_STORAGE_KEY);
      updateUrlSession(null);
      localStorage.setItem(AGENT_STORAGE_KEY, response.agent_id);
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
      await refreshSessionHistory(response.agent_id);
    } catch (error) {
      console.error('[ChatPanel] 创建 agent 失败', error);
      const errorMessage = extractErrorMessage(error, '新建 Agent 失败，请稍后重试');
      setCreateAgentError(errorMessage);
      setTimeline((prev) => [
        ...prev,
        {
          id: createTimelineId('assistant'),
          kind: 'assistant_turn',
          timestamp: new Date().toISOString(),
          status: 'error',
          toolCalls: [],
          toolsCollapsed: false,
          finalReply: `新建 Agent 失败：${errorMessage}`,
          finalReplyTimestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setCreatingAgent(false);
    }
  };

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim() || restoring || creatingAgent) return;

    const nextSessionId = sessionId || createSessionId();
    localStorage.setItem(AGENT_STORAGE_KEY, agentId);
    localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    updateUrlSession(nextSessionId);
    setSessionId(nextSessionId);

    const now = new Date().toISOString();
    const userMessage: UserTimelineItem = {
      id: createTimelineId('user'),
      kind: 'user',
      content: messageText,
      timestamp: now,
    };
    const assistantTurnId = createTimelineId('assistant');
    const assistantTurn: AssistantTurnItem = {
      id: assistantTurnId,
      kind: 'assistant_turn',
      timestamp: now,
      status: 'streaming',
      toolCalls: [],
      toolsCollapsed: false,
      finalReply: '',
      finalReplyTimestamp: now,
    };

    setTimeline((prev) => [...prev, userMessage, assistantTurn]);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setLoading(true);

    const updateAssistantTurn = (updater: (current: AssistantTurnItem) => AssistantTurnItem) => {
      setTimeline((prev) =>
        prev.map((item) => (item.kind === 'assistant_turn' && item.id === assistantTurnId ? updater(item) : item)),
      );
    };

    try {
      const request: AgentChatRequest = {
        agent_id: agentId,
        session_id: nextSessionId,
        message: messageText,
        ui_context: { date: currentDate, selected: selectedItem },
        options: { mode: 'auto', max_tool_calls: 100, enable_skills: enableSkills },
      };
      const response = await agentApi.streamMessage(request, {
        onEvent: ({ event, data }) => {
          const payload = (data ?? {}) as Record<string, unknown>;

          if (event === 'session_created') {
            const incomingAgentId = typeof payload.agent_id === 'string' ? payload.agent_id : agentId;
            const incomingSessionId = typeof payload.session_id === 'string' ? payload.session_id : null;
            if (incomingAgentId) {
              localStorage.setItem(AGENT_STORAGE_KEY, incomingAgentId);
              setAgentId(incomingAgentId);
            }
            if (incomingSessionId) {
              localStorage.setItem(SESSION_STORAGE_KEY, incomingSessionId);
              updateUrlSession(incomingSessionId);
              setSessionId(incomingSessionId);
            }
            void refreshSessionHistory(incomingAgentId);
            return;
          }

          if (event === 'assistant_message') {
            const nextChunk = normalizeMessageContent(payload.content).trim();
            if (!nextChunk) return;
            updateAssistantTurn((current) => ({
              ...current,
              finalReply: current.finalReply ? `${current.finalReply}\n\n${nextChunk}` : nextChunk,
              finalReplyTimestamp: typeof payload.timestamp === 'string' ? payload.timestamp : current.finalReplyTimestamp,
              status: payload.final ? 'completed' : current.status,
            }));
            return;
          }

          if (event === 'tool_start') {
            updateAssistantTurn((current) => {
              const toolCallId = typeof payload.tool_call_id === 'string' ? payload.tool_call_id : createTimelineId('tool');
              const existingIndex = current.toolCalls.findIndex((toolCall) => toolCall.id === toolCallId);
              const nextToolCall: ToolCallItem = {
                id: toolCallId,
                tool: typeof payload.tool === 'string' ? payload.tool : 'tool',
                input: payload.input,
                output: existingIndex >= 0 ? current.toolCalls[existingIndex].output : '',
                status: 'running',
                startedAt: typeof payload.timestamp === 'string' ? payload.timestamp : undefined,
                endedAt: existingIndex >= 0 ? current.toolCalls[existingIndex].endedAt : undefined,
                collapsed: false,
              };

              if (existingIndex >= 0) {
                const nextToolCalls = [...current.toolCalls];
                nextToolCalls[existingIndex] = {
                  ...nextToolCalls[existingIndex],
                  ...nextToolCall,
                  collapsed: nextToolCalls[existingIndex].collapsed,
                };
                return { ...current, toolCalls: nextToolCalls, toolsCollapsed: false };
              }

              return {
                ...current,
                toolCalls: [...current.toolCalls, nextToolCall],
                toolsCollapsed: false,
              };
            });
            return;
          }

          if (event === 'tool_end') {
            const toolCallId = typeof payload.tool_call_id === 'string' ? payload.tool_call_id : '';
            updateAssistantTurn((current) => ({
              ...current,
              toolCalls: current.toolCalls.map((toolCall) =>
                toolCall.id === toolCallId
                  ? {
                      ...toolCall,
                      tool: typeof payload.tool === 'string' ? payload.tool : toolCall.tool,
                      output: normalizeMessageContent(payload.output),
                      status: payload.success ? 'completed' : 'error',
                      endedAt: typeof payload.timestamp === 'string' ? payload.timestamp : toolCall.endedAt,
                    }
                  : toolCall,
              ),
            }));
            return;
          }

          if (event === 'error') {
            updateAssistantTurn((current) => ({
              ...current,
              status: 'error',
              finalReply: `流式执行出错：${normalizeMessageContent(payload.message)}`,
              finalReplyTimestamp: typeof payload.timestamp === 'string' ? payload.timestamp : current.finalReplyTimestamp,
            }));
          }
        },
      });

      setAgentId(response.agent_id || DEFAULT_AGENT_ID);
      updateUrlSession(response.session_id);
      setSessionId(response.session_id);
      updateAssistantTurn((current) => ({
        ...current,
        status: 'completed',
        finalReply: response.message_markdown || current.finalReply,
        finalReplyTimestamp: response.timestamp,
      }));
      await refreshSessionHistory(response.agent_id || DEFAULT_AGENT_ID);
      onAgentResponse?.(response);
    } catch (error) {
      console.error('[ChatPanel] 发送消息失败', error);
      updateAssistantTurn((current) => ({
        ...current,
        status: 'error',
        finalReply: '发送消息时出现错误，请稍后重试。',
        finalReplyTimestamp: new Date().toISOString(),
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleSelectHistorySession = async (historyItem: SessionHistoryItem) => {
    if (loading || restoring || creatingAgent) return;
    setRestoring(true);
    setCreateAgentError(null);
    try {
      await restoreFromTrace(agentId, historyItem.session_id);
    } catch (error) {
      console.error('[ChatPanel] 切换历史会话失败', error);
      setCreateAgentError(extractErrorMessage(error, '切换历史会话失败'));
    } finally {
      setRestoring(false);
    }
  };

  const toggleToolsPanel = (assistantTurnId: string) => {
    setTimeline((prev) =>
      prev.map((item) =>
        item.kind === 'assistant_turn' && item.id === assistantTurnId
          ? { ...item, toolsCollapsed: !item.toolsCollapsed }
          : item,
      ),
    );
  };

  const toggleToolCall = (assistantTurnId: string, toolCallId: string) => {
    setTimeline((prev) =>
      prev.map((item) => {
        if (item.kind !== 'assistant_turn' || item.id !== assistantTurnId) return item;
        return {
          ...item,
          toolCalls: item.toolCalls.map((toolCall) =>
            toolCall.id === toolCallId ? { ...toolCall, collapsed: !toolCall.collapsed } : toolCall,
          ),
        };
      }),
    );
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendMessage(input);
    }
  };

  const handleSuggestionClick = (question: string) => {
    setInput(question);
    setTimeout(() => {
      textareaRef.current?.focus();
      adjustTextareaHeight();
    }, 0);
  };

  const renderToolCall = (assistantTurnId: string, toolCall: ToolCallItem) => (
    <div key={toolCall.id} className={`tool-call-panel status-${toolCall.status}`} data-testid="chat-tool-call">
      <button
        type="button"
        className="tool-call-panel-header"
        onClick={() => toggleToolCall(assistantTurnId, toolCall.id)}
        data-testid="chat-tool-call-toggle"
      >
        <span className={`panel-toggle-icon ${toolCall.collapsed ? '' : 'expanded'}`}>▶</span>
        <span className="tool-call-panel-title">{toolCall.tool}</span>
        <span className={`tool-call-status status-${toolCall.status}`}>{getToolStatusLabel(toolCall.status)}</span>
      </button>
      {!toolCall.collapsed && (
        <div className="tool-call-panel-body">
          {toolCall.input !== undefined ? (
            <div className="tool-call-block">
              <div className="tool-call-block-label">Input</div>
              <pre className="tool-call-block-content">{normalizeMessageContent(toolCall.input)}</pre>
            </div>
          ) : null}
          {toolCall.output ? (
            <div className="tool-call-block" data-testid="chat-tool-call-result">
              <div className="tool-call-block-label">Result</div>
              <pre className="tool-call-block-content">{toolCall.output}</pre>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );

  const renderAssistantTurn = (item: AssistantTurnItem) => {
    const hasReply = item.finalReply.trim().length > 0;
    const shouldShowReplyPanel = item.toolCalls.length === 0 || hasReply || item.status !== 'completed';

    return (
      <div key={item.id} className="assistant-turn" data-testid="chat-assistant-turn">
        {item.toolCalls.length > 0 && (
          <div className="assistant-turn-panel assistant-turn-tools-panel" data-testid="chat-tool-panel">
            <button
              type="button"
              className="assistant-turn-panel-header"
              onClick={() => toggleToolsPanel(item.id)}
              data-testid="chat-tool-panel-toggle"
            >
              <span className={`panel-toggle-icon ${item.toolsCollapsed ? '' : 'expanded'}`}>▶</span>
              <span className="assistant-turn-panel-title">工具调用</span>
              <span className="assistant-turn-panel-meta">{item.toolCalls.length} 次</span>
            </button>
            {!item.toolsCollapsed && (
              <div className="assistant-turn-panel-body">
                {item.toolCalls.map((toolCall) => renderToolCall(item.id, toolCall))}
              </div>
            )}
          </div>
        )}

        {shouldShowReplyPanel && (
          <div className="assistant-turn-panel assistant-turn-reply-panel" data-testid="chat-assistant-reply">
            <div className="assistant-turn-reply-header">最终回复</div>
            <div className="assistant-turn-reply-body">
              {hasReply ? (
                <ReactMarkdown className="message-markdown">{item.finalReply}</ReactMarkdown>
              ) : item.status === 'streaming' ? (
                <div className="loading-dots"><span>.</span><span>.</span><span>.</span></div>
              ) : (
                <div className="message-text">暂无内容</div>
              )}
            </div>
            <div className="message-time">{formatTime(item.finalReplyTimestamp || item.timestamp)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderTimelineItem = (item: TimelineItem) => {
    if (item.kind === 'user') {
      return (
        <div key={item.id} className="message message-user" data-testid="chat-user-message">
          <div className="message-content">{item.content}</div>
          <div className="message-time">{formatTime(item.timestamp)}</div>
        </div>
      );
    }
    return renderAssistantTurn(item);
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-left">
          <h3>智能助手</h3>
          <span className={`status-badge status-${agentMode}`}>{agentMode === 'production' ? 'PROD' : 'MOCK'}</span>
          <span className="status-badge">agent:{agentId}</span>
        </div>
        <div className="chat-header-right">
          <div className="chat-create-agent-group">
            <input
              className="chat-create-agent-input"
              type="text"
              placeholder="可选：自定义 agent_id"
              value={newAgentIdInput}
              onChange={(e) => setNewAgentIdInput(e.target.value)}
              disabled={loading || restoring || creatingAgent}
            />
            <button type="button" className="chat-create-agent-button" onClick={() => void createAgent()} disabled={loading || restoring || creatingAgent} title="不填则随机生成，填写则使用指定 agent_id">
              <span>{creatingAgent ? '创建中...' : '新建 Agent'}</span>
            </button>
          </div>
          <div className={`skill-toggle ${enableSkills ? 'skill-enabled' : ''}`} title="切换是否允许 Agent 调用技能工具">
            <label className="skill-toggle-switch">
              <input type="checkbox" checked={enableSkills} onChange={(e) => setEnableSkills(e.target.checked)} disabled={loading || restoring || creatingAgent} />
              <span className="skill-toggle-slider" />
            </label>
            <div className="skill-toggle-text">
              <span>Skill 模式</span>
              <small>{enableSkills ? '已开启' : '已关闭'}</small>
            </div>
          </div>
          {currentDate && <span className="current-date">{currentDate}</span>}
          <button type="button" className="chat-clear-button" onClick={() => void clearChat()} disabled={loading || restoring || creatingAgent} title="开始新对话">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            <span>新对话</span>
          </button>
        </div>
      </div>

      {createAgentError && <div className="chat-create-agent-error">{createAgentError}</div>}

      <div className="chat-body">
        <div className="chat-main">
          <div className="chat-messages">
            {timeline.length === 0 && !restoring && (
              <div className="welcome-message">
                <h4>你好</h4>
                <p>这里会保留当前 Agent 的历史对话，你可以随时切换回来继续问。</p>
                <p>右侧列表使用每个会话的首条用户提问作为标题。</p>
              </div>
            )}
            {restoring && <div className="welcome-message"><p>正在恢复历史对话...</p></div>}
            {timeline.map((item) => renderTimelineItem(item))}
            <div ref={messagesEndRef} />
          </div>

          {!restoring && displayedSuggestions.length > 0 && (
            <div className={`chat-suggestions ${suggestionsVisible ? '' : 'collapsed'}`}>
              <div className="suggestions-header">
                <button type="button" className="suggestions-toggle-btn" onClick={() => setSuggestionsVisible(!suggestionsVisible)} title={suggestionsVisible ? '收起推荐问题' : '展开推荐问题'}>
                  <span className={`toggle-icon ${suggestionsVisible ? 'expanded' : ''}`}>▶</span>
                  <span className="suggestions-label">推荐问题</span>
                </button>
                {suggestionsVisible && allSuggestions.length > SUGGESTIONS_PER_PAGE && (
                  <button type="button" className="suggestions-refresh-btn" onClick={handleRefreshSuggestions} title="换一批">换一批</button>
                )}
              </div>
              {suggestionsVisible && (
                <div className="suggestions-list">
                  {displayedSuggestions.map((suggestion) => (
                    <button type="button" key={suggestion.id} className="suggestion-button" onClick={() => handleSuggestionClick(suggestion.question)} title={suggestion.category}>
                      {suggestion.question}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="chat-input-container">
            <textarea
              ref={textareaRef}
              className="chat-input"
              placeholder="输入问题（Enter 发送，Shift+Enter 换行）..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={loading || restoring || creatingAgent}
              rows={1}
            />
            <button type="button" className="chat-send-button" onClick={() => void sendMessage(input)} disabled={loading || restoring || creatingAgent || !input.trim()}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
          </div>
        </div>

        <aside className="chat-history-panel">
          <div className="chat-history-header">
            <div>
              <h4>历史会话</h4>
              <p>{loadingSessionHistory ? '正在加载...' : `${sessionHistory.length} 条记录`}</p>
            </div>
            <button type="button" className="chat-history-refresh" onClick={() => void refreshSessionHistory(agentId)} disabled={loading || restoring || creatingAgent || loadingSessionHistory}>
              刷新
            </button>
          </div>

          <div className="chat-history-list">
            {sessionHistoryError && <div className="chat-history-empty error">{sessionHistoryError}</div>}
            {!sessionHistoryError && loadingSessionHistory && sessionHistory.length === 0 && <div className="chat-history-empty">正在读取历史对话...</div>}
            {!sessionHistoryError && !loadingSessionHistory && sessionHistory.length === 0 && <div className="chat-history-empty">当前 Agent 暂无历史对话</div>}
            {sessionHistory.map((historyItem) => {
              const active = historyItem.session_id === sessionId;
              return (
                <button
                  type="button"
                  key={historyItem.session_id}
                  className={`chat-history-item ${active ? 'active' : ''}`}
                  onClick={() => void handleSelectHistorySession(historyItem)}
                  disabled={loading || restoring || creatingAgent}
                >
                  <div className="chat-history-item-title-row">
                    <span className="chat-history-item-title">{getSessionDisplayTitle(historyItem)}</span>
                    {active && <span className="chat-history-item-badge">当前</span>}
                  </div>
                  <div className="chat-history-item-meta">{formatTime(historyItem.updated_at || historyItem.created_at)}</div>
                  <div className="chat-history-item-meta">messages: {historyItem.messages_count}</div>
                  <div className="chat-history-item-id">session: {historyItem.session_id}</div>
                </button>
              );
            })}
          </div>
        </aside>
      </div>
    </div>
  );
};



