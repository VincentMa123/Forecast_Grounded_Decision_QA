import axios from 'axios';

const API_BASE_URL = '/api/agent';

export const DEFAULT_AGENT_ID = 'default';
export const SESSION_STORAGE_KEY = 'agent_session_id';
export const AGENT_STORAGE_KEY = 'agent_id';
export const SKILL_STORAGE_KEY = 'agent_enable_skills';

export interface UIContext {
  date?: string;
  selected?: {
    type: 'system' | 'node' | 'edge';
    id: string;
  };
  viewport?: {
    center: [number, number];
    zoom: number;
  };
}

export interface ChatOptions {
  mode?: 'analysis' | 'dispatch' | 'auto';
  max_tool_calls?: number;
  enable_streaming?: boolean;
  enable_skills?: boolean;
}

export interface TraceSummary {
  trace_path: string;
  status: string;
  updated_at?: string | null;
  messages_count: number;
  tool_calls_count: number;
  artifacts_count: number;
}

export interface SessionHistoryItem {
  session_id: string;
  created_at?: string | null;
  updated_at?: string | null;
  status: string;
  first_user_message: string;
  messages_count: number;
}

export interface SessionHistoryResponse {
  agent_id: string;
  sessions: SessionHistoryItem[];
  total_count: number;
}

export interface AgentChatRequest {
  agent_id?: string;
  session_id?: string;
  message: string;
  ui_context?: UIContext;
  options?: ChatOptions;
}

export interface AgentChatResponse {
  agent_id: string;
  session_id: string;
  message_markdown: string;
  trace_summary: TraceSummary;
  memory_updates?: string[];
  generated_artifacts?: string[];
  timestamp: string;
}

export interface CreateAgentRequest {
  agent_id?: string;
}

export interface CreateAgentResponse {
  agent_id: string;
  workspace_root: string;
  memory_root: string;
  assets_root: string;
  trace_root: string;
  temporary_dir: string;
  reports_dir: string;
  plan_path: string;
  timestamp: string;
}

export interface TraceMessage {
  role: 'user' | 'assistant' | string;
  content: string;
  timestamp: string;
}

export interface TraceData {
  session_id: string;
  agent_id: string;
  created_at: string;
  updated_at: string;
  status: string;
  messages: TraceMessage[];
  tool_calls: Array<Record<string, unknown>>;
  context_injection: Record<string, unknown>;
  decision_log: Array<Record<string, unknown>>;
  artifacts: string[];
  memory_commits: string[];
}

export interface MemorySummaryResponse {
  agent_id: string;
  workspace_root: string;
  timeline_files: string[];
  latest_timeline_path?: string | null;
  latest_timeline_excerpt: string;
  latest_trace_summary?: TraceSummary | null;
}

export interface WorkspaceTreeNode {
  name: string;
  path: string;
  node_type: 'file' | 'directory';
  size_bytes?: number | null;
  children: WorkspaceTreeNode[];
}

export interface WorkspaceTreeResponse {
  agent_id: string;
  workspace_root: string;
  requested_path: string;
  total_files: number;
  total_directories: number;
  tree: WorkspaceTreeNode;
}

export interface WorkspaceFileResponse {
  agent_id: string;
  workspace_root: string;
  path: string;
  size_bytes: number;
  is_text: boolean;
  truncated: boolean;
  content: string;
}

export interface QuestionSuggestion {
  id: string;
  category: string;
  question: string;
  intent: string;
}

export interface SuggestionsResponse {
  suggestions: QuestionSuggestion[];
  total: number;
}

export interface HealthCheckResponse {
  status: string;
  mode: 'mock' | 'production';
  config: {
    has_api_key: boolean;
    api_key_prefix: string | null;
    has_custom_base: boolean;
    api_base: string;
    model: string;
  };
  data: {
    available_dates: number;
    date_range: {
      start: string | null;
      end: string | null;
    };
  };
}

export interface AgentStreamEvent {
  event: string;
  data: unknown;
}

export interface AgentStreamHandlers {
  onEvent?: (event: AgentStreamEvent) => void;
}

export const agentApi = {
  async createAgent(request?: CreateAgentRequest): Promise<CreateAgentResponse> {
    const response = await axios.post(`${API_BASE_URL}/create`, request || {});
    return response.data;
  },

  async streamMessage(request: AgentChatRequest, handlers?: AgentStreamHandlers): Promise<AgentChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok || !response.body) {
      const errorText = await response.text();
      throw new Error(errorText || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let donePayload: AgentChatResponse | null = null;

    const processBlock = (block: string) => {
      const lines = block.split('\n');
      let eventName = 'message';
      const dataLines: string[] = [];
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }
      if (!dataLines.length) return;
      const data = JSON.parse(dataLines.join('\n'));
      handlers?.onEvent?.({ event: eventName, data });
      if (eventName === 'done') {
        donePayload = data as AgentChatResponse;
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let boundaryIndex = buffer.indexOf('\n\n');
      while (boundaryIndex !== -1) {
        const block = buffer.slice(0, boundaryIndex).trim();
        buffer = buffer.slice(boundaryIndex + 2);
        if (block) processBlock(block);
        boundaryIndex = buffer.indexOf('\n\n');
      }
      if (done) break;
    }

    if (buffer.trim()) {
      processBlock(buffer.trim());
    }

    if (!donePayload) {
      throw new Error('流式响应未返回 done 事件');
    }
    return donePayload;
  },

  async getSessions(agentId: string): Promise<SessionHistoryResponse> {
    const response = await axios.get(`${API_BASE_URL}/sessions/${agentId}`);
    return response.data;
  },

  async getTrace(agentId: string, sessionId: string): Promise<TraceData> {
    const response = await axios.get(`${API_BASE_URL}/trace/${agentId}/${sessionId}`);
    return response.data;
  },

  async getMemorySummary(agentId: string, sessionId?: string): Promise<MemorySummaryResponse> {
    const response = await axios.get(`${API_BASE_URL}/memory/${agentId}/summary`, {
      params: { session_id: sessionId },
    });
    return response.data;
  },

  async getWorkspaceTree(agentId: string, path = '.', maxDepth = 4): Promise<WorkspaceTreeResponse> {
    const response = await axios.get(`${API_BASE_URL}/workspace/${agentId}/tree`, {
      params: {
        path,
        max_depth: maxDepth,
      },
    });
    return response.data;
  },

  async getWorkspaceFile(agentId: string, path: string): Promise<WorkspaceFileResponse> {
    const response = await axios.get(`${API_BASE_URL}/workspace/${agentId}/file`, {
      params: { path },
    });
    return response.data;
  },

  async getSuggestions(date?: string, selectedType?: string, selectedId?: string): Promise<SuggestionsResponse> {
    const response = await axios.get(`${API_BASE_URL}/suggestions`, {
      params: {
        date,
        selected_type: selectedType,
        selected_id: selectedId,
      },
    });
    return response.data;
  },

  async getTemplates(): Promise<unknown> {
    const response = await axios.get(`${API_BASE_URL}/templates`);
    return response.data;
  },

  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  },
};
