import axios from 'axios';

const API_BASE_URL = '/api';

// Flow data types
export interface NodeFlowRecord {
  pipeline_division: string;
  station_name: string;
  lon: number;
  lat: number;
  node_type: string;
  control_type: string;
  input_flow: number | null;
  calculated_flow: number;
}

export interface PipelineFlowRecord {
  start_station: string;
  end_station: string;
  pipeline_type: string;
  pipeline_division: string;
  pipeline_flow: number;
}

export interface ConsumerFlowRecord {
  pipeline: string;
  location: string;
  supply_point: string;
  station_name: string;
  consumer: string;
  consumption: number;
}

export interface FlowDataResponse<T> {
  date: string;
  records: T[];
  total_records: number;
}

export interface AvailableDatesResponse {
  data_type: string;
  dates: string[];
  total_count: number;
  date_range: {
    start: string;
    end: string;
  };
}

export const api = {
  async getNodeFlow(date: string): Promise<FlowDataResponse<NodeFlowRecord>> {
    const response = await axios.get(`${API_BASE_URL}/flow/nodes`, {
      params: { query_date: date }
    });
    return response.data;
  },

  async getPipelineFlow(date: string): Promise<FlowDataResponse<PipelineFlowRecord>> {
    const response = await axios.get(`${API_BASE_URL}/flow/pipelines`, {
      params: { query_date: date }
    });
    return response.data;
  },

  async getConsumerFlow(date: string): Promise<FlowDataResponse<ConsumerFlowRecord>> {
    const response = await axios.get(`${API_BASE_URL}/flow/consumers`, {
      params: { query_date: date }
    });
    return response.data;
  },

  async getConsumersByNode(stationName: string, date: string): Promise<FlowDataResponse<ConsumerFlowRecord>> {
    const response = await axios.get(`${API_BASE_URL}/flow/consumers/by-node`, {
      params: { station_name: stationName, query_date: date }
    });
    return response.data;
  },

  async getAvailableDates(dataType: 'node_flow' | 'pipeline_flow' | 'consumer_flow'): Promise<AvailableDatesResponse> {
    const response = await axios.get(`${API_BASE_URL}/dates`, {
      params: { data_type: dataType }
    });
    return response.data;
  }
};
