import { useQuery } from '@tanstack/react-query';
import { api } from './client';

export const useNodeFlow = (date: string) => {
  return useQuery({
    queryKey: ['nodeFlow', date],
    queryFn: () => api.getNodeFlow(date),
    enabled: !!date,
  });
};

export const usePipelineFlow = (date: string) => {
  return useQuery({
    queryKey: ['pipelineFlow', date],
    queryFn: () => api.getPipelineFlow(date),
    enabled: !!date,
  });
};

export const useConsumerFlow = (date: string) => {
  return useQuery({
    queryKey: ['consumerFlow', date],
    queryFn: () => api.getConsumerFlow(date),
    enabled: !!date,
  });
};

export const useConsumersByNode = (stationName: string, date: string) => {
  return useQuery({
    queryKey: ['consumersByNode', stationName, date],
    queryFn: () => api.getConsumersByNode(stationName, date),
    enabled: !!stationName && !!date,
  });
};

export const useAvailableDates = (dataType: 'node_flow' | 'pipeline_flow' | 'consumer_flow') => {
  return useQuery({
    queryKey: ['availableDates', dataType],
    queryFn: () => api.getAvailableDates(dataType),
  });
};
