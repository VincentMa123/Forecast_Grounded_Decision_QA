import { useEffect, useRef, useState, useMemo } from 'react';
import maplibregl, { Map as MapLibreMap, MapLayerMouseEvent } from 'maplibre-gl';
import { useNavigate } from 'react-router-dom';
import 'maplibre-gl/dist/maplibre-gl.css';
import './MapView.css';
import { useNodeFlow, usePipelineFlow, useAvailableDates } from '../api/hooks';
import { DatePicker } from './DatePicker';
import { Legend } from './Legend';
import { RightDock } from './RightDock';
import { PIPELINE_COLORS } from '../utils/colorMapping';

interface TooltipData {
  x: number;
  y: number;
  content: {
    title: string;
    items: { label: string; value: string }[];
  } | null;
}

const MIN_CHAT_WIDTH = 320;
const DEFAULT_CHAT_WIDTH = 560;
const MIN_MAP_WIDTH = 360;

function getMaxChatWidth(): number {
  if (typeof window === 'undefined') return 1600;
  return Math.max(MIN_CHAT_WIDTH, window.innerWidth - MIN_MAP_WIDTH);
}

function getInitialChatWidth(): number {
  return Math.min(DEFAULT_CHAT_WIDTH, getMaxChatWidth());
}

export default function MapView() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const navigate = useNavigate();

  const [tooltip, setTooltip] = useState<TooltipData>({ x: 0, y: 0, content: null });
  const [selectedDate, setSelectedDate] = useState('2019-01-01');
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredPipelineDivision, setHoveredPipelineDivision] = useState<string | null>(null);
  const [showChat, setShowChat] = useState(true);
  const [chatWidth, setChatWidth] = useState(getInitialChatWidth);
  const [isResizing, setIsResizing] = useState(false);
  const [mapReady, setMapReady] = useState(false);
  const resizeRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Fetch data
  const { data: availableDatesData } = useAvailableDates('node_flow');
  const { data: nodeFlowData } = useNodeFlow(selectedDate);
  const { data: pipelineFlowData } = usePipelineFlow(selectedDate);

  const availableDates = availableDatesData?.dates || ['2019-01-01'];

  // Debug: Check pipeline_division values vs color mapping
  useEffect(() => {
    if (!nodeFlowData) return;

    console.log('=== nodeFlowData Debug ===');
    console.log('Total records:', nodeFlowData.records.length);
    console.log('First 3 records:', nodeFlowData.records.slice(0, 3));

    const uniq = Array.from(new Set(nodeFlowData.records.map(r => String((r as any).pipeline_division))));
    console.log('node uniq pipeline_division:', uniq);
    console.log('PIPELINE_COLORS keys:', Object.keys(PIPELINE_COLORS));

    // Print all unique values for each field
    const allFields = nodeFlowData.records[0] ? Object.keys(nodeFlowData.records[0]) : [];
    console.log('Available fields:', allFields);

    allFields.forEach(field => {
      const uniqueValues = Array.from(new Set(nodeFlowData.records.map(r => (r as any)[field])));
      if (uniqueValues.length <= 20) {
        console.log(`Unique ${field}:`, uniqueValues);
      } else {
        console.log(`Unique ${field}: (${uniqueValues.length} values, showing first 10)`, uniqueValues.slice(0, 10));
      }
    });
  }, [nodeFlowData]);

  // Convert flow data to GeoJSON
  const nodesGeoJSON = useMemo(() => {
    if (!nodeFlowData) return null;

    return {
      type: 'FeatureCollection' as const,
      features: nodeFlowData.records.map((record, idx) => ({
        type: 'Feature' as const,
        id: idx,
        properties: {
          station_name: record.station_name,
          pipeline_division: record.pipeline_division,
          node_type: record.node_type,
          control_type: record.control_type,
          input_flow: record.input_flow ?? 0,
          calculated_flow: record.calculated_flow ?? 0,
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [record.lon, record.lat],
        },
      })),
    };
  }, [nodeFlowData]);

  const pipelinesGeoJSON = useMemo(() => {
    if (!pipelineFlowData || !nodeFlowData) return null;

    // Create node lookup map using JavaScript's native Map
    const nodeMap = new globalThis.Map<string, { lon: number; lat: number }>(
      nodeFlowData.records.map(node => [node.station_name, { lon: node.lon, lat: node.lat }])
    );

    return {
      type: 'FeatureCollection' as const,
      features: pipelineFlowData.records
        .filter(record => {
          const start = nodeMap.get(record.start_station);
          const end = nodeMap.get(record.end_station);
          return start && end;
        })
        .map((record, idx) => {
          const start = nodeMap.get(record.start_station)!;
          const end = nodeMap.get(record.end_station)!;

          return {
            type: 'Feature' as const,
            id: idx,
            properties: {
              start_station: record.start_station,
              end_station: record.end_station,
              pipeline_type: record.pipeline_type,
              pipeline_division: record.pipeline_division,
              pipeline_flow: record.pipeline_flow ?? 0,
            },
            geometry: {
              type: 'LineString' as const,
              coordinates: [
                [start.lon, start.lat],
                [end.lon, end.lat],
              ],
            },
          };
        }),
    };
  }, [pipelineFlowData, nodeFlowData]);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current) return;
    if (map.current) return; // Prevent double initialization

    try {
      const mapInstance = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            osm: {
              type: 'raster',
              tiles: [
                'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
                'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
              ],
              tileSize: 256,
              attribution: '© CartoDB, © OpenStreetMap contributors',
            },
          },
          layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
        },
        center: [104.0, 35.0],
        zoom: 3.5,
      });

      map.current = mapInstance;

      const handleMapLoad = () => {
        setMapReady(true);
      };

      if (mapInstance.isStyleLoaded()) {
        setMapReady(true);
      } else {
        mapInstance.once('load', handleMapLoad);
      }

      mapInstance.addControl(new maplibregl.NavigationControl(), 'top-right');
      mapInstance.addControl(new maplibregl.ScaleControl({}), 'bottom-right');
    } catch (error) {
      console.error('Map initialization error:', error);
    }

    return () => {
      setMapReady(false);
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Update pipeline layer
  useEffect(() => {
    if (!map.current || !mapReady || !pipelinesGeoJSON) return;

    const updatePipelines = () => {
      if (!map.current) return;
      const mapInstance = map.current;

      if (mapInstance.getSource('pipelines')) {
        (mapInstance.getSource('pipelines') as maplibregl.GeoJSONSource).setData(pipelinesGeoJSON);
      } else {
        mapInstance.addSource('pipelines', {
          type: 'geojson',
          data: pipelinesGeoJSON,
        });

        mapInstance.addLayer({
          id: 'pipelines-line',
          type: 'line',
          source: 'pipelines',
          layout: {
            'line-join': 'round',
            'line-cap': 'round',
          },
          paint: {
            'line-color': [
              'match',
              ['get', 'pipeline_division'],
              ...Object.entries(PIPELINE_COLORS).flat(),
              '#999999'
            ] as any,
            'line-width': [
              'interpolate',
              ['linear'],
              ['zoom'],
              3,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'pipeline_flow']],
                0, 0.5,
                5000, 2.5
              ],
              6,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'pipeline_flow']],
                0, 1.25,
                5000, 5
              ],
              9,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'pipeline_flow']],
                0, 2,
                5000, 8
              ]
            ],
            'line-opacity': 0.9,
          },
        });

        // Mouse move handler
        mapInstance.on('mousemove', 'pipelines-line', (e: MapLayerMouseEvent) => {
          if (!map.current || !e.features || e.features.length === 0) return;

          map.current.getCanvas().style.cursor = 'pointer';

          const feature = e.features[0];
          const props = feature.properties as any;
          setHoveredPipelineDivision(props.pipeline_division || null);

          setTooltip({
            x: e.point.x,
            y: e.point.y,
            content: {
              title: `${props.start_station} → ${props.end_station}`,
              items: [
                { label: '管道划分', value: props.pipeline_division || '-' },
                { label: '管道流量', value: `${props.pipeline_flow?.toFixed(2) || '-'} 万方/天` },
                { label: '类型', value: props.pipeline_type || '-' },
              ],
            },
          });
        });

        mapInstance.on('mouseleave', 'pipelines-line', () => {
          if (!map.current) return;
          map.current.getCanvas().style.cursor = '';
          setHoveredPipelineDivision(null);
          setTooltip({ x: 0, y: 0, content: null });
        });
      }

      if (mapInstance.getLayer('nodes-circle') && mapInstance.getLayer('pipelines-line')) {
        mapInstance.moveLayer('nodes-circle');
      }
    };

    updatePipelines();
  }, [pipelinesGeoJSON, mapReady]);

  // Update nodes layer
  useEffect(() => {
    if (!map.current || !mapReady || !nodesGeoJSON) return;

    const updateNodes = () => {
      if (!map.current) return;
      const mapInstance = map.current;

      if (mapInstance.getSource('nodes')) {
        (mapInstance.getSource('nodes') as maplibregl.GeoJSONSource).setData(nodesGeoJSON);
      } else {
        mapInstance.addSource('nodes', {
          type: 'geojson',
          data: nodesGeoJSON,
        });

        mapInstance.addLayer({
          id: 'nodes-circle',
          type: 'circle',
          source: 'nodes',
          paint: {
            'circle-radius': [
              'interpolate',
              ['linear'],
              ['zoom'],
              3,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'calculated_flow']],
                0, 1.5,
                5000, 5
              ],
              6,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'calculated_flow']],
                0, 2.5,
                5000, 10
              ],
              9,
              [
                'interpolate',
                ['linear'],
                ['abs', ['get', 'calculated_flow']],
                0, 3.5,
                5000, 15
              ]
            ],
            'circle-color': [
              'match',
              ['get', 'pipeline_division'],
              ...Object.entries(PIPELINE_COLORS).flat(),
              '#999999'
            ] as any,
            'circle-stroke-width': [
              'interpolate',
              ['linear'],
              ['zoom'],
              3, 0.5,
              6, 1,
              9, 2
            ],
            'circle-stroke-color': '#ffffff',
            'circle-opacity': 0.9,
          },
        });

        // Mouse move handler
        mapInstance.on('mousemove', 'nodes-circle', (e: MapLayerMouseEvent) => {
          if (!map.current || !e.features || e.features.length === 0) return;

          map.current.getCanvas().style.cursor = 'pointer';

          const feature = e.features[0];
          const props = feature.properties as any;
          setHoveredNodeId(props.station_name || null);

          setTooltip({
            x: e.point.x,
            y: e.point.y,
            content: {
              title: props.station_name,
              items: [
                { label: '管道划分', value: props.pipeline_division || '-' },
                { label: '类型', value: props.node_type || '-' },
                { label: '控制类型', value: props.control_type || '-' },
                { label: '输入流量', value: props.input_flow != null ? `${props.input_flow.toFixed(2)} 万方/天` : '-' },
                { label: '计算流量', value: `${props.calculated_flow?.toFixed(2) || '-'} 万方/天` },
              ],
            },
          });
        });

        mapInstance.on('mouseleave', 'nodes-circle', () => {
          if (!map.current) return;
          map.current.getCanvas().style.cursor = '';
          setHoveredNodeId(null);
          setTooltip({ x: 0, y: 0, content: null });
        });

        // Click handler - navigate to detail page
        mapInstance.on('click', 'nodes-circle', (e: MapLayerMouseEvent) => {
          if (!e.features || e.features.length === 0) return;

          const feature = e.features[0];
          const stationName = feature.properties?.station_name;

          if (stationName) {
            navigate(`/node/${encodeURIComponent(stationName)}?date=${selectedDate}`);
          }
        });
      }

      if (mapInstance.getLayer('pipelines-line') && mapInstance.getLayer('nodes-circle')) {
        mapInstance.moveLayer('nodes-circle');
      }
    };

    updateNodes();
  }, [nodesGeoJSON, navigate, selectedDate, mapReady]);

  // 拖动调整宽度
  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    resizeRef.current = {
      startX: e.clientX,
      startWidth: chatWidth
    };
  };

  useEffect(() => {
    const handleResizeMove = (e: MouseEvent) => {
      if (!isResizing || !resizeRef.current) return;

      const deltaX = resizeRef.current.startX - e.clientX;
      const maxChatWidth = getMaxChatWidth();
      const newWidth = Math.min(Math.max(resizeRef.current.startWidth + deltaX, MIN_CHAT_WIDTH), maxChatWidth);
      setChatWidth(newWidth);
    };

    const handleResizeEnd = () => {
      setIsResizing(false);
      resizeRef.current = null;
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleResizeMove);
      document.addEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  useEffect(() => {
    const handleWindowResize = () => {
      setChatWidth((current) => Math.min(current, getMaxChatWidth()));
    };

    window.addEventListener('resize', handleWindowResize);
    return () => window.removeEventListener('resize', handleWindowResize);
  }, []);

  return (
    <div className="map-view-with-chat">
      {/* 左侧：地图区域 */}
      <div className="map-area">
        <div ref={mapContainer} className="map-canvas" />

        <DatePicker
          selectedDate={selectedDate}
          availableDates={availableDates}
          onChange={setSelectedDate}
        />

        <Legend pipelineColors={PIPELINE_COLORS} />

        {tooltip.content && (
          <div
            className="map-tooltip"
            style={{
              left: tooltip.x + 15,
              top: tooltip.y - 10,
            }}
          >
            <h4>{tooltip.content.title}</h4>
            {tooltip.content.items.map((item, idx) => (
              <div key={idx} className="map-tooltip-row">
                <span className="map-tooltip-label">{item.label}:</span>
                <span className="map-tooltip-value">{item.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 右侧：聊天面板 - 使用 CSS 隐藏而不是卸载组件 */}
      <div
        className={`chat-sidebar ${isResizing ? 'resizing' : ''}`}
        style={{
          display: showChat ? 'flex' : 'none',
          width: chatWidth
        }}
      >
        {/* 拖动手柄 */}
        <div
          className="chat-resize-handle"
          onMouseDown={handleResizeStart}
          title="拖动调整宽度"
        />
        <RightDock
          currentDate={selectedDate}
          selectedItem={hoveredNodeId ? { type: 'node', id: hoveredNodeId } : hoveredPipelineDivision ? { type: 'system', id: hoveredPipelineDivision } : undefined}
        />
      </div>

      {/* 切换按钮 */}
      <button
        className={`chat-toggle-btn ${showChat ? 'chat-visible' : 'chat-hidden'}`}
        onClick={() => setShowChat(!showChat)}
        title={showChat ? '隐藏助手' : '显示助手'}
        style={{ right: chatWidth }}
      >
        {showChat ? '◀' : '▶'}
      </button>
    </div>
  );
}
