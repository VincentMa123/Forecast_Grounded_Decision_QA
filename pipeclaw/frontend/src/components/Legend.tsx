import React from 'react';
import './Legend.css';

interface LegendProps {
  pipelineColors: Record<string, string>;
}

export const Legend: React.FC<LegendProps> = ({ pipelineColors }) => {
  return (
    <div className="legend">
      <div className="legend-section">
        <h3 className="legend-title">管道划分</h3>
        <div className="legend-items">
          {Object.entries(pipelineColors).map(([name, color]) => (
            <div key={name} className="legend-item">
              <span
                className="legend-color"
                style={{ backgroundColor: color }}
              />
              <span className="legend-label">{name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="legend-section">
        <h3 className="legend-title">流量图例</h3>
        <div className="legend-scale">
          <div className="legend-scale-item">
            <span className="legend-icon">━━</span>
            <span className="legend-text">线条粗细 = 管道流量</span>
          </div>
          <div className="legend-scale-item">
            <span className="legend-icon">●</span>
            <span className="legend-text">节点大小 = 计算流量</span>
          </div>
        </div>
      </div>
    </div>
  );
};
