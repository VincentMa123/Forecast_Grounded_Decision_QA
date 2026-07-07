import { useState } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useConsumersByNode, useAvailableDates } from '../api/hooks';
import { DatePicker } from '../components/DatePicker';
import './NodeDetailPage.css';

export default function NodeDetailPage() {
  const { stationName } = useParams<{ stationName: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const initialDate = searchParams.get('date') || '2019-01-01';
  const [selectedDate, setSelectedDate] = useState(initialDate);

  const decodedStationName = decodeURIComponent(stationName || '');

  const { data: availableDatesData } = useAvailableDates('consumer_flow');
  const { data, isLoading, error } = useConsumersByNode(decodedStationName, selectedDate);

  const availableDates = availableDatesData?.dates || ['2019-01-01'];

  const totalConsumption = data?.records.reduce((sum, r) => sum + r.consumption, 0) || 0;

  if (isLoading) {
    return (
      <div className="node-detail-page">
        <div className="loading">加载中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="node-detail-page">
        <div className="error">加载失败,请稍后重试</div>
      </div>
    );
  }

  return (
    <div className="node-detail-page">
      <header className="detail-header">
        <button className="back-button" onClick={() => navigate(-1)}>
          ← 返回地图
        </button>
        <h1>{decodedStationName} - 消耗量详情</h1>
      </header>

      <section className="date-selector-section">
        <DatePicker
          selectedDate={selectedDate}
          availableDates={availableDates}
          onChange={setSelectedDate}
          label="查询日期"
        />
      </section>

      <section className="consumer-table-section">
        {!data || data.records.length === 0 ? (
          <div className="no-data">该节点在选定日期没有消耗量数据</div>
        ) : (
          <>
            <div className="table-container">
              <table className="consumer-table">
                <thead>
                  <tr>
                    <th>管线</th>
                    <th>所属地</th>
                    <th>供气点</th>
                    <th>站名</th>
                    <th>用户</th>
                    <th>消耗量 (万方/天)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.records.map((record, index) => (
                    <tr key={index}>
                      <td>{record.pipeline}</td>
                      <td>{record.location}</td>
                      <td>{record.supply_point}</td>
                      <td>{record.station_name}</td>
                      <td>{record.consumer}</td>
                      <td className="number">{record.consumption.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <footer className="table-footer">
              <div className="footer-item">
                <span className="footer-label">总记录数:</span>
                <span className="footer-value">{data.total_records}</span>
              </div>
              <div className="footer-item">
                <span className="footer-label">总消耗量:</span>
                <span className="footer-value">{totalConsumption.toFixed(2)} 万方/天</span>
              </div>
            </footer>
          </>
        )}
      </section>
    </div>
  );
}
