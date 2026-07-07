import { Routes, Route } from 'react-router-dom';
import './App.css';
import MapView from './components/MapView';
import NodeDetailPage from './pages/NodeDetailPage';
import ChatTestPage from './pages/ChatTestPage';

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <span>⚡</span>
          <span>GIS管网可视化调度系统</span>
        </h1>
        <p className="app-header-subtitle">
          Gas Pipeline Network Visualization & Dispatch System
        </p>
      </header>
      <main className="app-content">
        <Routes>
          <Route path="/" element={<MapView />} />
          <Route path="/node/:stationName" element={<NodeDetailPage />} />
          <Route path="/chat" element={<ChatTestPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
