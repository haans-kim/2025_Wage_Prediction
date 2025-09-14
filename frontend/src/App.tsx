import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DataUpload } from './pages/DataUpload';
import { Modeling } from './pages/Modeling';
import Analysis from './pages/Analysis';
import { Dashboard } from './pages/Dashboard';
import { Effects } from './pages/Effects';
import { ExplainerDashboard } from './pages/ExplainerDashboard';
import StrategicDashboard from './pages/StrategicDashboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/strategic" replace />} />
          <Route path="data" element={<DataUpload />} />
          <Route path="modeling" element={<Modeling />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="effects" element={<Effects />} />
          <Route path="explainer" element={<ExplainerDashboard />} />
          <Route path="strategic" element={<StrategicDashboard />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
