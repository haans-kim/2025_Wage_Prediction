import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Alert, AlertDescription } from './ui/alert';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  AlertTriangle,
  CheckCircle,
  Sliders
} from 'lucide-react';
import { apiClient } from '../lib/api';

interface ScenarioResult {
  scenario: string;
  changes: Record<string, number>;
  predictions: {
    baseup?: {
      current: number;
      scenario: number;
      change: number;
    };
    performance?: {
      current: number;
      scenario: number;
      change: number;
    };
    total?: {
      current: number;
      scenario: number;
      change: number;
    };
  };
}

export const WhatIfScenario: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<string>('moderate');
  const [customValues, setCustomValues] = useState({
    gdp_change: 0,
    cpi_change: 0,
    major_group_rate_change: 0,
    revenue_growth_change: 0
  });
  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scenarios = [
    {
      id: 'recession',
      name: '경기 침체',
      icon: TrendingDown,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      description: 'GDP -2%, CPI -1%, 대기업 인상률 -1%'
    },
    {
      id: 'moderate',
      name: '안정 성장',
      icon: Activity,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      description: 'GDP 2%, CPI 2.5%, 대기업 인상률 4.5%'
    },
    {
      id: 'custom',
      name: '사용자 정의',
      icon: Sliders,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      description: '직접 변수 값을 조정'
    },
    {
      id: 'inflation',
      name: '고인플레이션',
      icon: AlertTriangle,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      description: 'GDP 0.5%, CPI 5%, 대기업 인상률 2%'
    },
    {
      id: 'boom',
      name: '경제 호황',
      icon: TrendingUp,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      description: 'GDP 4%, CPI 3%, 매출 20% 증가'
    }
  ];

  const analyzeScenario = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const payload = {
        scenario_type: selectedScenario,
        ...customValues
      };
      
      const response = await apiClient.post('/api/analysis/what-if-scenario', payload);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze scenario');
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatChange = (value: number) => {
    const formatted = formatPercent(value);
    if (value > 0) return `+${formatted}`;
    return formatted;
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>What-If 시나리오 분석</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            {scenarios.map(scenario => (
              <button
                key={scenario.id}
                onClick={() => setSelectedScenario(scenario.id)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedScenario === scenario.id
                    ? `border-blue-500 ${scenario.bgColor}`
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <scenario.icon className={`w-8 h-8 mb-2 ${scenario.color}`} />
                <h3 className="font-semibold">{scenario.name}</h3>
                <p className="text-xs text-gray-600 mt-1">{scenario.description}</p>
              </button>
            ))}
          </div>

          {selectedScenario === 'custom' && (
            <div className="grid grid-cols-2 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-sm font-medium mb-1">GDP 변화 (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={customValues.gdp_change}
                  onChange={(e) => setCustomValues({...customValues, gdp_change: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">CPI 변화 (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={customValues.cpi_change}
                  onChange={(e) => setCustomValues({...customValues, cpi_change: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">대기업 인상률 변화 (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={customValues.major_group_rate_change}
                  onChange={(e) => setCustomValues({...customValues, major_group_rate_change: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border rounded-md"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">매출 증가율 변화 (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={customValues.revenue_growth_change}
                  onChange={(e) => setCustomValues({...customValues, revenue_growth_change: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 border rounded-md"
                />
              </div>
            </div>
          )}

          <Button 
            onClick={analyzeScenario} 
            disabled={loading}
            className="w-full"
          >
            {loading ? '분석 중...' : '시나리오 분석 실행'}
          </Button>

          {error && (
            <Alert className="mt-4 bg-red-50 border-red-200">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-red-800">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>시나리오 분석 결과</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {result.predictions.baseup && (
                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-semibold text-blue-900 mb-3">Base-up 인상률</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">현재:</span>
                      <span className="font-medium">{formatPercent(result.predictions.baseup.current)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">시나리오:</span>
                      <span className="font-medium">{formatPercent(result.predictions.baseup.scenario)}</span>
                    </div>
                    <div className="flex justify-between border-t pt-2">
                      <span className="text-gray-600">변화:</span>
                      <span className={`font-bold ${result.predictions.baseup.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatChange(result.predictions.baseup.change)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {result.predictions.performance && (
                <div className="p-4 bg-green-50 rounded-lg">
                  <h3 className="font-semibold text-green-900 mb-3">성과급 인상률</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">현재:</span>
                      <span className="font-medium">{formatPercent(result.predictions.performance.current)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">시나리오:</span>
                      <span className="font-medium">{formatPercent(result.predictions.performance.scenario)}</span>
                    </div>
                    <div className="flex justify-between border-t pt-2">
                      <span className="text-gray-600">변화:</span>
                      <span className={`font-bold ${result.predictions.performance.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatChange(result.predictions.performance.change)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {result.predictions.total && (
                <div className="p-4 bg-purple-50 rounded-lg">
                  <h3 className="font-semibold text-purple-900 mb-3">총 인상률</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">현재:</span>
                      <span className="font-medium">{formatPercent(result.predictions.total.current)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">시나리오:</span>
                      <span className="font-medium">{formatPercent(result.predictions.total.scenario)}</span>
                    </div>
                    <div className="flex justify-between border-t pt-2">
                      <span className="text-gray-600">변화:</span>
                      <span className={`font-bold ${result.predictions.total.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {formatChange(result.predictions.total.change)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold mb-2">적용된 변화</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {Object.entries(result.changes).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-gray-600">{key.replace(/_/g, ' ')}:</span>
                    <span className="font-medium">{formatChange(value as number)}</span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};