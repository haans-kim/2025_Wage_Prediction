import React, { useState, useEffect } from 'react';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Alert, AlertDescription } from './ui/alert';
import { InfoIcon, TrendingUp, TrendingDown } from 'lucide-react';
import { apiClient } from '../lib/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface ShapData {
  feature_names: string[];
  shap_values: number[];
  base_value: number;
  prediction: number;
  feature_importance: Record<string, number>;
}

interface ShapAnalysisProps {
  modelType?: 'baseup' | 'performance' | 'both';
  className?: string;
}

export const ShapAnalysis: React.FC<ShapAnalysisProps> = ({ 
  modelType = 'both',
  className = ''
}) => {
  const [baseupShap, setBaseupShap] = useState<ShapData | null>(null);
  const [performanceShap, setPerformanceShap] = useState<ShapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadShapData();
  }, [modelType]);

  const loadShapData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      if (modelType === 'baseup' || modelType === 'both') {
        const baseupResponse = await apiClient.get('/api/analysis/shap', {
          params: { target: 'wage_increase_bu_sbl', top_n: 10 }
        });
        setBaseupShap(baseupResponse.data);
      }
      
      if (modelType === 'performance' || modelType === 'both') {
        const performanceResponse = await apiClient.get('/api/analysis/shap', {
          params: { target: 'wage_increase_mi_sbl', top_n: 10 }
        });
        setPerformanceShap(performanceResponse.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load SHAP analysis');
    } finally {
      setLoading(false);
    }
  };

  const renderShapChart = (data: ShapData, title: string, color: string) => {
    const chartData = {
      labels: data.feature_names,
      datasets: [{
        label: '기여도 (%p)',
        data: data.shap_values,
        backgroundColor: data.shap_values.map(v => v > 0 ? color : '#EF4444'),
        borderColor: data.shap_values.map(v => v > 0 ? color : '#EF4444'),
        borderWidth: 1
      }]
    };

    const options = {
      indexAxis: 'y' as const,
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: `${title} - 예측값: ${(data.prediction * 100).toFixed(2)}%`
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const value = context.raw;
              return `기여도: ${value > 0 ? '+' : ''}${(value * 100).toFixed(3)}%p`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: '기여도 (%p)'
          }
        }
      }
    };

    return (
      <div style={{ height: '400px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  const renderImportancePie = (data: ShapData, title: string) => {
    const sortedFeatures = Object.entries(data.feature_importance)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5);

    const pieData = {
      labels: sortedFeatures.map(([name]) => name),
      datasets: [{
        data: sortedFeatures.map(([, value]) => value),
        backgroundColor: [
          '#3B82F6',
          '#60A5FA',
          '#93C5FD',
          '#DBEAFE',
          '#EFF6FF'
        ],
        borderWidth: 1
      }]
    };

    const options = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right' as const
        },
        title: {
          display: true,
          text: `${title} - 변수 중요도`
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const label = context.label || '';
              const value = context.parsed || 0;
              return `${label}: ${(value * 100).toFixed(1)}%`;
            }
          }
        }
      }
    };

    return (
      <div style={{ height: '300px' }}>
        <Pie data={pieData} options={options} />
      </div>
    );
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-4 text-gray-600">SHAP 분석 로딩 중...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent>
          <Alert className="bg-red-50 border-red-200">
            <AlertDescription className="text-red-800">
              {error}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (modelType === 'both') {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            SHAP 분석 - 임금인상률 기여도
            <div className="ml-auto">
              <InfoIcon className="w-4 h-4 text-gray-400 cursor-help" />
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="baseup" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="baseup">Base-up 인상률</TabsTrigger>
              <TabsTrigger value="performance">성과급 인상률</TabsTrigger>
            </TabsList>
            
            <TabsContent value="baseup" className="space-y-4">
              {baseupShap && (
                <>
                  {renderShapChart(baseupShap, 'Base-up 인상률', '#3B82F6')}
                  <div className="mt-6">
                    {renderImportancePie(baseupShap, 'Base-up')}
                  </div>
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-semibold text-blue-900 mb-2">주요 영향 요인</h4>
                    <ul className="space-y-1 text-sm text-blue-800">
                      <li>• 대기업 인상률이 가장 큰 영향 (상관계수 0.62)</li>
                      <li>• 미국 CPI가 두 번째로 중요한 요인</li>
                      <li>• 최저임금과 공공기관 인상률은 음의 영향</li>
                    </ul>
                  </div>
                </>
              )}
            </TabsContent>
            
            <TabsContent value="performance" className="space-y-4">
              {performanceShap && (
                <>
                  {renderShapChart(performanceShap, '성과급 인상률', '#10B981')}
                  <div className="mt-6">
                    {renderImportancePie(performanceShap, '성과급')}
                  </div>
                  <div className="mt-4 p-4 bg-green-50 rounded-lg">
                    <h4 className="font-semibold text-green-900 mb-2">주요 영향 요인</h4>
                    <ul className="space-y-1 text-sm text-green-800">
                      <li>• 매출 증가율이 압도적으로 중요 (상관계수 0.88)</li>
                      <li>• 영업이익 증가율도 양의 영향</li>
                      <li>• 그룹 BU 인상률은 음의 상관관계</li>
                    </ul>
                  </div>
                </>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    );
  }

  // Single model type rendering
  const shapData = modelType === 'baseup' ? baseupShap : performanceShap;
  const color = modelType === 'baseup' ? '#3B82F6' : '#10B981';
  const title = modelType === 'baseup' ? 'Base-up 인상률' : '성과급 인상률';

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title} SHAP 분석</CardTitle>
      </CardHeader>
      <CardContent>
        {shapData && (
          <>
            {renderShapChart(shapData, title, color)}
            <div className="mt-6">
              {renderImportancePie(shapData, title)}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};