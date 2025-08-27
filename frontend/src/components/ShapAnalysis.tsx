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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
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
  available: boolean;
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  sample_explanation?: any;
  explainer_type: string;
  n_features: number;
  n_samples_analyzed: number;
  target?: string;
  message?: string;
  error?: string;
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
  }, [modelType]); // eslint-disable-line react-hooks/exhaustive-deps

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
    // Filter out features with zero importance
    const nonZeroFeatures = data.feature_importance.filter(f => Math.abs(f.importance) > 0.0001);
    
    // If no non-zero features, use top 10 features anyway
    const featuresToShow = nonZeroFeatures.length > 0 ? nonZeroFeatures : data.feature_importance.slice(0, 10);
    
    const chartData = {
      labels: featuresToShow.map(f => f.feature),
      datasets: [{
        label: '기여도 (%p)',
        data: featuresToShow.map(f => f.importance),
        backgroundColor: featuresToShow.map(f => f.importance > 0 ? color : '#EF4444'),
        borderColor: featuresToShow.map(f => f.importance > 0 ? color : '#EF4444'),
        borderWidth: 1
      }]
    };

    const options = {
      indexAxis: 'y' as const,
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          bottom: 20
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: title,
          padding: {
            top: 10,
            bottom: 10
          }
        },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const value = context.raw;
              return `중요도: ${value > 0 ? '+' : ''}${(value * 100).toFixed(3)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: '중요도'
          }
        }
      }
    };

    return (
      <div style={{ height: '350px' }}>
        <Bar data={chartData} options={options} />
      </div>
    );
  };

  const renderImportancePie = (data: ShapData, title: string) => {
    // Filter out features with zero importance
    const nonZeroFeatures = data.feature_importance.filter(f => Math.abs(f.importance) > 0.0001);
    
    // Sort and get top 5 features
    const sortedFeatures = nonZeroFeatures.length > 0 
      ? nonZeroFeatures.sort((a, b) => b.importance - a.importance).slice(0, 5)
      : data.feature_importance.slice(0, 5);

    const pieData = {
      labels: sortedFeatures.map(f => f.feature),
      datasets: [{
        data: sortedFeatures.map(f => Math.abs(f.importance)),
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
      <>
        <Card className={className}>
          <CardHeader>
            <CardTitle className="text-xl">
              특성 중요도 분석
            </CardTitle>
            <CardDescription>
              각 변수가 임금인상률 예측에 미치는 영향도
            </CardDescription>
          </CardHeader>
        </Card>

        {/* 2열 레이아웃으로 SHAP 분석 표시 */}
        <div className="grid grid-cols-2 gap-4">
          {/* Base-up SHAP 분석 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg text-blue-600 dark:text-blue-400">
                Base-up 특성 중요도
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pb-6">
              {baseupShap && baseupShap.available ? (
                <div style={{ height: '350px', paddingBottom: '20px' }}>
                  {renderShapChart(baseupShap, 'Base-up', '#3B82F6')}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  SHAP 분석 데이터를 로딩 중...
                </div>
              )}
            </CardContent>
          </Card>

          {/* 성과급 SHAP 분석 */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg text-green-600 dark:text-green-400">
                성과급 특성 중요도
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 pb-6">
              {performanceShap && performanceShap.available && performanceShap.feature_importance && performanceShap.feature_importance.length > 0 ? (
                <div style={{ height: '350px', paddingBottom: '20px' }}>
                  {renderShapChart(performanceShap, '성과급', '#10B981')}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  SHAP 분석 데이터를 로딩 중...
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </>
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