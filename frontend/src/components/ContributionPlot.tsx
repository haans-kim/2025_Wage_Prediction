import React, { useState, useEffect, useCallback } from 'react';
import { Chart } from 'react-chartjs-2';
import { apiClient } from '../lib/api';

interface ContributionData {
  feature: string;
  feature_korean: string;
  feature_value: number;
  contribution: number;
  contribution_abs: number;
}

interface ContributionPlotData {
  message: string;
  sample_index: number;
  prediction: number;
  actual_value: number;
  baseline_prediction: number;
  total_contribution: number;
  residual: number;
  contributions: ContributionData[];
  n_features: number;
}

interface ContributionPlotProps {
  sampleIndex?: number;
  topN?: number;
  height?: number;
}

const ContributionPlot: React.FC<ContributionPlotProps> = ({ 
  sampleIndex = 0, 
  topN = 10, 
  height = 500 
}) => {
  const [data, setData] = useState<ContributionPlotData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchContributionData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = new URLSearchParams();
      if (sampleIndex !== undefined) {
        params.append('sample_index', sampleIndex.toString());
      }
      params.append('top_n_features', topN.toString());
      
      const response = await apiClient.get(`/api/analysis/contribution-plot?${params}`);
      console.log('🔍 Contribution API response:', response.data);
      
      // Check if response contains an error
      if (response.data.error || response.data.available === false) {
        throw new Error(response.data.error || 'Analysis not available');
      }
      
      setData(response.data);
    } catch (err: any) {
      console.error('Failed to fetch contribution plot data:', err);
      setError(err.message || err.response?.data?.detail || 'Failed to load contribution analysis');
    } finally {
      setLoading(false);
    }
  }, [sampleIndex, topN]);

  useEffect(() => {
    fetchContributionData();
  }, [fetchContributionData]);

  const getChartData = () => {
    if (!data || !data.contributions || !Array.isArray(data.contributions)) return null;

    const chartData = data.contributions
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .map(item => ({
        label: item.feature_korean,
        contribution: Math.round((item.contribution * 100) * 10) / 10, // Convert to percentage points and round to 1 decimal
        backgroundColor: item.contribution >= 0 ? '#10b981' : '#ef4444'
      }));

    return {
      labels: chartData.map(item => item.label),
      datasets: [
        {
          label: '기여도 (%p)',
          data: chartData.map(item => item.contribution),
          backgroundColor: chartData.map(item => item.backgroundColor),
          borderColor: chartData.map(item => item.backgroundColor),
          borderWidth: 1,
        }
      ]
    };
  };

  const getChartOptions = () => ({
    indexAxis: 'y' as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: '총 인상률 예측 기여도 분석',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.x;
            const percentage = data ? (Math.abs(value) / Math.abs(data.total_contribution * 100)) * 100 : 0;
            return `기여도: ${value >= 0 ? '+' : ''}${value.toFixed(1)}%p (${percentage.toFixed(1)}%)`;
          }
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        title: {
          display: true,
          text: '임금인상률 기여도 (%p)'
        },
        ticks: {
          callback: (value: any) => {
            const sign = value >= 0 ? '+' : '';
            return `${sign}${value.toFixed(1)}%p`;
          }
        },
        grid: {
          drawBorder: false,
          color: (context: any) => {
            if (context.tick.value === 0) {
              return 'rgba(0, 0, 0, 0.3)';
            }
            return 'rgba(0, 0, 0, 0.1)';
          }
        }
      },
      y: {
        ticks: {
          autoSkip: false,
          font: {
            size: 11
          }
        },
        grid: {
          display: false
        }
      }
    }
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <span className="ml-3 text-gray-600">기여도 분석 로딩 중...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-md">
        <div className="flex">
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">기여도 분석 오류</h3>
            <p className="mt-2 text-sm text-red-700">{error}</p>
            <button
              onClick={fetchContributionData}
              className="mt-3 bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded text-sm"
            >
              다시 시도
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">
          총 인상률 예측 기여도 분석
        </h3>
        
        {/* 예측 정보 요약 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-xs text-gray-500">예측값</div>
            <div className="text-lg font-bold text-blue-600">
              {(data.prediction * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">실제값</div>
            <div className="text-lg font-bold text-green-600">
              {(data.actual_value * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">기준선</div>
            <div className="text-lg font-bold text-gray-600">
              {(data.baseline_prediction * 100).toFixed(2)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500">총 기여도</div>
            <div className="text-lg font-bold text-purple-600">
              {data.total_contribution >= 0 ? '+' : ''}{(data.total_contribution * 100).toFixed(2)}%p
            </div>
          </div>
        </div>
        
        <p className="text-sm text-gray-600">
          샘플 #{data.sample_index}에 대한 각 요인의 예측 기여도를 보여줍니다.
          양수는 인상률을 높이는 요인, 음수는 낮추는 요인입니다.
        </p>
      </div>

      {/* 기여도 차트 */}
      <div style={{ height }}>
        {getChartData() ? (
          <Chart 
            type='bar'
            data={getChartData()!} 
            options={getChartOptions()} 
          />
        ) : (
          <div className="h-full bg-gray-50 border rounded-md flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">차트 데이터 준비 중...</p>
            </div>
          </div>
        )}
      </div>

      {/* 범례 */}
      <div className="mt-4 flex justify-center space-x-6 text-sm">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-sm mr-2"></div>
          <span>인상률을 높이는 요인</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-sm mr-2"></div>
          <span>인상률을 낮추는 요인</span>
        </div>
      </div>
      
      {/* 상세 데이터 테이블 */}
      <div className="mt-6">
        <h4 className="text-md font-medium text-gray-900 mb-3">상세 기여도</h4>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-gray-500 uppercase tracking-wider">
                  요인
                </th>
                <th className="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">
                  값
                </th>
                <th className="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">
                  기여도 (%p)
                </th>
                <th className="px-4 py-2 text-right font-medium text-gray-500 uppercase tracking-wider">
                  비중 (%)
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.contributions && Array.isArray(data.contributions) && data.contributions.map((item, index) => {
                const percentage = (Math.abs(item.contribution) / Math.abs(data.total_contribution)) * 100;
                return (
                  <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-4 py-2 text-sm text-gray-900">
                      {item.feature_korean}
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-600 text-right">
                      {item.feature_value.toFixed(1)}
                    </td>
                    <td className="px-4 py-2 text-sm text-right">
                      <span className={`font-medium ${item.contribution >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {item.contribution >= 0 ? '+' : ''}{(item.contribution * 100).toFixed(1)}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-600 text-right">
                      {percentage.toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ContributionPlot;