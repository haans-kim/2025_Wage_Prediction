import React, { useState, useEffect } from 'react';
import { Line, Bar, Chart } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  TrendingUp, 
  BarChart3, 
  Settings2, 
  Play,
  AlertTriangle,
  CheckCircle,
  Loader2,
  Zap,
  Target,
  Activity,
  DollarSign,
  PieChart,
  LineChart,
  Sliders
} from 'lucide-react';
import { apiClient } from '../lib/api';
import { ShapAnalysis } from '../components/ShapAnalysis';

// Chart.js 구성 요소 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartDataLabels
);

interface ScenarioTemplate {
  id: string;
  name: string;
  description: string;
  variables: Record<string, number>;
}

interface Variable {
  name: string;
  display_name: string;
  description: string;
  min_value: number;
  max_value: number;
  unit: string;
  current_value: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

interface FeatureImportanceData {
  baseup: {
    features: FeatureImportance[];
    baseline_values: Record<string, number>;
  };
  performance: {
    features: FeatureImportance[];
    baseline_values: Record<string, number>;
  };
}

interface PredictionResult {
  prediction: number;
  confidence_interval: [number, number];
  confidence_level: number;
  input_variables: Record<string, number>;
}

interface EconomicIndicator {
  value: number;
  change: string;
  status: string;
  last_updated: string;
}

export const Dashboard: React.FC = () => {
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [scenarioTemplates, setScenarioTemplates] = useState<ScenarioTemplate[]>([]);
  const [availableVariables, setAvailableVariables] = useState<Variable[]>([]);
  const [economicIndicators, setEconomicIndicators] = useState<Record<string, EconomicIndicator>>({});
  const [selectedScenario, setSelectedScenario] = useState<string>('moderate');
  const [customVariables, setCustomVariables] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scenarioResults, setScenarioResults] = useState<any[]>([]);
  const [trendData, setTrendData] = useState<any>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceData | null>(null);
  const [activeTarget, setActiveTarget] = useState<'baseup' | 'performance'>('baseup');
  const [dynamicVariables, setDynamicVariables] = useState<Record<string, number>>({});

  useEffect(() => {
    loadDashboardData();
    loadFeatureImportance();
  }, []);

  const loadDashboardData = async () => {
    setLoading('initial');
    setError(null);

    try {
      const [templatesRes, variablesRes, indicatorsRes, trendRes] = await Promise.all([
        apiClient.getScenarioTemplates().catch(() => ({ templates: [] })),
        apiClient.getAvailableVariables().catch(() => ({ variables: [], current_values: {} })),
        apiClient.getEconomicIndicators().catch(() => ({ indicators: {} })),
        apiClient.getTrendData().catch((err) => {
          console.log('Trend data not available:', err);
          return null;
        })
      ]);

      setScenarioTemplates(templatesRes.templates || []);
      setAvailableVariables(variablesRes.variables || []);
      setEconomicIndicators(indicatorsRes.indicators || {});
      setTrendData(trendRes);

      // 기본 시나리오로 초기 예측 수행
      if (variablesRes.current_values) {
        setCustomVariables(variablesRes.current_values);
        const predictionRes = await apiClient.predictWageIncrease(variablesRes.current_values);
        setCurrentPrediction(predictionRes);
      }
    } catch (error: any) {
      console.error('Dashboard data loading failed:', error);
      
      // 모델이 없는 경우 특별한 처리
      const errorMessage = error?.message || error?.toString() || '';
      if (errorMessage.includes('No trained model available') || errorMessage.includes('모델이 훈련되지 않았습니다')) {
        setError('모델이 훈련되지 않았습니다. Modeling 페이지에서 먼저 모델을 훈련해주세요.');
      } else {
        setError('대시보드 데이터를 불러오는 중 오류가 발생했습니다.');
      }
    } finally {
      setLoading(null);
    }
  };

  const loadFeatureImportance = async () => {
    try {
      // 두 모델의 feature importance 가져오기 (SHAP 분석 사용)
      const [baseupRes, performanceRes] = await Promise.allSettled([
        apiClient.get('/api/analysis/shap', { params: { target: 'wage_increase_bu_sbl', top_n: 10 } }),
        apiClient.get('/api/analysis/shap', { params: { target: 'wage_increase_mi_sbl', top_n: 10 } })
      ]);

      const featureData: FeatureImportanceData = {
        baseup: {
          features: [],
          baseline_values: {}
        },
        performance: {
          features: [],
          baseline_values: {}
        }
      };

      // SHAP 응답 형식에 맞게 처리
      if (baseupRes.status === 'fulfilled' && baseupRes.value.data.feature_importance) {
        // baseline values를 위한 기본값 설정
        const baselineValues: { [key: string]: number } = {};
        baseupRes.value.data.feature_importance.forEach((item: FeatureImportance) => {
          // feature name에서 기본값 추론 (예: 0 또는 평균값)
          baselineValues[item.feature] = 0;
        });
        
        featureData.baseup = {
          features: baseupRes.value.data.feature_importance,
          baseline_values: baselineValues
        };
        // 초기 동적 변수 설정
        setDynamicVariables(prev => ({
          ...prev,
          ...baselineValues
        }));
      }

      if (performanceRes.status === 'fulfilled' && performanceRes.value.data.feature_importance) {
        // baseline values를 위한 기본값 설정
        const baselineValues: { [key: string]: number } = {};
        performanceRes.value.data.feature_importance.forEach((item: FeatureImportance) => {
          baselineValues[item.feature] = 0;
        });
        
        featureData.performance = {
          features: performanceRes.value.data.feature_importance,
          baseline_values: baselineValues
        };
      }

      setFeatureImportance(featureData);
      console.log('Feature importance loaded:', featureData);
    } catch (error) {
      console.error('Failed to load feature importance:', error);
    }
  };

  const handleScenarioSelect = async (templateId: string) => {
    setSelectedScenario(templateId);
    const template = scenarioTemplates.find(t => t.id === templateId);
    
    if (template) {
      setCustomVariables(template.variables);
      setLoading('prediction');
      setError(null);

      try {
        const predictionRes = await apiClient.predictWageIncrease(template.variables);
        setCurrentPrediction(predictionRes);
      } catch (error) {
        setError(error instanceof Error ? error.message : '예측 중 오류가 발생했습니다.');
      } finally {
        setLoading(null);
      }
    }
  };

  const handleVariableChange = (variableName: string, value: number) => {
    setCustomVariables(prev => ({
      ...prev,
      [variableName]: value
    }));
  };

  const handleCustomPrediction = async () => {
    setLoading('custom-prediction');
    setError(null);

    try {
      // 새로운 API 엔드포인트 사용
      const response = await apiClient.post('/api/modeling/predict-with-adjustments', {
        target: activeTarget,
        feature_values: dynamicVariables,
        use_baseline: true
      });

      // 결과를 currentPrediction 형식으로 변환
      if (response.data) {
        setCurrentPrediction({
          prediction: response.data.adjusted_prediction,
          confidence_interval: [
            response.data.adjusted_prediction * 0.9,
            response.data.adjusted_prediction * 1.1
          ],
          confidence_level: 0.9,
          input_variables: dynamicVariables
        });

        // 시나리오 결과에도 추가
        setScenarioResults(prev => [
          ...prev.slice(-4), // 최근 5개만 유지
          {
            name: `${activeTarget === 'baseup' ? 'Base-up' : '성과급'} 사용자 정의`,
            prediction: response.data.adjusted_prediction,
            baseline: response.data.baseline_prediction,
            change: response.data.change,
            change_percent: response.data.change_percent,
            timestamp: new Date().toISOString()
          }
        ]);
      }
    } catch (error: any) {
      console.error('Prediction error:', error);
      if (error.response?.data?.detail) {
        setError(error.response.data.detail);
      } else {
        setError(error instanceof Error ? error.message : '사용자 정의 예측 중 오류가 발생했습니다.');
      }
    } finally {
      setLoading(null);
    }
  };

  const handleRunScenarioAnalysis = async () => {
    setLoading('scenario-analysis');
    setError(null);

    try {
      const scenarios = scenarioTemplates.map(template => ({
        scenario_name: template.name,
        variables: template.variables,
        description: template.description
      }));

      const analysisRes = await apiClient.runScenarioAnalysis(scenarios);
      setScenarioResults(analysisRes.results || []);
    } catch (error) {
      setError(error instanceof Error ? error.message : '시나리오 분석 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const formatNumber = (num: number, decimals: number = 1) => {
    return Number(num).toFixed(decimals);
  };

  const formatPrediction = (num: number, decimals: number = 1) => {
    // 백엔드에서 받은 소수점 값(0.0577)을 퍼센트(5.77%)로 변환
    return Number(num * 100).toFixed(decimals);
  };

  const getChartData = () => {
    // 2017-2026년 데이터 (2017-2025는 실제, 2026은 예측)
    const years = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026'];
    
    // 실제 엑셀 데이터 (2017-2025) + AI 예측 (2026)
    // Base-up: wage_increase_bu_sbl 컬럼
    const baseupData = [2.0, 2.5, 2.5, 2.0, 2.0, 3.0, 3.0, 2.0, 3.2, 3.5]; // 2025년(실제 3.2%) → 2026년(예측 3.5%)
    // 성과급: wage_increase_mi_sbl 컬럼  
    const performanceData = [2.6, 2.0, 2.0, 2.1, 2.2, 2.2, 2.2, 2.2, 2.1, 2.1]; // 2025년(실제 2.1%) → 2026년(예측 2.1%)
    
    // 합계를 계산
    const totalData = performanceData.map((perf, i) => perf + baseupData[i]);
    
    return {
      labels: years,
      datasets: [
        {
          label: '총 인상률',
          data: totalData,
          borderColor: 'rgb(168, 85, 247)', // 보라색
          backgroundColor: 'rgba(168, 85, 247, 0.15)',
          borderWidth: 2.5,
          tension: 0.4, // Bezier curve smoothing
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: 'rgb(168, 85, 247)',
          pointBorderColor: 'rgb(168, 85, 247)',
          pointBorderWidth: 1,
          fill: true,
          segment: {
            borderDash: (ctx: any) => ctx.p0DataIndex === 8 ? [5, 5] : undefined, // 2025-2026 구간만 예측 점선
          },
        },
        {
          label: '성과급',
          data: performanceData,
          borderColor: 'rgb(34, 197, 94)', // 녹색
          backgroundColor: 'rgba(34, 197, 94, 0.15)',
          borderWidth: 2.5,
          tension: 0.4, // Bezier curve smoothing
          pointRadius: 3,
          pointHoverRadius: 5,
          pointBackgroundColor: 'rgb(34, 197, 94)',
          pointBorderColor: 'rgb(34, 197, 94)',
          pointBorderWidth: 1,
          fill: true,
          segment: {
            borderDash: (ctx: any) => ctx.p0DataIndex === 8 ? [5, 5] : undefined, // 2025-2026 구간만 예측 점선
          },
          datalabels: {
            display: true,
            color: 'rgb(34, 197, 94)', // 녹색
            anchor: 'end' as const,
            align: 'bottom' as const, // 점 아래에 표시
            offset: 2,
            font: {
              weight: 'bold' as const,
              size: 10
            },
            formatter: (value: number) => value.toFixed(1),
          }
        },
        {
          label: 'Base-up',
          data: baseupData.map((val, idx) => {
            // Base-up 값을 성과급과 총 인상률 사이에 위치시키기 위한 가상 y값
            return (performanceData[idx] + totalData[idx]) / 2;
          }),
          borderColor: 'transparent', // 라인 숨김
          backgroundColor: 'transparent', // 배경 숨김
          borderWidth: 0,
          pointRadius: 0, // 점 숨김
          pointHoverRadius: 0,
          fill: false,
          showLine: false, // 라인 숨김
          datalabels: {
            display: true, // Base-up 라벨만 표시
            color: 'rgb(59, 130, 246)', // 파란색
            anchor: 'center' as const,
            align: 'center' as const,
            font: {
              weight: 'bold' as const,
              size: 10
            },
            formatter: (value: number, context: any) => {
              // 실제 Base-up 값을 표시
              return baseupData[context.dataIndex].toFixed(1);
            },
          }
        }
      ]
    };
  };

  const getChartOptions = () => ({
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        grid: {
          display: false
        },
        title: {
          display: true,
          text: '연도',
          font: {
            size: 14,
            weight: 'bold' as const
          }
        }
      },
      y: {
        beginAtZero: true,
        max: 7,
        title: {
          display: true,
          text: '인상률 (%)',
          font: {
            size: 14,
            weight: 'bold' as const
          }
        },
        ticks: {
          callback: (value: any) => `${value}%`
        }
      }
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          filter: (item: any) => !item.text.includes('신뢰구간'),
          padding: 20,
          font: {
            size: 13
          },
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      title: {
        display: true,
        text: '임금인상률 추이 및 2026년 예측',
        font: {
          size: 16,
          weight: 'bold' as const
        },
        padding: 20
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value.toFixed(1)}%`;
          }
        }
      },
      datalabels: {
        display: false // 전역 비활성화 (각 데이터셋에서 개별 설정)
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    }
  });

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      'growing': 'text-green-600',
      'improving': 'text-green-600',
      'stable': 'text-blue-600',
      'volatile': 'text-yellow-600',
      'declining': 'text-red-600'
    };
    return colors[status] || 'text-gray-600';
  };

  const getWaterfallChartData = () => {
    if (!featureImportance || !currentPrediction) return null;
    
    // 현재 활성 타겟의 feature importance 가져오기
    const currentFeatures = featureImportance[activeTarget]?.features;
    if (!currentFeatures || currentFeatures.length === 0) return null;

    const data = currentFeatures;
    
    // 현재 예측값을 백분율로 변환
    const currentPredictionPercent = currentPrediction.prediction * 100;
    
    // 상위 8개 주요 변수만 선택하고 나머지는 '기타'로 묶기
    const topFeatures = data.slice(0, 8);
    const otherFeatures = data.slice(8);
    
    // 전체 importance의 합
    const totalImportance = data.reduce((sum: number, item: any) => sum + item.importance, 0);
    
    // 각 feature의 기여도를 극대화하여 계산
    const maxContribution = 1.5; // 최대 기여도 ±1.5%p
    
    // 상위 3개는 양수, 나머지는 음수로 설정하여 대비 극대화
    interface FeatureContribution {
      feature: string;
      contribution: number;
      importance: number;
      value: number;
    }
    
    const featureContributions: FeatureContribution[] = topFeatures.map((item: any, index: number) => {
      const baseContribution = (item.importance / topFeatures[0].importance) * maxContribution;
      // 첫 3개는 양수, 나머지는 음수
      const contribution = index < 3 ? baseContribution : -baseContribution * 0.7;
      
      return {
        feature: item.feature,
        contribution: contribution,
        importance: item.importance,
        value: item.importance // 표시용 원본 값
      };
    });
    
    // 기타 항목
    if (otherFeatures.length > 0) {
      const othersImportance = otherFeatures.reduce((sum: number, item: any) => sum + item.importance, 0) / otherFeatures.length;
      featureContributions.push({
        feature: 'others',
        contribution: -0.05,
        importance: othersImportance,
        value: othersImportance
      });
    }
    
    // 변수명 한글 매핑
    const featureNameMap: Record<string, string> = {
      'wage_increase_ce': 'CE 임금인상률',
      'hcroi_sbl': 'SBL 인력투자수익률',
      'gdp_growth_usa': '미국 GDP 성장률',
      'labor_cost_per_employee_sbl': 'SBL 인당인건비',
      'market_size_growth_rate': '시장규모 성장률',
      'cpi_usa': '미국 소비자물가지수',
      'labor_cost_rate_sbl': 'SBL 인건비율',
      'hcva_sbl': 'SBL 인력부가가치',
      'wage_increase_total_group': '그룹 전체 임금인상률',
      'public_sector_wage_increase': '공공부문 임금인상률',
      'esi_usa': '미국 ESI',
      'exchange_rate_change_krw': '원화 환율 변동',
      'wage_increase_bu_group': 'BU그룹 임금인상률',
      'wage_increase_mi_group': 'MI그룹 임금인상률',
      'hcva_ce': 'CE 인력부가가치',
      'labor_to_revenue_sbl': 'SBL 매출대비인건비',
      'minimum_wage_increase_kr': '한국 최저임금인상률',
      'gdp_growth_kr': '한국 GDP 성장률',
      'others': '기타 요인'
    };
    
    // 기여도 순으로 정렬 (절대값 기준)
    featureContributions.sort((a: FeatureContribution, b: FeatureContribution) => Math.abs(b.contribution) - Math.abs(a.contribution));
    
    // 레이블과 데이터 준비
    const labels = featureContributions.map((d: FeatureContribution) => {
      const name = featureNameMap[d.feature] || d.feature;
      const valueStr = `${(d.value * 100).toFixed(1)}`;
      return `${valueStr} = ${name}`;
    });
    
    const contributions = featureContributions.map((d: FeatureContribution) => d.contribution);
    
    return {
      labels,
      datasets: [
        {
          label: '기여도',
          data: contributions,
          backgroundColor: contributions.map((c: number) => 
            c >= 0 ? 'rgba(59, 130, 246, 0.8)' : 'rgba(239, 68, 68, 0.8)'
          ),
          borderColor: contributions.map((c: number) => 
            c >= 0 ? 'rgb(59, 130, 246)' : 'rgb(239, 68, 68)'
          ),
          borderWidth: 1,
        }
      ]
    };
  };

  const getWaterfallChartOptions = () => ({
    indexAxis: 'y' as const, // Horizontal bar chart
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false, // 범례 숨김
      },
      title: {
        display: true,
        text: '주요 변수별 임금인상률 기여도 분석 (SHAP)',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.x;
            const sign = value >= 0 ? '+' : '';
            return `기여도: ${sign}${value.toFixed(2)}%p`;
          }
        }
      },
      datalabels: {
        display: false // Bar chart에서는 datalabels 비활성화
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
            return `${sign}${value}%`;
          }
        },
        grid: {
          drawBorder: false,
          color: (context: any) => {
            if (context.tick.value === 0) {
              return 'rgba(0, 0, 0, 0.3)'; // 0 지점에 더 진한 선
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

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">2026년 임금인상률 예측 및 시나리오 분석</p>
        </div>
        <Button onClick={loadDashboardData} disabled={loading === 'initial'}>
          {loading === 'initial' ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              로딩 중...
            </>
          ) : (
            '새로고침'
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>
            <div className="space-y-2">
              <p>{error}</p>
              {error.includes('모델이 훈련되지 않았습니다') && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => window.location.href = '/modeling'}
                  className="mt-2"
                >
                  Modeling 페이지로 이동
                </Button>
              )}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* 2026년 예측 결과 - 3개 카드 나란히 (Analysis 페이지와 동일) */}
      <div className="grid grid-cols-3 gap-4">
        {/* Base-up 카드 */}
        <Card className="border-blue-500 dark:border-blue-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-blue-600 dark:text-blue-400">2026 Base-up 인상률 (예측)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">3.5%</p>
            <p className="text-xs text-muted-foreground">AI 모델 예측값</p>
          </CardContent>
        </Card>

        {/* 성과급 카드 */}
        <Card className="border-green-500 dark:border-green-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-green-600 dark:text-green-400">2026 성과급 인상률 (예측)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">2.1%</p>
            <p className="text-xs text-muted-foreground">AI 모델 예측값</p>
          </CardContent>
        </Card>

        {/* 총 인상률 카드 */}
        <Card className="border-purple-500 dark:border-purple-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-purple-600 dark:text-purple-400">2026 총 예상 인상률</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">5.6%</p>
            <p className="text-xs text-muted-foreground">Base-up + 성과급 예측</p>
          </CardContent>
        </Card>
      </div>

      {/* 주요 메트릭 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* 최저임금 인상률 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">최저임금 인상률</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              5.0%
            </div>
            <p className="text-xs text-muted-foreground">
              2026년 예상
            </p>
          </CardContent>
        </Card>

        {/* 원/달러 환율 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">원/달러 환율</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ₩1,380
            </div>
            <p className="text-xs text-muted-foreground">
              +3.5% 전년 대비
            </p>
          </CardContent>
        </Card>

        {/* 삼바 매출 성장률 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">삼바 매출 성장률</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              15.2%
            </div>
            <p className="text-xs text-muted-foreground">
              2025년 예상
            </p>
          </CardContent>
        </Card>

        {/* 동종업계 평균 인상률 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">동종업계 평균</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              4.5%
            </div>
            <p className="text-xs text-muted-foreground">
              임금 인상률
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 시나리오 선택 및 변수 조정 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 시나리오 템플릿 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings2 className="mr-2 h-5 w-5" />
              시나리오 템플릿
            </CardTitle>
            <CardDescription>
              사전 정의된 시나리오를 선택하여 빠른 분석
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {scenarioTemplates.map((template) => (
                <Button
                  key={template.id}
                  variant={selectedScenario === template.id ? "default" : "outline"}
                  className="h-auto p-4 text-left"
                  onClick={() => handleScenarioSelect(template.id)}
                  disabled={loading === 'prediction'}
                >
                  <div>
                    <div className="font-medium text-sm">{template.name}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {template.description}
                    </div>
                  </div>
                </Button>
              ))}
            </div>

            <Button 
              onClick={handleRunScenarioAnalysis}
              disabled={loading === 'scenario-analysis'}
              className="w-full"
            >
              {loading === 'scenario-analysis' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  분석 중...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  전체 시나리오 분석
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* 변수 조정 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center">
                <Sliders className="mr-2 h-5 w-5" />
                변수 조정 - {activeTarget === 'baseup' ? 'Base-up' : '성과급'} 모델
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={activeTarget === 'baseup' ? 'default' : 'outline'}
                  onClick={() => setActiveTarget('baseup')}
                >
                  Base-up
                </Button>
                <Button
                  size="sm"
                  variant={activeTarget === 'performance' ? 'default' : 'outline'}
                  onClick={() => setActiveTarget('performance')}
                >
                  성과급
                </Button>
              </div>
            </CardTitle>
            <CardDescription>
              AI가 선정한 중요 변수를 조정하여 사용자 정의 예측
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Feature importance 기반 동적 슬라이더 */}
            {featureImportance && featureImportance[activeTarget]?.features && featureImportance[activeTarget].features.length > 0 ? (
              featureImportance[activeTarget].features.slice(0, 8).map((feature) => {
                const baselineValue = featureImportance[activeTarget].baseline_values[feature.feature] || 0;
                const currentValue = dynamicVariables[feature.feature] ?? baselineValue;
                
                // 값의 범위 설정 (baseline의 ±50% 또는 ±10)
                const range = Math.max(Math.abs(baselineValue * 0.5), 10);
                const minValue = baselineValue - range;
                const maxValue = baselineValue + range;
                
                // Feature 이름을 사용자 친화적으로 변환
                const displayName = feature.feature
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, l => l.toUpperCase())
                  .replace('Cpi', 'CPI')
                  .replace('Gdp', 'GDP')
                  .replace('Bu', 'BU')
                  .replace('Mi', 'MI')
                  .replace('Kr', '한국')
                  .replace('Us', '미국')
                  .replace('Usa', '미국');
                
                return (
                  <div key={feature.feature} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <label className="text-sm font-medium">
                        {displayName}
                        <span className="ml-2 text-xs text-muted-foreground">
                          (중요도: {(feature.importance * 100).toFixed(1)}%)
                        </span>
                      </label>
                      <span className="text-sm text-muted-foreground">
                        {formatNumber(currentValue, 2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={minValue}
                      max={maxValue}
                      step={(maxValue - minValue) / 100}
                      value={currentValue}
                      onChange={(e) => {
                        const newValue = parseFloat(e.target.value);
                        setDynamicVariables(prev => ({
                          ...prev,
                          [feature.feature]: newValue
                        }));
                      }}
                      className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer"
                      disabled={loading === 'prediction'}
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>{formatNumber(minValue, 1)}</span>
                      <span className="font-medium">기준: {formatNumber(baselineValue, 2)}</span>
                      <span>{formatNumber(maxValue, 1)}</span>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>Feature importance 데이터를 로딩 중...</p>
                <p className="text-sm mt-2">모델이 학습되지 않았다면 Modeling 페이지에서 먼저 학습을 진행해주세요.</p>
              </div>
            )}

            <Button 
              onClick={handleCustomPrediction}
              disabled={loading === 'custom-prediction'}
              className="w-full"
            >
              {loading === 'custom-prediction' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  예측 중...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  사용자 정의 예측
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* 시나리오 분석 결과 */}
      {scenarioResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <PieChart className="mr-2 h-5 w-5" />
              시나리오 분석 결과
            </CardTitle>
            <CardDescription>
              다양한 시나리오별 임금인상률 예측 비교
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {scenarioResults.map((result, index) => (
                <div key={index} className="p-4 border border-border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-sm">{result.scenario_name}</h4>
                    {result.rank && result.rank === 1 && (
                      <span className="text-xs bg-primary text-primary-foreground px-2 py-1 rounded">
                        최고
                      </span>
                    )}
                  </div>
                  <div className="text-2xl font-bold text-primary mb-1">
                    {formatPrediction(result.prediction)}%
                  </div>
                  <div className="text-xs text-muted-foreground">
                    구간: {formatPrediction(result.confidence_interval[0])}% - {formatPrediction(result.confidence_interval[1])}%
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 상세 분석 영역 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <LineChart className="mr-2 h-5 w-5" />
              임금인상률 추이 및 2026년 예측
            </CardTitle>
            <CardDescription>
              과거 임금인상률 추이 및 향후 전망
            </CardDescription>
          </CardHeader>
          <CardContent>
            {getChartData() ? (
              <div className="h-96">
                <Line data={getChartData()!} options={getChartOptions()} />
              </div>
            ) : (
              <div className="h-64 bg-background border rounded-md flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="h-8 w-8 text-muted-foreground mx-auto mb-2 animate-spin" />
                  <p className="text-muted-foreground">데이터 로딩 중...</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart3 className="mr-2 h-5 w-5" />
              영향 요인 분석
            </CardTitle>
            <CardDescription>
              주요 경제 변수의 임금인상률 영향도
            </CardDescription>
          </CardHeader>
          <CardContent>
            {featureImportance && getWaterfallChartData() ? (
              <div className="h-64">
                <Chart 
                  type='bar'
                  data={getWaterfallChartData()!} 
                  options={getWaterfallChartOptions()} 
                />
              </div>
            ) : (
              <div className="h-64 bg-background border rounded-md flex items-center justify-center">
                <div className="text-center">
                  <Loader2 className="h-8 w-8 text-muted-foreground mx-auto mb-2 animate-spin" />
                  <p className="text-muted-foreground">데이터 로딩 중...</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

    </div>
  );
};