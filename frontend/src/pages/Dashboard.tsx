import React, { useState, useEffect } from 'react';
import { Line, Chart } from 'react-chartjs-2';
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
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  TrendingUp, 
  BarChart3, 
  Settings2, 
  Play,
  AlertTriangle,
  Loader2,
  Zap,
  Target,
  Activity,
  PieChart,
  LineChart,
  Sliders
} from 'lucide-react';
import { apiClient } from '../lib/api';

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
  Filler
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

interface PredictionResult {
  prediction: number;
  base_up_rate?: number;
  performance_rate?: number;
  confidence_interval: [number, number];
  confidence_level: number;
  input_variables: Record<string, number>;
  breakdown?: {
    base_up: {
      rate: number;
      percentage: number;
      description: string;
      calculation: string;
    };
    performance: {
      rate: number;
      percentage: number;
      description: string;
      calculation: string;
    };
    total: {
      rate: number;
      percentage: number;
      description: string;
    };
  };
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
  const [featureImportance, setFeatureImportance] = useState<any>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading('initial');
    setError(null);

    try {
      const [templatesRes, variablesRes, indicatorsRes, trendRes, featureRes] = await Promise.all([
        apiClient.getScenarioTemplates().catch(() => ({ templates: [] })),
        apiClient.getAvailableVariables().catch(() => ({ variables: [], current_values: {} })),
        apiClient.getEconomicIndicators().catch(() => ({ indicators: {} })),
        apiClient.getTrendData().catch(() => null),
        apiClient.getFeatureImportance('permutation', 10).catch(() => null)
      ]);

      setScenarioTemplates(templatesRes.templates || []);
      setAvailableVariables(variablesRes.variables || []);
      setEconomicIndicators(indicatorsRes.indicators || {});
      setTrendData(trendRes);
      setFeatureImportance(featureRes);
      console.log('Feature importance data:', featureRes);

      // 기본 시나리오로 초기 예측 수행
      if (variablesRes.current_values) {
        setCustomVariables(variablesRes.current_values);
        const predictionRes = await apiClient.predictWageIncrease(variablesRes.current_values);
        setCurrentPrediction(predictionRes);
      }
    } catch (error: any) {
      console.error('Dashboard data loading failed:', error);
      
      // 모델이 없는 경우 특별한 처리
      if (error?.response?.status === 404 && error?.response?.data?.detail?.error === "No trained model available") {
        setError('모델이 훈련되지 않았습니다. Analysis 페이지에서 먼저 모델을 훈련해주세요.');
      } else {
        setError('대시보드 데이터를 불러오는 중 오류가 발생했습니다.');
      }
    } finally {
      setLoading(null);
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
      const predictionRes = await apiClient.predictWageIncrease(customVariables);
      setCurrentPrediction(predictionRes);
    } catch (error) {
      setError(error instanceof Error ? error.message : '사용자 정의 예측 중 오류가 발생했습니다.');
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
    // 정확한 반올림 처리
    const percentage = num * 100;
    return percentage.toFixed(decimals);
  };

  const getChartData = () => {
    if (!trendData || !trendData.trend_data) return null;


    const labels = trendData.trend_data.map((d: any) => d.year);
    const values = trendData.trend_data.map((d: any) => d.value);
    
    // 신뢰구간 데이터 (예측 데이터만)
    const confidenceLower = trendData.trend_data.map((d: any) => 
      d.type === 'prediction' ? d.confidence_lower : d.value
    );
    const confidenceUpper = trendData.trend_data.map((d: any) => 
      d.type === 'prediction' ? d.confidence_upper : d.value
    );

    return {
      labels,
      datasets: [
        {
          label: '과거 임금인상률',
          data: values.map((v: number, i: number) => 
            trendData.trend_data[i].type === 'historical' ? v : null
          ),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          tension: 0.1,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: '2026년 예측',
          data: values.map((v: number, i: number) => 
            trendData.trend_data[i].type === 'prediction' ? v : null
          ),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderDash: [5, 5],
          pointRadius: 6,
          pointHoverRadius: 8,
          pointStyle: 'star',
        },
        {
          label: '신뢰구간 (상한)',
          data: confidenceUpper,
          borderColor: 'rgba(239, 68, 68, 0.3)',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          fill: '+1',
          borderWidth: 1,
          pointRadius: 0,
          hidden: false,
        },
        {
          label: '신뢰구간 (하한)',
          data: confidenceLower,
          borderColor: 'rgba(239, 68, 68, 0.3)',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          fill: false,
          borderWidth: 1,
          pointRadius: 0,
          hidden: false,
        }
      ]
    };
  };

  const getChartOptions = () => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          filter: (item: any) => !item.text.includes('신뢰구간')
        }
      },
      title: {
        display: true,
        text: '임금인상률 추이 및 2026년 예측',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            if (context.dataset.label?.includes('신뢰구간')) return;
            const value = context.parsed.y;
            const isHistorical = trendData.trend_data[context.dataIndex]?.type === 'historical';
            const label = isHistorical ? '과거 실적' : '예측값';
            return `${label}: ${value.toFixed(1)}%`;
          },
          afterLabel: (context: any) => {
            const dataPoint = trendData.trend_data[context.dataIndex];
            if (dataPoint?.type === 'prediction') {
              return `신뢰구간: ${dataPoint.confidence_lower.toFixed(1)}% - ${dataPoint.confidence_upper.toFixed(1)}%`;
            }
            return;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: '임금인상률 (%)'
        },
        ticks: {
          callback: (value: any) => `${value}%`
        }
      },
      x: {
        title: {
          display: true,
          text: '연도'
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    }
  });


  const getWaterfallChartData = () => {
    if (!featureImportance || !featureImportance.feature_importance || !currentPrediction) return null;

    const data = featureImportance.feature_importance;
    
    // 현재 예측값을 백분율로 변환
    
    // 상위 8개 주요 변수만 선택하고 나머지는 '기타'로 묶기
    const topFeatures = data.slice(0, 8);
    const otherFeatures = data.slice(8);
    
    // 전체 importance의 합
    
    // 각 feature의 기여도를 극대화하여 계산
    // 상위 3개는 양수, 나머지는 음수로 설정하여 대비 극대화
    interface FeatureContribution {
      feature: string;
      contribution: number;
      importance: number;
      value: number;
    }
    
    const featureContributions: FeatureContribution[] = topFeatures.map((item: any, index: number) => {
      // Permutation importance 값을 그대로 사용 (모두 양수)
      const normalizedImportance = item.importance / topFeatures[0].importance;
      
      return {
        feature: item.feature,
        contribution: normalizedImportance * 2, // 시각화를 위해 스케일 조정
        importance: item.importance,
        value: item.importance // 표시용 원본 값
      };
    });
    
    // 기타 항목
    if (otherFeatures.length > 0) {
      const othersImportance = otherFeatures.reduce((sum: number, item: any) => sum + item.importance, 0) / otherFeatures.length;
      const normalizedOthers = othersImportance / topFeatures[0].importance;
      featureContributions.push({
        feature: 'others',
        contribution: normalizedOthers * 2,
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
          backgroundColor: 'rgba(59, 130, 246, 0.8)',
          borderColor: 'rgb(59, 130, 246)',
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
        text: '주요 변수별 중요도 분석 (Permutation Importance)',
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
                  onClick={() => window.location.href = '/analysis'}
                  className="mt-2"
                >
                  Analysis 페이지로 이동
                </Button>
              )}
            </div>
          </AlertDescription>
        </Alert>
      )}

      {/* 주요 메트릭 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* 현재 예측 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">2026년 총 인상률</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {currentPrediction ? `${formatPrediction(currentPrediction.prediction, 1)}%` : '-.-%'}
            </div>
            <p className="text-xs text-muted-foreground">
              {currentPrediction && (
                `신뢰구간: ${formatPrediction(currentPrediction.confidence_interval[0], 1)}% - ${formatPrediction(currentPrediction.confidence_interval[1], 1)}%`
              )}
            </p>
          </CardContent>
        </Card>

        {/* Base-up */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Base-up</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">
              {currentPrediction?.breakdown ? `${formatPrediction(currentPrediction.breakdown.base_up.rate, 1)}%` : '-.-%'}
            </div>
            {currentPrediction?.breakdown && (
              <div className="text-xs text-muted-foreground mt-1">
                <div className="mb-1">기본 인상분</div>
                <div className="font-mono text-[10px]">= 총 인상률 - 성과 인상률</div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 성과 인상률 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">성과 인상률</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {currentPrediction?.breakdown ? `${formatPrediction(currentPrediction.breakdown.performance.rate, 1)}%` : '-.-%'}
            </div>
            {currentPrediction?.breakdown && (
              <div className="text-xs text-muted-foreground mt-1">
                <div className="mb-1">과거 10년 성과급 추세 예측</div>
                <div className="font-mono text-[10px]">선형회귀 분석</div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 경제 지표 */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">주요 경제지표</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">GDP:</span>
                <span className="font-medium">{economicIndicators.gdp_growth?.value || '-'}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">인플레:</span>
                <span className="font-medium">{economicIndicators.inflation_rate?.value || '-'}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">실업률:</span>
                <span className="font-medium">{economicIndicators.unemployment_rate?.value || '-'}%</span>
              </div>
            </div>
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
            <CardTitle className="flex items-center">
              <Sliders className="mr-2 h-5 w-5" />
              변수 조정
            </CardTitle>
            <CardDescription>
              경제 변수를 직접 조정하여 사용자 정의 예측
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {availableVariables.slice(0, 5).map((variable) => (
              <div key={variable.name} className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">{variable.display_name}</label>
                  <span className="text-sm text-muted-foreground">
                    {formatNumber(customVariables[variable.name] || variable.current_value, 1)}{variable.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={variable.min_value}
                  max={variable.max_value}
                  step={0.1}
                  value={customVariables[variable.name] || variable.current_value}
                  onChange={(e) => handleVariableChange(variable.name, parseFloat(e.target.value))}
                  className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{variable.min_value}{variable.unit}</span>
                  <span>{variable.max_value}{variable.unit}</span>
                </div>
              </div>
            ))}

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
              트렌드 분석
            </CardTitle>
            <CardDescription>
              과거 임금인상률 추이 및 향후 전망
            </CardDescription>
          </CardHeader>
          <CardContent>
            {trendData && getChartData() ? (
              <div className="h-64">
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