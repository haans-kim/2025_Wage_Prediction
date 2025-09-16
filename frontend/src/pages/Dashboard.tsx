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
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import {
  TrendingUp,
  BarChart3,
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

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
  
  // 디바운스 타이머 state 추가 (선언을 위쪽으로 이동)
  const [debounceTimer, setDebounceTimer] = useState<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);
  
  // 컴포넌트 언마운트 시 타이머 정리
  useEffect(() => {
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  }, [debounceTimer]);

  const loadDashboardData = async () => {
    setLoading('initial');
    setError(null);

    try {
      console.log('Loading dashboard data from:', API_BASE_URL);
      // Strategic API endpoints 사용 + 실제 모델의 Feature Importance
      const [scenariosRes, historicalRes, sensitivityRes, featureRes, modelFeatureRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/strategic/scenarios`).then(r => r.json()).catch(err => { console.error('Scenarios error:', err); return { scenarios: [] }; }),
        fetch(`${API_BASE_URL}/api/strategic/historical`).then(r => r.json()).catch(err => { console.error('Historical error:', err); return { data: [] }; }),
        fetch(`${API_BASE_URL}/api/strategic/sensitivity`).then(r => r.json()).catch(err => { console.error('Sensitivity error:', err); return { analysis: [] }; }),
        fetch(`${API_BASE_URL}/api/strategic/feature-importance`).then(r => r.json()).catch(err => { console.error('Feature importance error:', err); return { features: [] }; }),
        fetch(`${API_BASE_URL}/api/dashboard/model-feature-importance`).then(r => r.json()).catch(err => { console.error('Model feature importance error:', err); return null; })
      ]);

      console.log('API responses:', { scenariosRes, historicalRes, sensitivityRes, featureRes, modelFeatureRes });

      // 시나리오 템플릿 형식으로 변환
      const templates = scenariosRes.scenarios?.map((s: any) => ({
        id: s.name.toLowerCase().replace(' ', '_'),
        name: s.name,
        description: s.description || '',
        variables: s.variables || {}
      })) || [];
      setScenarioTemplates(templates);

      // Simple Regression 모델의 9개 조정 가능한 변수 (전년도 인상률 제외)
      const variables = [
        { name: 'minimum_wage', display_name: '최저임금 인상률', description: '최저임금 인상률', min_value: 0, max_value: 5, unit: '%', current_value: 1.7 },
        { name: 'us_eci', display_name: '미국 임금비용지수', description: '미국 ECI', min_value: 2, max_value: 6, unit: '%', current_value: 3.9 },
        { name: 'gdp_growth', display_name: 'GDP 성장률', description: 'GDP 성장률', min_value: 0, max_value: 4, unit: '%', current_value: 1.8 },
        { name: 'revenue_growth', display_name: '매출액 증가율', description: '매출액 증가율', min_value: -5, max_value: 10, unit: '%', current_value: 3.0 },
        { name: 'operating_margin', display_name: '영업이익률', description: '영업이익률', min_value: 0, max_value: 15, unit: '%', current_value: 5.5 },
        { name: 'cpi', display_name: '소비자물가상승률', description: '소비자물가상승률', min_value: 0, max_value: 5, unit: '%', current_value: 1.9 },
        { name: 'unemployment_rate', display_name: '실업률', description: '실업률', min_value: 2, max_value: 6, unit: '%', current_value: 3.8 },
        { name: 'interest_rate', display_name: '시장금리', description: '시장금리', min_value: 1, max_value: 5, unit: '%', current_value: 2.75 },
        { name: 'exchange_rate', display_name: '원달러환율', description: '원달러환율', min_value: 1000, max_value: 1500, unit: '원', current_value: 1350 }
      ];
      setAvailableVariables(variables);

      // 현재 값으로 customVariables 초기화
      const currentValues = variables.reduce((acc, v) => ({ ...acc, [v.name]: v.current_value }), {});
      setCustomVariables(currentValues);

      // 경제 지표 설정
      setEconomicIndicators({
        current_gdp_growth: {
          value: 2.2,
          change: '+0.3%',
          status: 'stable',
          last_updated: new Date().toISOString()
        },
        current_inflation: {
          value: 1.8,
          change: '-0.2%',
          status: 'stable',
          last_updated: new Date().toISOString()
        },
        current_unemployment: {
          value: 3.5,
          change: '+0.1%',
          status: 'stable',
          last_updated: new Date().toISOString()
        }
      });

      // Historical data를 트렌드 데이터로 변환
      if (historicalRes.data && historicalRes.data.length > 0) {
        const trendData = {
          trend_data: historicalRes.data.map((d: any) => ({
            year: d.year,
            value: d.actual_increase * 100,
            base_up: d.base_up ? d.base_up * 100 : null,
            type: d.year === 2026 ? 'prediction' : 'actual'
          }))
        };
        setTrendData(trendData);
      }

      // Feature importance 설정 (실제 모델 우선, 없으면 기본값)
      if (modelFeatureRes && modelFeatureRes.feature_importance && modelFeatureRes.feature_importance.length > 0) {
        // 실제 모델의 Feature Importance 사용
        console.log('Using actual model feature importance');
        setFeatureImportance(modelFeatureRes);
      } else if (featureRes.features && featureRes.features.length > 0) {
        // 전략적 대시보드의 기본값 사용
        console.log('Using strategic default feature importance');
        setFeatureImportance({
          feature_importance: featureRes.features.map((f: any) => ({
            feature: f.name,
            feature_korean: f.korean_name || f.name,
            importance: f.importance
          }))
        });
      }

      // 초기 예측 수행
      try {
        console.log('Making prediction with values:', currentValues);
        console.log('API URL:', `${API_BASE_URL}/api/strategic/predict`);
        const response = await fetch(`${API_BASE_URL}/api/strategic/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(currentValues)
        });
        const predictionRes = await response.json();
        console.log('Strategic prediction result:', predictionRes);

        // Strategic API 응답 형식에 맞게 변환 (result 안의 prediction 사용)
        const result = predictionRes.result || predictionRes;
        const prediction = result.prediction || {};

        const formattedPrediction: PredictionResult = {
          prediction: (prediction.total || 0) / 100,  // 4.0% -> 0.04
          base_up_rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
          performance_rate: 0.021,  // 2.1%로 고정
          confidence_interval: [
            ((prediction.total || 0) - 0.5) / 100,
            ((prediction.total || 0) + 0.5) / 100
          ] as [number, number],
          confidence_level: result.confidence?.overall || 0.85,
          input_variables: currentValues,
          breakdown: {
            base_up: {
              rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
              percentage: (prediction.total || 0) - 2.1,
              description: 'Base-up 인상률',
              calculation: ''
            },
            performance: {
              rate: 0.021,  // 2.1%로 고정
              percentage: 2.1,
              description: '성과급 인상률',
              calculation: ''
            },
            total: {
              rate: (prediction.total || 0) / 100,
              percentage: prediction.total || 0,
              description: '총 인상률'
            }
          }
        };
        setCurrentPrediction(formattedPrediction);
      } catch (predError) {
        console.error('Strategic prediction failed:', predError);
      }
    } catch (error: any) {
      console.error('Dashboard data loading failed:', error);
      setError('대시보드 데이터를 불러오는 중 오류가 발생했습니다.');
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
        // Strategic predict endpoint 사용
        const response = await fetch(`${API_BASE_URL}/api/strategic/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(template.variables)
        });
        const predictionRes = await response.json();

        const result = predictionRes.result || predictionRes;
        const prediction = result.prediction || {};

        const formattedPrediction: PredictionResult = {
          prediction: (prediction.total || 0) / 100,
          base_up_rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
          performance_rate: 0.021,  // 2.1%로 고정
          confidence_interval: [
            ((prediction.total || 0) - 0.5) / 100,
            ((prediction.total || 0) + 0.5) / 100
          ] as [number, number],
          confidence_level: result.confidence?.overall || 0.85,
          input_variables: template.variables,
          breakdown: {
            base_up: {
              rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
              percentage: (prediction.total || 0) - 2.1,
              description: 'Base-up 인상률',
              calculation: ''
            },
            performance: {
              rate: 0.021,  // 2.1%로 고정
              percentage: 2.1,
              description: '성과급 인상률',
              calculation: ''
            },
            total: {
              rate: (prediction.total || 0) / 100,
              percentage: prediction.total || 0,
              description: '총 인상률'
            }
          }
        };
        setCurrentPrediction(formattedPrediction);
      } catch (error) {
        setError(error instanceof Error ? error.message : '예측 중 오류가 발생했습니다.');
      } finally {
        setLoading(null);
      }
    }
  };

  const handleVariableChange = (variableName: string, value: number) => {
    const newVariables = {
      ...customVariables,
      [variableName]: value
    };
    setCustomVariables(newVariables);
    
    // 슬라이더 변경 시 자동으로 예측 실행 (디바운스 적용)
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    
    const timer = setTimeout(() => {
      handleCustomPrediction(newVariables);
    }, 500); // 500ms 디바운스
    
    setDebounceTimer(timer);
  };

  const handleCustomPrediction = async (variables?: Record<string, number>) => {
    setLoading('custom-prediction');
    setError(null);

    try {
      const variablesToUse = variables || customVariables;

      // Simple Regression predict endpoint 사용
      const response = await fetch(`${API_BASE_URL}/api/strategic/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          year: 2026,
          scenario: 'custom',
          custom_params: variablesToUse  // custom_params로 전달
        })
      });
      const predictionRes = await response.json();

      // Strategic API 응답 형식에 맞게 변환
      const result = predictionRes.result || predictionRes;
      const prediction = result.prediction || {};

      const formattedPrediction: PredictionResult = {
        prediction: (prediction.total || 0) / 100,
        base_up_rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
        performance_rate: 0.021,  // 2.1%로 고정
        confidence_interval: [
          ((prediction.total || 0) - 0.5) / 100,
          ((prediction.total || 0) + 0.5) / 100
        ] as [number, number],
        confidence_level: result.confidence || 0.85,
        input_variables: variablesToUse,
        breakdown: {
          base_up: {
            rate: ((prediction.total || 0) / 100) - 0.021,  // 총인상률 - 성과인상률(2.1%)
            percentage: (prediction.total || 0) - 2.1,
            description: 'Base-up 인상률',
            calculation: ''
          },
          performance: {
            rate: 0.021,  // 2.1%로 고정
            percentage: 2.1,
            description: '성과급 인상률',
            calculation: ''
          },
          total: {
            rate: (prediction.total || 0) / 100,
            percentage: prediction.total || 0,
            description: '총 인상률'
          }
        }
      };
      setCurrentPrediction(formattedPrediction);
    } catch (error) {
      setError(error instanceof Error ? error.message : '사용자 정의 예측 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  // 버튼 클릭용 래퍼 함수
  const handleCustomPredictionClick = () => {
    handleCustomPrediction();
  };

  const handleRunScenarioAnalysis = async () => {
    setLoading('scenario-analysis');
    setError(null);

    try {
      // Strategic scenarios endpoint 사용
      const response = await fetch(`${API_BASE_URL}/api/strategic/scenarios`);
      const scenariosRes = await response.json();

      const results = scenariosRes.scenarios?.map((scenario: any, index: number) => ({
        scenario_name: scenario.name,
        prediction: scenario.total_increase,
        confidence_interval: [
          scenario.total_increase - 0.005,
          scenario.total_increase + 0.005
        ],
        rank: index === 0 ? 1 : (index === scenariosRes.scenarios.length - 1 ? 3 : 2)
      })) || [];

      setScenarioResults(results);
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
    // 정확한 반올림 처리 - 소수점 첫째자리에서 반올림
    const percentage = num * 100;
    return Math.round(percentage * 10) / 10;  // 소수점 첫째자리에서 정확한 반올림
  };

  const getTopImportantVariables = () => {
    // Simple Regression 모델의 9개 변수를 중요도 순으로 반환 (전년도 인상률 제외)
    // Feature Importance API에서 받은 순서대로 사용
    if (!featureImportance || !featureImportance.feature_importance) {
      // Feature Importance가 없으면 기본 순서로 반환
      return availableVariables;
    }

    // Feature 이름을 Dashboard 변수 이름으로 매핑 (Simple Regression 모델)
    const featureToVariableMap: { [key: string]: string } = {
      'minimum_wage_adjustment': 'minimum_wage',  // 최저임금 조정효과
      'us_eci': 'us_eci',                        // 미국 임금비용지수
      'gdp_adjustment': 'gdp_growth',            // GDP 성장률 조정
      'revenue_growth': 'revenue_growth',        // 매출액 증가율
      'operating_margin': 'operating_margin',    // 영업이익률
      'cpi': 'cpi',                             // 소비자물가상승률
      'unemployment_rate': 'unemployment_rate',  // 실업률
      'interest_rate': 'interest_rate',         // 시장금리
      'exchange_rate': 'exchange_rate'          // 원달러환율
    };

    // Feature Importance 순서대로 변수 매핑
    interface MappedFeature {
      variable: Variable;
      importance: number;
      feature: string;
    }
    const mappedFeatures: MappedFeature[] = [];

    for (const featureItem of featureImportance.feature_importance) {
      const featureName = featureItem.feature || featureItem.name;
      // previous_year_increase는 고정값이므로 제외
      if (featureName === 'previous_year_increase') continue;

      const variableName = featureToVariableMap[featureName];

      if (variableName) {
        const variable = availableVariables.find(v => v.name === variableName);
        if (variable) {
          mappedFeatures.push({
            variable,
            importance: featureItem.importance,
            feature: featureName
          });
        }
      }
    }

    // 중요도 순으로 정렬 (이미 정렬되어 있지만 확실하게)
    mappedFeatures.sort((a, b) => b.importance - a.importance);

    // 매핑된 변수들 반환
    const importantVariables: Variable[] = mappedFeatures.map(item => item.variable);

    // 매핑되지 않은 변수가 있으면 추가 (순서 유지)
    for (const variable of availableVariables) {
      if (!importantVariables.some(v => v.name === variable.name)) {
        importantVariables.push(variable);
      }
    }

    console.log('🔍 Simple Regression Variables:');
    console.log('Feature importance:', featureImportance?.feature_importance?.slice(0, 9));
    console.log('Mapped variables:', importantVariables.map((v, i) => `${i+1}. ${v.display_name} (${v.name})`));

    return importantVariables;
  };

  const getChartData = () => {
    if (!trendData || !trendData.trend_data) return null;

    const labels = trendData.trend_data.map((d: any) => d.year);
    
    // 총 인상률 (2026년은 현재 예측값으로 덮어쓰기)
    const totalData = trendData.trend_data.map((d: any) => {
      if (d.year === 2026 && currentPrediction) {
        // 현재 예측값을 퍼센트로 변환하여 사용
        return currentPrediction.prediction * 100;
      }
      return d.value;
    });
    
    // Base-up 데이터 (2026년은 현재 예측값으로 덮어쓰기)
    const baseupData = trendData.trend_data.map((d: any) => {
      if (d.year === 2026 && currentPrediction && currentPrediction.base_up_rate) {
        // 현재 Base-up 예측값을 퍼센트로 변환하여 사용
        return currentPrediction.base_up_rate * 100;
      }
      return d.base_up;
    });
    const hasBaseupData = baseupData.some((v: any) => v !== null && v !== undefined);
    
    // 2026년 예측값 인덱스 찾기
    const prediction2026Index = trendData.trend_data.findIndex((d: any) => d.year === 2026);
    
    const datasets = [];
    
    // Base-up 데이터가 있으면 먼저 추가
    if (hasBaseupData) {
      datasets.push({
        label: 'Base-up',
        data: baseupData,
        borderColor: 'rgb(59, 130, 246)', // 파란색
        backgroundColor: 'rgba(59, 130, 246, 0.15)',
        borderWidth: 2.5,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        pointBorderColor: 'rgb(59, 130, 246)',
        pointBorderWidth: 1,
        fill: false,
        datalabels: {
          display: true,
          align: 'bottom' as const,
          offset: 5,
          formatter: (value: any) => value ? value.toFixed(1) : '',
          font: {
            size: 10,
            weight: 'bold' as const
          },
          color: 'rgb(59, 130, 246)'
        }
      });
    }
    
    // 총 인상률 차트
    datasets.push({
          label: '총 인상률',
          data: totalData,
          borderColor: (ctx: any) => {
            // 2026년 구간은 빨간색으로 표시
            if (ctx.type === 'segment' && ctx.p0DataIndex === prediction2026Index - 1) {
              return 'rgb(239, 68, 68)';
            }
            return 'rgb(34, 197, 94)'; // 기본 초록색
          },
          backgroundColor: 'rgba(34, 197, 94, 0.15)',
          borderWidth: 2.5,
          tension: 0.4,
          pointRadius: (ctx: any) => {
            // 2026년 예측값은 더 큰 포인트로 표시
            return ctx.dataIndex === prediction2026Index ? 8 : 4;
          },
          pointHoverRadius: (ctx: any) => {
            return ctx.dataIndex === prediction2026Index ? 10 : 6;
          },
          pointBackgroundColor: (ctx: any) => {
            // 2026년 예측값은 빨간색으로 표시
            return ctx.dataIndex === prediction2026Index ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)';
          },
          pointBorderColor: (ctx: any) => {
            return ctx.dataIndex === prediction2026Index ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)';
          },
          pointBorderWidth: (ctx: any) => {
            return ctx.dataIndex === prediction2026Index ? 3 : 1;
          },
          fill: false,
          segment: {
            borderDash: (ctx: any) => {
              // 2025-2026 구간은 점선으로 표시
              return ctx.p0DataIndex === prediction2026Index - 1 ? [5, 5] : undefined;
            }
          },
          datalabels: {
            display: true,
            align: 'top' as const,
            offset: 5,
            formatter: (value: any) => value ? value.toFixed(1) : '',
            font: {
              size: 10,
              weight: 'bold' as const
            },
            color: (ctx: any) => {
              return ctx.dataIndex === prediction2026Index ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)';
            }
          }
        });
    
    return {
      labels,
      datasets
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
      datalabels: {
        display: false // 전역적으로 비활성화 (각 dataset에서 개별 설정)
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            if (context.dataset.label?.includes('신뢰구간')) return '';
            const value = context.parsed.y;
            const year = trendData.trend_data[context.dataIndex]?.year;
            
            if (year === 2026) {
              return `🎯 2026년 예측값: ${value.toFixed(1)}%`;
            }
            return `${year}년 실적: ${value.toFixed(1)}%`;
          },
          afterLabel: (context: any) => {
            const dataPoint = trendData.trend_data[context.dataIndex];
            if (dataPoint?.type === 'prediction' && 
                dataPoint.confidence_lower !== undefined && 
                dataPoint.confidence_upper !== undefined) {
              return `신뢰구간: ${dataPoint.confidence_lower.toFixed(1)}% - ${dataPoint.confidence_upper.toFixed(1)}%`;
            }
            return '';
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
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
    console.log('getWaterfallChartData called');
    console.log('featureImportance:', featureImportance);
    console.log('currentPrediction:', currentPrediction);
    
    if (!featureImportance || !featureImportance.feature_importance) {
      console.log('Returning null - missing feature importance data');
      return null;
    }

    const data = featureImportance.feature_importance;

    // 모든 10개 변수를 표시 (others 없이)
    const allFeatures = data.slice(0, 10);  // 최대 10개까지만

    // 각 feature의 기여도를 계산
    interface FeatureContribution {
      feature: string;
      feature_korean: string;
      contribution: number;
      importance: number;
      value: number;
    }

    const featureContributions: FeatureContribution[] = allFeatures.map((item: any, index: number) => {
      // importance 값을 그대로 사용 (모두 양수)
      const normalizedImportance = item.importance / allFeatures[0].importance;

      return {
        feature: item.feature || item.name,
        feature_korean: item.feature_korean || item.korean_name || item.feature || item.name,
        contribution: normalizedImportance * 2, // 시각화를 위해 스케일 조정
        importance: item.importance,
        value: item.importance // 표시용 원본 값
      };
    });

    // 기여도 순으로 정렬 (이미 정렬되어 있지만 확실하게)
    featureContributions.sort((a: FeatureContribution, b: FeatureContribution) => b.importance - a.importance);

    // 레이블과 데이터 준비
    const labels = featureContributions.map((d: FeatureContribution) => {
      return d.feature_korean; // 한글 이름 표시
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
        text: '주요 변수별 중요도 분석 (Regression Weights)',
        font: {
          size: 20,
          weight: 'bold' as const
        }
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const value = context.parsed.x;
            const dataIndex = context.dataIndex;
            const featureData = featureImportance?.feature_importance[dataIndex];
            if (featureData) {
              return `중요도: ${(featureData.importance * 100).toFixed(1)}%`;
            }
            return `기여도: ${value.toFixed(2)}`;
          }
        },
        bodyFont: {
          size: 14
        }
      },
      datalabels: {
        color: 'white',
        font: {
          weight: 'bold' as const,
          size: 14
        },
        anchor: 'center' as const,
        align: 'center' as const,
        formatter: (value: any, context: any) => {
          const dataIndex = context.dataIndex;
          const featureData = featureImportance?.feature_importance[dataIndex];
          if (featureData) {
            return `${(featureData.importance * 100).toFixed(1)}%`;
          }
          return '';
        }
      }
    },
    scales: {
      x: {
        beginAtZero: true,
        title: {
          display: true,
          text: '임금인상률 기여도 (%p)',
          font: {
            size: 14,
            weight: 500
          }
        },
        ticks: {
          callback: (value: any) => {
            const sign = value >= 0 ? '+' : '';
            return `${sign}${value}%`;
          },
          font: {
            size: 12
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
            size: 14,
            weight: 500
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
              {currentPrediction?.breakdown ? `${currentPrediction.breakdown.total.percentage.toFixed(1)}%` : 
               currentPrediction ? `${formatPrediction(currentPrediction.prediction, 1)}%` : '-.-%'}
            </div>
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
              {currentPrediction?.breakdown ? 
                `${(currentPrediction.breakdown.total.percentage - currentPrediction.breakdown.performance.percentage).toFixed(1)}%` : '-.-%'}
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
              {currentPrediction?.breakdown ? `${currentPrediction.breakdown.performance.percentage.toFixed(1)}%` : '-.-%'}
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
                <span className="font-medium">{economicIndicators.current_gdp_growth ? `${economicIndicators.current_gdp_growth.value}%` : '-%'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">인플레:</span>
                <span className="font-medium">{economicIndicators.current_inflation ? `${economicIndicators.current_inflation.value}%` : '-%'}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">실업률:</span>
                <span className="font-medium">{economicIndicators.current_unemployment ? `${economicIndicators.current_unemployment.value}%` : '-%'}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 변수 조정과 분석 차트를 2열로 배치 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 왼쪽: 변수 조정 (1/3 너비) */}
        <div className="lg:col-span-1">
          <Card className="h-full">
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
              {getTopImportantVariables().map((variable) => (
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
                onClick={handleCustomPredictionClick}
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

        {/* 오른쪽: 영향 요인 분석과 트렌드 분석 (2/3 너비) */}
        <div className="lg:col-span-2 space-y-6">
          {/* 영향 요인 분석 */}
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
              {(() => {
                const chartData = getWaterfallChartData();
                console.log('Chart data:', chartData);
                
                if (chartData) {
                  return (
                    <div className="h-96">
                      <Chart
                        type='bar'
                        data={chartData}
                        options={getWaterfallChartOptions()}
                      />
                    </div>
                  );
                } else {
                  return (
                    <div className="h-96 bg-background border rounded-md flex items-center justify-center">
                      <div className="text-center">
                        <Loader2 className="h-8 w-8 text-muted-foreground mx-auto mb-2 animate-spin" />
                        <p className="text-muted-foreground">데이터 로딩 중...</p>
                      </div>
                    </div>
                  );
                }
              })()}
            </CardContent>
          </Card>

          {/* 트렌드 분석 */}
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
        </div>
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
    </div>
  );
};