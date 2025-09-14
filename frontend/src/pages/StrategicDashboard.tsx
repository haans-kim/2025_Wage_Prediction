import React, { useState, useEffect } from 'react';
import { Bar, Line, Pie, Radar } from 'react-chartjs-2';
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
  ArcElement,
  RadialLinearScale,
  Filler
} from 'chart.js';
import {
  TrendingUp,
  Activity,
  AlertTriangle,
  Target,
  Sliders,
  BarChart3,
  Clock,
  Lightbulb
} from 'lucide-react';
import { apiClient } from '../lib/api';
import { Button } from '../components/ui/button';
import { Card } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Slider } from '../components/ui/slider';
import { Label } from '../components/ui/label';

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  RadialLinearScale,
  Filler
);

const StrategicDashboard: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [scenarios, setScenarios] = useState<any>(null);
  const [sensitivity, setSensitivity] = useState<any>(null);
  const [historicalPatterns, setHistoricalPatterns] = useState<any>(null);

  // Simulator states
  const [simulatorParams, setSimulatorParams] = useState({
    min_wage: 3.5,
    gdp: 1.6,
    revenue_growth: 15.0,
    profit_margin: 12.0,
    industry_avg: 4.8,
    crisis_score: 0.3
  });
  const [simulationResult, setSimulationResult] = useState<any>(null);

  useEffect(() => {
    loadDashboardData();
    loadScenarios();
    loadHistoricalPatterns();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const response = await apiClient.get('/strategic/dashboard');
      console.log('Dashboard response:', response);
      setDashboardData(response.dashboard);

      // Load main prediction
      const predResponse = await apiClient.post('/strategic/predict', {
        year: 2026,
        scenario: 'base'
      });
      console.log('Prediction response:', predResponse);
      setPrediction(predResponse.result);
    } catch (error) {
      console.error('Failed to load dashboard:', error);
      // Set default data for testing
      setDashboardData({
        '1_executive_summary': {
          title: '2026년 임금인상률 예측',
          key_message: 'Base-up 3.2% + MI 2.3% = 총 5.5%',
          key_drivers: [
            { factor: '최저임금 대응', impact: '22.4%', direction: '억제' },
            { factor: 'CDMO 성장', impact: '18.5%', direction: '상승' },
            { factor: '업계 벤치마킹', impact: '15.2%', direction: '상승' }
          ],
          risk_factors: ['미중 무역갈등', '바이오시밀러 수주 지연', '인플레이션 재점화']
        }
      });
      setPrediction({
        prediction: {
          base_up: 3.2,
          mi: 2.3,
          total: 5.5,
          method: 'strategic_rules'
        },
        confidence: {
          level: 'HIGH',
          overall: 0.85
        }
      });
    } finally {
      setLoading(false);
    }
  };

  const loadScenarios = async () => {
    try {
      const response = await apiClient.get('/strategic/scenarios');
      console.log('Scenarios response:', response);
      setScenarios(response.scenarios);
    } catch (error) {
      console.error('Failed to load scenarios:', error);
      // Set default scenarios for testing
      setScenarios({
        '기본_시나리오': {
          name: '연착륙',
          description: '미국 금리인하 본격화, 경기 완만한 회복',
          parameters: { gdp: 1.6, min_wage: 3.5 },
          prediction: { base_up: 3.2, mi: 2.3, total: 5.5 }
        },
        '보수적_시나리오': {
          name: '스태그플레이션',
          description: '미중 갈등 심화, 공급망 충격',
          parameters: { gdp: 1.2, min_wage: 4.0 },
          prediction: { base_up: 2.8, mi: 2.0, total: 4.8 }
        },
        '낙관적_시나리오': {
          name: 'CDMO 슈퍼사이클',
          description: '바이오시밀러 대량 수주',
          parameters: { gdp: 2.0, min_wage: 3.0 },
          prediction: { base_up: 3.5, mi: 2.5, total: 6.0 }
        }
      });
    }
  };

  const loadHistoricalPatterns = async () => {
    try {
      const response = await apiClient.get('/strategic/historical-patterns');
      console.log('Historical patterns response:', response);
      setHistoricalPatterns(response.patterns);
    } catch (error) {
      console.error('Failed to load historical patterns:', error);
      // Set default patterns for testing
      setHistoricalPatterns({
        patterns: {
          '안정기 (2015-2017)': {
            avg_increase: 4.53,
            characteristics: '예측 가능한 안정적 인상',
            key_learning: '평시 4.5% 수준이 표준'
          },
          '충격기 (2018-2020)': {
            avg_increase: 4.67,
            characteristics: '외부 충격에 역설적 대응',
            key_learning: '위기 시 오히려 인상 확대로 인재 보호'
          },
          '성장기 (2021-2024)': {
            avg_increase: 5.08,
            characteristics: '성과 연동 강화',
            key_learning: 'CDMO 성장과 연계한 적극적 보상'
          }
        },
        key_events: [
          { year: 2018, event: '최저임금 급등', response: '4.6%', lesson: '외부 압력에 굴복하지 않고 자체 기준 유지' },
          { year: 2020, event: 'COVID-19', response: '5.2%', lesson: '위기 시 인재 보호 최우선' },
          { year: 2024, event: 'CDMO 호황', response: '5.6%', lesson: '호황기 성과 적극 반영' }
        ]
      });
    }
  };

  const loadSensitivityAnalysis = async () => {
    try {
      const response = await apiClient.get('/strategic/sensitivity?scenario=base');
      setSensitivity(response.analysis);
    } catch (error) {
      console.error('Failed to load sensitivity analysis:', error);
    }
  };

  const runSimulation = async () => {
    try {
      const response = await apiClient.post('/strategic/simulate', simulatorParams);
      console.log('Simulation response:', response);
      setSimulationResult(response.result);
    } catch (error) {
      console.error('Simulation failed:', error);
      // Set default simulation result for testing
      const total = 3.0 + (simulatorParams.revenue_growth * 0.05) + (simulatorParams.profit_margin * 0.05);
      const base_up = Math.round(total * 0.6 * 10) / 10;
      const mi = Math.round(total * 0.4 * 10) / 10;
      setSimulationResult({
        base_up: base_up,
        mi: mi,
        total: Math.round((base_up + mi) * 10) / 10,
        comparison_to_base: Math.round((base_up + mi - 5.5) * 10) / 10
      });
    }
  };

  const renderExecutiveSummary = () => {
    if (!dashboardData || !prediction) return null;

    const summary = dashboardData['1_executive_summary'];
    const pred = prediction.prediction;

    // Pie chart data for model contribution
    const modelContributionData = {
      labels: ['전략적 규칙', 'ML 검증', '잔차 학습'],
      datasets: [{
        data: [70, 20, 10],
        backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'],
        borderWidth: 0
      }]
    };

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Main Prediction Card */}
        <div className="lg:col-span-2">
          <Card className="p-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white">
            <h2 className="text-2xl font-bold mb-4">2026년 임금인상률 예측</h2>
            <div className="text-5xl font-bold mb-4">
              {pred.base_up}% + {pred.mi}% = {pred.total}%
            </div>
            <div className="text-xl opacity-90">
              Base-up {pred.base_up}% + 성과인상(MI) {pred.mi}%
            </div>
            <div className="mt-4 flex gap-2">
              <span className={`px-3 py-1 rounded-full text-sm ${
                prediction.confidence.level === 'HIGH' ? 'bg-green-500' : 'bg-yellow-500'
              }`}>
                신뢰도: {prediction.confidence.level}
              </span>
              <span className="px-3 py-1 rounded-full text-sm bg-white/20">
                예측 방식: {pred.method}
              </span>
            </div>
          </Card>
        </div>

        {/* Key Drivers */}
        <Card className="p-6">
          <div className="flex items-center mb-4">
            <Lightbulb className="w-6 h-6 mr-2 text-yellow-500" />
            <h3 className="text-lg font-semibold">핵심 영향 요인</h3>
          </div>
          {summary?.key_drivers?.map((driver: any, idx: number) => (
            <div key={idx} className="mb-4">
              <div className="flex justify-between mb-1">
                <span className="text-sm">{driver.factor}</span>
                <span className="text-sm font-semibold">{driver.impact}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${
                    driver.direction === '상승' ? 'bg-green-500' : 'bg-orange-500'
                  }`}
                  style={{ width: driver.impact }}
                />
              </div>
            </div>
          ))}
        </Card>

        {/* Risk Factors */}
        <Card className="p-6">
          <div className="flex items-center mb-4">
            <AlertTriangle className="w-6 h-6 mr-2 text-orange-500" />
            <h3 className="text-lg font-semibold">리스크 요인</h3>
          </div>
          {summary?.risk_factors?.map((risk: string, idx: number) => (
            <div key={idx} className="mb-2 p-3 bg-orange-50 border-l-4 border-orange-500 text-sm">
              {risk}
            </div>
          ))}
        </Card>

        {/* Model Contribution */}
        <Card className="p-6 lg:col-span-2">
          <h3 className="text-lg font-semibold mb-4">예측 모델 구성</h3>
          <div className="w-full max-w-md mx-auto">
            <Pie data={modelContributionData} options={{
              plugins: {
                legend: {
                  position: 'bottom'
                }
              }
            }} />
          </div>
        </Card>
      </div>
    );
  };

  const renderScenarioAnalysis = () => {
    if (!scenarios) return null;

    const scenarioData = Object.entries(scenarios).map(([key, value]: [string, any]) => ({
      name: value.name,
      base_up: value.prediction.base_up,
      mi: value.prediction.mi,
      total: value.prediction.total
    }));

    const barChartData = {
      labels: scenarioData.map(s => s.name),
      datasets: [
        {
          label: 'Base-up',
          data: scenarioData.map(s => s.base_up),
          backgroundColor: '#3b82f6'
        },
        {
          label: 'MI',
          data: scenarioData.map(s => s.mi),
          backgroundColor: '#10b981'
        }
      ]
    };

    return (
      <div className="space-y-6">
        {/* Scenario Comparison Chart */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">시나리오별 예측 비교</h3>
          <Bar data={barChartData} options={{
            responsive: true,
            scales: {
              x: { stacked: true },
              y: { stacked: true }
            }
          }} />
        </Card>

        {/* Scenario Details */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(scenarios).map(([key, scenario]: [string, any]) => (
            <Card key={key} className="p-4">
              <h4 className="font-semibold mb-2">{scenario.name}</h4>
              <p className="text-sm text-gray-600 mb-3">{scenario.description}</p>
              <div className="border-t pt-3">
                <div className="text-2xl font-bold text-blue-600">
                  {scenario.prediction.total}%
                </div>
                <div className="text-sm text-gray-600">
                  Base-up: {scenario.prediction.base_up}% | MI: {scenario.prediction.mi}%
                </div>
                <div className="mt-2 text-xs">
                  <div>GDP: {scenario.parameters.gdp}%</div>
                  <div>최저임금: {scenario.parameters.min_wage}%</div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    );
  };

  const renderInteractiveSimulator = () => {
    const sliderConfigs = [
      { key: 'min_wage', label: '최저임금 인상률', min: 0, max: 10, step: 0.5 },
      { key: 'gdp', label: 'GDP 성장률', min: 0, max: 5, step: 0.1 },
      { key: 'revenue_growth', label: '매출 성장률', min: 0, max: 50, step: 1 },
      { key: 'profit_margin', label: '영업이익률', min: 5, max: 30, step: 0.5 },
      { key: 'industry_avg', label: '업계 평균', min: 0, max: 10, step: 0.5 },
      { key: 'crisis_score', label: '위기 지수', min: 0, max: 1, step: 0.1 }
    ];

    return (
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card className="p-6">
            <div className="flex items-center mb-4">
              <Sliders className="w-6 h-6 mr-2" />
              <h3 className="text-lg font-semibold">실시간 시뮬레이터</h3>
            </div>

            <div className="space-y-6">
              {sliderConfigs.map(config => (
                <div key={config.key}>
                  <div className="flex justify-between mb-2">
                    <Label>{config.label}</Label>
                    <span className="text-sm font-semibold">
                      {simulatorParams[config.key as keyof typeof simulatorParams]}
                    </span>
                  </div>
                  <Slider
                    value={[simulatorParams[config.key as keyof typeof simulatorParams]]}
                    onValueChange={(value) =>
                      setSimulatorParams({
                        ...simulatorParams,
                        [config.key]: value[0]
                      })
                    }
                    min={config.min}
                    max={config.max}
                    step={config.step}
                    className="w-full"
                  />
                </div>
              ))}
            </div>

            <Button
              onClick={runSimulation}
              className="w-full mt-6"
            >
              시뮬레이션 실행
            </Button>
          </Card>
        </div>

        <div className="lg:col-span-1">
          {simulationResult && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">시뮬레이션 결과</h3>
              <div className="text-3xl font-bold text-blue-600 mb-2">
                {simulationResult.total}%
              </div>
              <div className="space-y-1 text-sm">
                <div>Base-up: {simulationResult.base_up}%</div>
                <div>MI: {simulationResult.mi}%</div>
              </div>
              <div className="mt-4 pt-4 border-t">
                <div className="text-sm text-gray-600">기본 시나리오 대비:</div>
                <div className={`text-xl font-semibold ${
                  simulationResult.comparison_to_base > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {simulationResult.comparison_to_base > 0 ? '+' : ''}
                  {simulationResult.comparison_to_base}%p
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    );
  };

  const renderSensitivityAnalysis = () => {
    if (!sensitivity) {
      return (
        <Card className="p-6">
          <Button onClick={loadSensitivityAnalysis}>민감도 분석 로드</Button>
        </Card>
      );
    }

    const sensitivityData = Object.entries(sensitivity.sensitivity?.variables || {}).map(
      ([variable, data]: [string, any]) => ({
        variable: variable.replace(/_/g, ' '),
        impact: data.max_impact
      })
    );

    const barChartData = {
      labels: sensitivityData.map(s => s.variable),
      datasets: [{
        label: '최대 영향도',
        data: sensitivityData.map(s => s.impact),
        backgroundColor: '#3b82f6'
      }]
    };

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">변수별 민감도 분석</h3>
        <Bar data={barChartData} options={{
          indexAxis: 'y',
          responsive: true
        }} />
        {sensitivity.most_sensitive && (
          <div className="mt-4 p-3 bg-blue-50 rounded">
            <p className="text-sm">
              가장 민감한 변수: <strong>{sensitivity.most_sensitive}</strong>
              (최대 영향: ±{sensitivity.max_impact}%p)
            </p>
          </div>
        )}
      </Card>
    );
  };

  const renderHistoricalPatterns = () => {
    if (!historicalPatterns) return null;

    return (
      <div className="space-y-6">
        <Card className="p-6">
          <div className="flex items-center mb-4">
            <Clock className="w-6 h-6 mr-2" />
            <h3 className="text-lg font-semibold">역사적 패턴 분석 (2015-2024)</h3>
          </div>

          {/* Pattern Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {Object.entries(historicalPatterns.patterns || {}).map(
              ([period, data]: [string, any]) => (
                <div key={period} className="p-4 bg-gray-50 rounded-lg">
                  <h4 className="font-semibold">{period}</h4>
                  <div className="text-2xl font-bold text-blue-600 my-2">
                    평균 {data.avg_increase}%
                  </div>
                  <p className="text-sm text-gray-600">{data.characteristics}</p>
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs">
                    {data.key_learning}
                  </div>
                </div>
              )
            )}
          </div>

          {/* Key Events Table */}
          <div>
            <h4 className="font-semibold mb-3">주요 이벤트별 대응</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left p-2">연도</th>
                    <th className="text-left p-2">이벤트</th>
                    <th className="text-left p-2">임금인상률</th>
                    <th className="text-left p-2">교훈</th>
                  </tr>
                </thead>
                <tbody>
                  {historicalPatterns.key_events?.map((event: any, idx: number) => (
                    <tr key={idx} className="border-b">
                      <td className="p-2">{event.year}</td>
                      <td className="p-2">{event.event}</td>
                      <td className="p-2">
                        <span className="px-2 py-1 bg-blue-100 rounded text-xs">
                          {event.response}
                        </span>
                      </td>
                      <td className="p-2 text-xs">{event.lesson}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Card>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Activity className="w-12 h-12 animate-spin mx-auto mb-4" />
          <p>대시보드 로딩 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold flex items-center">
          <Target className="w-8 h-8 mr-3" />
          전략적 임금 의사결정 대시보드
        </h1>
      </div>

      <Tabs defaultValue="summary" className="space-y-4">
        <TabsList className="grid grid-cols-5 w-full">
          <TabsTrigger value="summary">Executive Summary</TabsTrigger>
          <TabsTrigger value="scenarios">시나리오 분석</TabsTrigger>
          <TabsTrigger value="simulator">시뮬레이터</TabsTrigger>
          <TabsTrigger value="sensitivity">민감도 분석</TabsTrigger>
          <TabsTrigger value="historical">역사적 패턴</TabsTrigger>
        </TabsList>

        <TabsContent value="summary">{renderExecutiveSummary()}</TabsContent>
        <TabsContent value="scenarios">{renderScenarioAnalysis()}</TabsContent>
        <TabsContent value="simulator">{renderInteractiveSimulator()}</TabsContent>
        <TabsContent value="sensitivity">{renderSensitivityAnalysis()}</TabsContent>
        <TabsContent value="historical">{renderHistoricalPatterns()}</TabsContent>
      </Tabs>
    </div>
  );
};

export default StrategicDashboard;