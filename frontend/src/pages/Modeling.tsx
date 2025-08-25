import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  Settings, 
  Play, 
  BarChart, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Target,
  Brain,
  TrendingUp,
  Loader2,
  Trash2,
  Info
} from 'lucide-react';
import { apiClient } from '../lib/api';

interface ModelingStatus {
  pycaret_available: boolean;
  environment_setup: boolean;
  model_trained: boolean;
  models_compared: boolean;
  data_loaded: boolean;
  current_model_type?: string;
}

export const Modeling: React.FC = () => {
  const [status, setStatus] = useState<ModelingStatus | null>(null);
  const [recommendations, setRecommendations] = useState<any>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dualModelResults, setDualModelResults] = useState<{
    baseup?: any;
    performance?: any;
  }>({});

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const statusRes = await apiClient.getModelingStatus();
      setStatus(statusRes);
      
      if (statusRes.data_loaded) {
        const recsRes = await apiClient.getModelingRecommendations();
        setRecommendations(recsRes);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  };

  const handleCompareAndTrainBothModels = async () => {
    setLoading('dual-process');
    setError(null);
    setDualModelResults({});

    try {
      const results: any = {};
      
      // Base-up 모델 처리
      console.log('Base-up 모델 처리 시작...');
      await apiClient.setupModeling('wage_increase_bu_sbl');
      const baseupComparison = await apiClient.compareModels(5);
      
      if (baseupComparison.comparison_results && baseupComparison.comparison_results[0]) {
        const bestBaseupModel = baseupComparison.comparison_results[0].Model;
        const baseupTraining = await apiClient.trainModel(bestBaseupModel, false);
        results.baseup = {
          comparison: baseupComparison,
          training: baseupTraining,
          selectedModel: bestBaseupModel,
          metrics: baseupComparison.comparison_results[0]
        };
      }
      
      // Performance 모델 처리
      console.log('성과급 모델 처리 시작...');
      await apiClient.setupModeling('wage_increase_mi_sbl');
      const performanceComparison = await apiClient.compareModels(5);
      
      if (performanceComparison.comparison_results && performanceComparison.comparison_results[0]) {
        const bestPerformanceModel = performanceComparison.comparison_results[0].Model;
        const performanceTraining = await apiClient.trainModel(bestPerformanceModel, false);
        results.performance = {
          comparison: performanceComparison,
          training: performanceTraining,
          selectedModel: bestPerformanceModel,
          metrics: performanceComparison.comparison_results[0]
        };
      }
      
      setDualModelResults(results);
      await loadInitialData();
    } catch (error) {
      setError(error instanceof Error ? error.message : '듀얼 모델 처리 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const handleClearModels = async () => {
    setLoading('clear');
    setError(null);

    try {
      await apiClient.clearModels();
      setDualModelResults({});
      await loadInitialData();
    } catch (error) {
      setError(error instanceof Error ? error.message : '모델 초기화 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const renderStatusAlert = () => {
    if (!status) return null;

    if (!status.pycaret_available) {
      return (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>PyCaret 설치 필요</AlertTitle>
          <AlertDescription>
            PyCaret이 설치되지 않았습니다. 다음 명령어로 설치해주세요:
            <code className="block mt-2 p-2 bg-background border rounded">pip install pycaret</code>
          </AlertDescription>
        </Alert>
      );
    }

    if (!status.data_loaded) {
      return (
        <Alert variant="warning">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>데이터 없음</AlertTitle>
          <AlertDescription>
            모델링을 위해 먼저 데이터를 업로드해주세요.
          </AlertDescription>
        </Alert>
      );
    }

    return null;
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">모델링</h1>
          <p className="text-muted-foreground">PyCaret을 사용한 머신러닝 모델 학습</p>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {renderStatusAlert()}

      {/* 권고사항 */}
      {recommendations && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Info className="mr-2 h-5 w-5" />
              모델링 권고사항
            </CardTitle>
            <CardDescription>
              현재 데이터에 최적화된 모델링 전략
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-background border p-3 rounded-lg">
                <p className="text-sm font-medium">데이터 크기</p>
                <p className="text-lg font-bold">{recommendations.data_analysis?.data_size.toLocaleString()}</p>
              </div>
              <div className="bg-background border p-3 rounded-lg">
                <p className="text-sm font-medium">피처 수</p>
                <p className="text-lg font-bold">{recommendations.data_analysis?.feature_count}</p>
              </div>
              <div className="bg-background border p-3 rounded-lg">
                <p className="text-sm font-medium">데이터/피처 비율</p>
                <p className="text-lg font-bold">{recommendations.data_analysis?.data_to_feature_ratio}</p>
              </div>
            </div>
            
            <div>
              <h4 className="font-medium mb-2">권고사항:</h4>
              <ul className="list-disc list-inside space-y-1 text-sm">
                {recommendations.recommendations?.map((rec: string, index: number) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 메인 액션 카드 - 듀얼 모델 자동 처리 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Brain className="mr-2 h-5 w-5" />
            듀얼 모델 자동 학습
            {dualModelResults.baseup && dualModelResults.performance && (
              <CheckCircle className="ml-auto h-5 w-5 text-green-600" />
            )}
          </CardTitle>
          <CardDescription>
            Base-up과 성과급 모델을 자동으로 비교하고 최적 모델을 선정하여 학습합니다
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!dualModelResults.baseup && !dualModelResults.performance ? (
            <Button 
              className="w-full" 
              size="lg"
              onClick={handleCompareAndTrainBothModels}
              disabled={loading === 'dual-process' || !status?.data_loaded}
            >
              {loading === 'dual-process' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  자동 처리 중...
                </>
              ) : (
                '🚀 듀얼 모델 자동 학습 시작'
              )}
            </Button>
          ) : (
            <div className="space-y-3">
              <div className="text-center p-3 bg-green-100 dark:bg-green-900/20 rounded-lg">
                <CheckCircle className="mx-auto h-12 w-12 text-green-600 mb-2" />
                <p className="text-sm font-semibold text-green-900 dark:text-green-100">
                  두 모델 모두 학습 완료!
                </p>
              </div>
              <Button 
                variant="outline"
                className="w-full" 
                onClick={handleClearModels}
                disabled={loading === 'clear'}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                {loading === 'clear' ? '초기화 중...' : '모델 초기화'}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 결과 표시 */}
      {(dualModelResults.baseup || dualModelResults.performance) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              모델링 결과
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Base-up 모델 결과 */}
            {dualModelResults.baseup && (
              <div className="p-4 rounded-lg border-2" style={{
                backgroundColor: 'rgb(59 130 246 / 0.15)',
                borderColor: 'rgb(59 130 246 / 0.5)'
              }}>
                <h4 className="font-bold text-lg text-blue-900 dark:text-blue-100 mb-3">
                  📊 Base-up 모델 (wage_increase_bu_sbl)
                </h4>
                <div className="bg-white/50 dark:bg-black/30 p-3 rounded-lg mb-3">
                  <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                    선정된 최적 모델:
                  </p>
                  <p className="text-2xl font-bold text-blue-800 dark:text-blue-200 mt-1">
                    {dualModelResults.baseup.selectedModel}
                  </p>
                </div>
                {dualModelResults.baseup.metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-blue-700 dark:text-blue-300">MAE</p>
                      <p className="text-sm font-bold text-blue-900 dark:text-blue-100">
                        {dualModelResults.baseup.metrics.MAE?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-blue-700 dark:text-blue-300">RMSE</p>
                      <p className="text-sm font-bold text-blue-900 dark:text-blue-100">
                        {dualModelResults.baseup.metrics.RMSE?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-blue-700 dark:text-blue-300">R2 Score</p>
                      <p className="text-sm font-bold text-blue-900 dark:text-blue-100">
                        {dualModelResults.baseup.metrics.R2?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-blue-700 dark:text-blue-300">MAPE</p>
                      <p className="text-sm font-bold text-blue-900 dark:text-blue-100">
                        {dualModelResults.baseup.metrics.MAPE?.toFixed(2) || 'N/A'}%
                      </p>
                    </div>
                  </div>
                )}
                <p className="text-xs text-blue-600 dark:text-blue-400 mt-3">
                  총 {dualModelResults.baseup.comparison?.models_compared || 0}개 모델 비교 완료
                </p>
              </div>
            )}

            {/* Performance 모델 결과 */}
            {dualModelResults.performance && (
              <div className="p-4 rounded-lg border-2" style={{
                backgroundColor: 'rgb(34 197 94 / 0.15)',
                borderColor: 'rgb(34 197 94 / 0.5)'
              }}>
                <h4 className="font-bold text-lg text-green-900 dark:text-green-100 mb-3">
                  💰 성과급 모델 (wage_increase_mi_sbl)
                </h4>
                <div className="bg-white/50 dark:bg-black/30 p-3 rounded-lg mb-3">
                  <p className="text-sm font-semibold text-green-900 dark:text-green-100">
                    선정된 최적 모델:
                  </p>
                  <p className="text-2xl font-bold text-green-800 dark:text-green-200 mt-1">
                    {dualModelResults.performance.selectedModel}
                  </p>
                </div>
                {dualModelResults.performance.metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-green-700 dark:text-green-300">MAE</p>
                      <p className="text-sm font-bold text-green-900 dark:text-green-100">
                        {dualModelResults.performance.metrics.MAE?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-green-700 dark:text-green-300">RMSE</p>
                      <p className="text-sm font-bold text-green-900 dark:text-green-100">
                        {dualModelResults.performance.metrics.RMSE?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-green-700 dark:text-green-300">R2 Score</p>
                      <p className="text-sm font-bold text-green-900 dark:text-green-100">
                        {dualModelResults.performance.metrics.R2?.toFixed(4) || 'N/A'}
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-black/30 p-2 rounded">
                      <p className="text-xs text-green-700 dark:text-green-300">MAPE</p>
                      <p className="text-sm font-bold text-green-900 dark:text-green-100">
                        {dualModelResults.performance.metrics.MAPE?.toFixed(2) || 'N/A'}%
                      </p>
                    </div>
                  </div>
                )}
                <p className="text-xs text-green-600 dark:text-green-400 mt-3">
                  총 {dualModelResults.performance.comparison?.models_compared || 0}개 모델 비교 완료
                </p>
              </div>
            )}

            {/* 종합 결과 */}
            {dualModelResults.baseup && dualModelResults.performance && (
              <div className="p-4 rounded-lg border-2" style={{
                backgroundColor: 'rgb(168 85 247 / 0.15)',
                borderColor: 'rgb(168 85 247 / 0.5)'
              }}>
                <h4 className="font-bold text-lg text-purple-900 dark:text-purple-100 mb-3">
                  ✅ 최종 결과 요약
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white/50 dark:bg-black/30 p-3 rounded-lg">
                    <p className="text-xs text-purple-700 dark:text-purple-300 mb-1">Base-up 모델</p>
                    <p className="text-lg font-bold text-purple-900 dark:text-purple-100">
                      {dualModelResults.baseup.selectedModel}
                    </p>
                    <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                      MAE: {dualModelResults.baseup.metrics?.MAE?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                  <div className="bg-white/50 dark:bg-black/30 p-3 rounded-lg">
                    <p className="text-xs text-purple-700 dark:text-purple-300 mb-1">성과급 모델</p>
                    <p className="text-lg font-bold text-purple-900 dark:text-purple-100">
                      {dualModelResults.performance.selectedModel}
                    </p>
                    <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
                      MAE: {dualModelResults.performance.metrics?.MAE?.toFixed(4) || 'N/A'}
                    </p>
                  </div>
                </div>
                <p className="text-sm text-purple-700 dark:text-purple-300 mt-3 text-center">
                  💡 전체 인상률 = Base-up + 성과급
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};