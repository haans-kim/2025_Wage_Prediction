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
import { apiClient, ModelingStatus as ApiModelingStatus } from '../lib/api';

interface ModelingStatus extends ApiModelingStatus {
  environment_setup: boolean;
  model_trained: boolean;
  models_compared: boolean;
  data_loaded: boolean;
  current_model_type?: string;
}

interface AvailableModel {
  code: string;
  name: string;
  recommended: boolean;
}

export const Modeling: React.FC = () => {
  const [status, setStatus] = useState<ModelingStatus | null>(null);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [recommendations, setRecommendations] = useState<any>(null);
  const [targetColumn, setTargetColumn] = useState<string>('wage_increase_total_sbl');
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [setupResult, setSetupResult] = useState<any>(null);
  const [comparisonResult, setComparisonResult] = useState<any>(null);
  const [trainingResult, setTrainingResult] = useState<any>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      // 모델링 상태 및 데이터 확인
      const [statusRes, currentDataRes] = await Promise.all([
        apiClient.getModelingStatus(),
        apiClient.getCurrentData(5, false).catch(() => null)
      ]);

      setStatus({
        ...statusRes,
        environment_setup: statusRes.environment_setup || false,
        model_trained: statusRes.model_trained || false,
        models_compared: statusRes.models_compared || false,
        data_loaded: statusRes.data_loaded || false
      });
      
      if (currentDataRes?.summary?.columns) {
        setAvailableColumns(currentDataRes.summary.columns);
        // wage_increase_total_sbl이 컬럼에 있으면 기본으로 선택
        if (currentDataRes.summary.columns.includes('wage_increase_total_sbl')) {
          setTargetColumn('wage_increase_total_sbl');
        }
      }

      // 권고사항 로드 (데이터가 있는 경우)
      if (statusRes.data_loaded) {
        const recsRes = await apiClient.getModelingRecommendations();
        setRecommendations(recsRes);

        const modelsRes = await apiClient.getAvailableModels();
        setAvailableModels(modelsRes.available_models || []);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  };

  const handleSetupModeling = async () => {
    if (!targetColumn) {
      setError('타겟 컬럼을 선택해주세요.');
      return;
    }

    setLoading('setup');
    setError(null);

    try {
      const result = await apiClient.setupModeling(targetColumn);
      setSetupResult(result);
      await loadInitialData(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '환경 설정 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const handleCompareModels = async () => {
    setLoading('compare');
    setError(null);

    try {
      const result = await apiClient.compareModels(3);
      setComparisonResult(result);
      await loadInitialData(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '모델 비교 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const handleTrainModel = async (modelCode: string) => {
    setLoading(`train-${modelCode}`);
    setError(null);

    try {
      const result = await apiClient.trainModel(modelCode, 0.2);
      setTrainingResult(result);
      await loadInitialData(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '모델 학습 중 오류가 발생했습니다.');
    } finally {
      setLoading(null);
    }
  };

  const handleClearModels = async () => {
    setLoading('clear');
    setError(null);

    try {
      await apiClient.clearModels();
      setSetupResult(null);
      setComparisonResult(null);
      setTrainingResult(null);
      await loadInitialData(); // 상태 새로고침
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

    if (status.environment_setup && status.model_trained) {
      return (
        <Alert variant="success">
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>모델링 완료</AlertTitle>
          <AlertDescription>
            {status.current_model_type} 모델이 성공적으로 학습되었습니다.
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
        {status?.environment_setup && (
          <Button variant="outline" onClick={handleClearModels} disabled={loading === 'clear'}>
            <Trash2 className="mr-2 h-4 w-4" />
            {loading === 'clear' ? '초기화 중...' : '모델 초기화'}
          </Button>
        )}
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

      {/* 메인 액션 카드들 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* 환경 설정 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Settings className="mr-2 h-5 w-5" />
              환경 설정
              {status?.environment_setup && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              타겟 변수 선택 및 PyCaret 환경 구성
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!status?.environment_setup ? (
              <>
                <div>
                  <label className="text-sm font-medium">타겟 컬럼 선택</label>
                  <select 
                    value={targetColumn} 
                    onChange={(e) => setTargetColumn(e.target.value)}
                    className="w-full mt-1 p-2 border border-border rounded-md bg-background"
                    disabled={!availableColumns.length}
                  >
                    <option value="">선택하세요</option>
                    {availableColumns.slice().reverse().map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                </div>
                <Button 
                  className="w-full" 
                  onClick={handleSetupModeling}
                  disabled={loading === 'setup' || !targetColumn || !status?.data_loaded}
                >
                  {loading === 'setup' ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      설정 중...
                    </>
                  ) : (
                    '환경 설정 시작'
                  )}
                </Button>
              </>
            ) : (
              <div className="text-center">
                <CheckCircle className="mx-auto h-8 w-8 text-green-600 mb-2" />
                <p className="text-sm text-muted-foreground">환경 설정 완료</p>
                <p className="text-xs text-muted-foreground mt-1">
                  타겟: {setupResult?.setup_request?.target_column}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 모델 비교 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <BarChart className="mr-2 h-5 w-5" />
              모델 비교
              {status?.models_compared && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              여러 모델의 성능을 자동으로 비교
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleCompareModels}
              disabled={loading === 'compare' || !status?.environment_setup}
            >
              {loading === 'compare' ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  비교 중...
                </>
              ) : (
                '모델 비교 시작'
              )}
            </Button>
            {comparisonResult && (
              <div className="mt-4 p-3 bg-background border rounded-lg">
                <p className="text-sm">
                  <strong>추천 모델:</strong> {comparisonResult.recommended_model_type}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {comparisonResult.models_compared}개 모델 비교 완료
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 모델 학습 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Play className="mr-2 h-5 w-5" />
              모델 학습
              {status?.model_trained && <CheckCircle className="ml-auto h-5 w-5 text-green-600" />}
            </CardTitle>
            <CardDescription>
              선택된 모델 학습 및 하이퍼파라미터 튜닝
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!status?.model_trained ? (
              <Button 
                variant="outline" 
                className="w-full"
                onClick={() => handleTrainModel('lr')}
                disabled={loading?.startsWith('train') || !status?.environment_setup}
              >
                {loading?.startsWith('train') ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    학습 중...
                  </>
                ) : (
                  '자동 모델 학습'
                )}
              </Button>
            ) : (
              <div className="text-center">
                <Brain className="mx-auto h-8 w-8 text-blue-600 mb-2" />
                <p className="text-sm text-muted-foreground">모델 학습 완료</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {status.current_model_type}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 사용 가능한 모델 목록 */}
      {availableModels.length > 0 && status?.environment_setup && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="mr-2 h-5 w-5" />
              사용 가능한 모델
            </CardTitle>
            <CardDescription>
              현재 데이터에 최적화된 모델 목록
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {availableModels.map((model) => (
                <div key={model.code} className="border border-border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="font-medium text-sm">{model.name}</h4>
                    {model.recommended && (
                      <span className="text-xs bg-primary text-primary-foreground px-2 py-1 rounded">
                        추천
                      </span>
                    )}
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="w-full"
                    onClick={() => handleTrainModel(model.code)}
                    disabled={loading === `train-${model.code}`}
                  >
                    {loading === `train-${model.code}` ? (
                      <>
                        <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                        학습 중...
                      </>
                    ) : (
                      '학습'
                    )}
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 결과 요약 */}
      {(setupResult || comparisonResult || trainingResult) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <TrendingUp className="mr-2 h-5 w-5" />
              모델링 결과
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {setupResult && (
              <div className="p-3 rounded-lg border" style={{
                backgroundColor: 'rgb(59 130 246 / 0.15)',
                borderColor: 'rgb(59 130 246 / 0.3)'
              }}>
                <h4 className="font-medium text-blue-900 dark:text-blue-100">환경 설정 완료</h4>
                <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                  모델링 데이터: {setupResult.data_info?.final_shape?.[0]} × {setupResult.data_info?.final_shape?.[1]}
                  {setupResult.data_info?.removed_target_missing && (
                    <span className="text-xs block text-blue-600 dark:text-blue-400">
                      (타겟 결측값 {setupResult.data_info.removed_target_missing}개 행 제외)
                    </span>
                  )}
                </p>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  사용 가능한 모델: {setupResult.available_models?.join(', ')}
                </p>
              </div>
            )}
            
            {comparisonResult && (
              <div className="p-3 rounded-lg border" style={{
                backgroundColor: 'rgb(34 197 94 / 0.15)',
                borderColor: 'rgb(34 197 94 / 0.3)'
              }}>
                <h4 className="font-medium text-green-900 dark:text-green-100">모델 비교 완료</h4>
                <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                  최고 성능: {comparisonResult.recommended_model_type}
                </p>
                <p className="text-sm text-green-700 dark:text-green-300">
                  비교된 모델 수: {comparisonResult.models_compared}개
                </p>
              </div>
            )}

            {trainingResult && (
              <div className="p-3 rounded-lg border" style={{
                backgroundColor: 'rgb(168 85 247 / 0.15)',
                borderColor: 'rgb(168 85 247 / 0.3)'
              }}>
                <h4 className="font-medium text-purple-900 dark:text-purple-100">모델 학습 완료</h4>
                <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                  학습된 모델: {trainingResult.model_type}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};