import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  AlertTriangle,
  Loader2
} from 'lucide-react';
import { apiClient } from '../lib/api';
import { ShapAnalysis } from '../components/ShapAnalysis';

interface FeatureImportance {
  feature: string;
  importance: number;
  std?: number;
}

interface ShapAnalysisData {
  available: boolean;
  feature_importance: FeatureImportance[];
  sample_explanation?: any;
  explainer_type: string;
  n_features: number;
  error?: string;
}

interface ModelPerformance {
  performance: {
    train_metrics: {
      mse: number;
      mae: number;
      r2: number;
    };
    test_metrics?: {
      mse: number;
      mae: number;
      r2: number;
    };
    residual_analysis: {
      mean_residual: number;
      std_residual: number;
      residual_range: [number, number];
    };
  };
  model_type: string;
  error?: string;
}

export const Analysis: React.FC = () => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [shapAnalysis, setShapAnalysis] = useState<ShapAnalysisData | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [baseupShapData, setBaseupShapData] = useState<FeatureImportance[]>([]);
  const [performanceShapData, setPerformanceShapData] = useState<FeatureImportance[]>([]);
  const [baseupModelName, setBaseupModelName] = useState<string>('');
  const [performanceModelName, setPerformanceModelName] = useState<string>('');

  useEffect(() => {
    loadAnalysisData();
  }, []);

  const loadAnalysisData = async () => {
    setLoading('initial');
    setError(null);

    try {
      const [shapRes, featureRes, performanceRes, baseupShapRes, performanceShapRes] = await Promise.all([
        apiClient.getShapAnalysis().catch(() => ({ available: false, error: 'SHAP 분석을 사용할 수 없습니다.' })),
        apiClient.getFeatureImportance().catch(() => ({ feature_importance: [], error: 'Feature importance 분석을 사용할 수 없습니다.' })),
        apiClient.getModelPerformance().catch(() => ({ performance: {}, error: '성능 분석을 사용할 수 없습니다.' })),
        apiClient.get('/api/analysis/shap', { params: { target: 'wage_increase_bu_sbl', top_n: 3 }}).catch(() => ({ data: { feature_importance: [] }})),
        apiClient.get('/api/analysis/shap', { params: { target: 'wage_increase_mi_sbl', top_n: 3 }}).catch(() => ({ data: { feature_importance: [] }}))
      ]);

      setShapAnalysis(shapRes);
      setFeatureImportance(featureRes.feature_importance || []);
      setModelPerformance(performanceRes);
      setBaseupShapData(baseupShapRes.data?.feature_importance || []);
      setPerformanceShapData(performanceShapRes.data?.feature_importance || []);
      setBaseupModelName(baseupShapRes.data?.model_name || '');
      setPerformanceModelName(performanceShapRes.data?.model_name || '');
    } catch (error) {
      setError('분석 데이터를 불러오는 중 오류가 발생했습니다.');
      console.error('Analysis data loading failed:', error);
    } finally {
      setLoading(null);
    }
  };




  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Analysis</h1>
          <p className="text-muted-foreground">2026년 임금인상률 예측 분석</p>
        </div>
        <Button onClick={loadAnalysisData} disabled={loading === 'initial'}>
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
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* 2026년 예측 결과 - 3개 카드 나란히 */}
      <div className="grid grid-cols-3 gap-4">
        {/* Base-up 카드 */}
        <Card className="border-blue-500 dark:border-blue-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-blue-600 dark:text-blue-400">Base-up 인상률</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">3.5%</p>
            <p className="text-xs text-muted-foreground">신뢰구간: 2.8% ~ 3.8%</p>
          </CardContent>
        </Card>

        {/* 성과급 카드 */}
        <Card className="border-green-500 dark:border-green-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-green-600 dark:text-green-400">성과급 인상률</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">2.1%</p>
            <p className="text-xs text-muted-foreground">신뢰구간: 1.8% ~ 2.3%</p>
          </CardContent>
        </Card>

        {/* 총 인상률 카드 */}
        <Card className="border-purple-500 dark:border-purple-600">
          <CardHeader className="pb-3">
            <CardTitle className="text-base font-medium text-purple-600 dark:text-purple-400">총 예상 인상률</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            <p className="text-3xl font-bold">5.6%</p>
            <p className="text-xs text-muted-foreground">Base-up + 성과급</p>
          </CardContent>
        </Card>
      </div>

      {/* 모델 성능 비교 - 2열 레이아웃 */}
      <div className="grid grid-cols-2 gap-6">
        {/* Base-up 모델 성능 */}
        <div>
          <h3 className="text-sm font-semibold text-blue-600 dark:text-blue-400 mb-3">Base-up 모델 성능 {baseupModelName && `(${baseupModelName})`}</h3>
          <div className="grid grid-cols-3 gap-2">
            <Card className="border-blue-400 dark:border-blue-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">MAE</p>
                <p className="text-base font-bold">0.0065</p>
              </CardContent>
            </Card>
            <Card className="border-blue-400 dark:border-blue-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">RMSE</p>
                <p className="text-base font-bold">0.0083</p>
              </CardContent>
            </Card>
            <Card className="border-blue-400 dark:border-blue-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">R² Score</p>
                <p className="text-base font-bold">-2.836</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* 성과급 모델 성능 */}
        <div>
          <h3 className="text-sm font-semibold text-green-600 dark:text-green-400 mb-3">성과급 모델 성능 {performanceModelName && `(${performanceModelName})`}</h3>
          <div className="grid grid-cols-3 gap-2">
            <Card className="border-green-400 dark:border-green-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">MAE</p>
                <p className="text-base font-bold">0.0011</p>
              </CardContent>
            </Card>
            <Card className="border-green-400 dark:border-green-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">RMSE</p>
                <p className="text-base font-bold">0.0013</p>
              </CardContent>
            </Card>
            <Card className="border-green-400 dark:border-green-600">
              <CardContent className="pt-3 pb-2 px-3">
                <p className="text-xs text-muted-foreground">R² Score</p>
                <p className="text-base font-bold">-3.417</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* SHAP 분석 - ShapAnalysis 컴포넌트 사용 */}
      <ShapAnalysis />

      {/* 주요 인사이트 */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold text-blue-600 dark:text-blue-400">
              Base-up 핵심 동인
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-1">
            {baseupShapData.length > 0 ? (
              <>
                {baseupShapData.slice(0, 3).map((item, idx) => (
                  <p key={idx}>• {item.feature} ({(item.importance * 100).toFixed(1)}%)</p>
                ))}
                <p className="font-semibold pt-1">• 권고: 3.5% 인상</p>
              </>
            ) : (
              <>
                <p>• 미국 CPI (64.7%)</p>
                <p>• 대기업 인상률 (23.7%)</p>
                <p>• 권고: 3.5% 인상</p>
              </>
            )}
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold text-green-600 dark:text-green-400">
              성과급 핵심 동인
            </CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-1">
            {performanceShapData.length > 0 ? (
              <>
                {performanceShapData.slice(0, 3).map((item, idx) => (
                  <p key={idx}>• {item.feature} ({(item.importance * 100).toFixed(1)}%)</p>
                ))}
                <p className="font-semibold pt-1">• 권고: 2.1% 지급</p>
              </>
            ) : (
              <>
                <p>• 매출증가율 (핵심)</p>
                <p>• 실적 연동 설계</p>
                <p>• 권고: 2.1% 지급</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 경고 메시지 */}
      <Alert className="border-amber-200 bg-amber-50 dark:bg-amber-950">
        <AlertTriangle className="h-4 w-4 text-amber-600" />
        <AlertDescription className="text-sm">
          현재 10개 샘플로는 정확한 예측이 제한적입니다. 최소 30-50개 데이터가 권장되며, 트렌드 분석과 도메인 지식을 함께 활용하시기 바랍니다.
        </AlertDescription>
      </Alert>
    </div>
  );
};