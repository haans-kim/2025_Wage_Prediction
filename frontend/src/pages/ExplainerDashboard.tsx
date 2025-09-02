import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { 
  LineChart,
  BarChart3,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Info,
  ExternalLink
} from 'lucide-react';
import { apiClient } from '../lib/api';

export const ExplainerDashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [explainerUrl, setExplainerUrl] = useState<string | null>(null);

  useEffect(() => {
    checkAndGenerateDashboard();
  }, []);

  const checkAndGenerateDashboard = async () => {
    setLoading(true);
    setError(null);

    try {
      // 먼저 ExplainerDashboard 상태 확인
      const response = await apiClient.getExplainerDashboard();
      if (response.url) {
        setExplainerUrl(response.url);
      } else {
        // 대시보드가 없으면 자동으로 생성
        console.log('ExplainerDashboard가 없습니다. 자동으로 생성합니다...');
        await generateExplainerDashboard();
      }
    } catch (err: any) {
      // 상태 확인 실패 시에도 생성 시도
      console.log('상태 확인 실패. 새로 생성을 시도합니다...');
      await generateExplainerDashboard();
    } finally {
      setLoading(false);
    }
  };

  const checkExplainerStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      // 백엔드에 ExplainerDashboard URL 요청
      const response = await apiClient.getExplainerDashboard();
      if (response.url) {
        setExplainerUrl(response.url);
      } else {
        // 대시보드가 없으면 자동으로 생성
        await generateExplainerDashboard();
      }
    } catch (err: any) {
      setError(err.message || 'ExplainerDashboard 상태 확인 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const generateExplainerDashboard = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiClient.generateExplainerDashboard();
      if (response.url) {
        setExplainerUrl(response.url);
      }
    } catch (err: any) {
      setError(err.message || 'ExplainerDashboard 생성 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Explainer Dashboard</h1>
          <p className="text-muted-foreground">모델의 예측을 시각적으로 설명하고 이해하기 위한 대화형 대시보드</p>
        </div>
        <Button 
          onClick={checkExplainerStatus} 
          disabled={loading}
          variant="outline"
        >
          {loading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              확인 중...
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              새로고침
            </>
          )}
        </Button>
      </div>

      {loading && !explainerUrl && (
        <Alert>
          <Loader2 className="h-4 w-4 animate-spin" />
          <AlertTitle>ExplainerDashboard 준비 중</AlertTitle>
          <AlertDescription>
            대시보드를 생성하고 있습니다. 잠시만 기다려주세요...
          </AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">개요</TabsTrigger>
          <TabsTrigger value="dashboard" disabled={!explainerUrl}>
            대시보드
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Info className="mr-2 h-5 w-5" />
                ExplainerDashboard란?
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                ExplainerDashboard는 머신러닝 모델의 예측을 이해하고 설명하기 위한 대화형 웹 애플리케이션입니다.
                다음과 같은 기능을 제공합니다:
              </p>
              <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                <li>개별 예측에 대한 SHAP 값 시각화</li>
                <li>Feature importance 분석</li>
                <li>What-if 시나리오 분석</li>
                <li>모델 성능 지표 및 시각화</li>
                <li>예측 분포 및 신뢰도 분석</li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="mr-2 h-5 w-5" />
                주요 기능
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <h4 className="font-medium">모델 성능 개요</h4>
                <p className="text-sm text-muted-foreground">
                  모델의 정확도, R², MAE, RMSE 등 주요 성능 지표를 한눈에 확인
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">Feature Importance</h4>
                <p className="text-sm text-muted-foreground">
                  각 변수가 예측에 미치는 영향도를 다양한 방법으로 분석
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">개별 예측 설명</h4>
                <p className="text-sm text-muted-foreground">
                  특정 데이터 포인트에 대한 예측 과정을 상세히 분석
                </p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium">What-if 분석</h4>
                <p className="text-sm text-muted-foreground">
                  입력값을 변경했을 때 예측이 어떻게 바뀌는지 실시간으로 확인
                </p>
              </div>
            </CardContent>
          </Card>

          {!explainerUrl && !loading && (
            <Card>
              <CardHeader>
                <CardTitle>ExplainerDashboard 재생성</CardTitle>
                <CardDescription>
                  대시보드를 다시 생성하려면 아래 버튼을 클릭하세요.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button 
                  onClick={generateExplainerDashboard}
                  disabled={loading}
                  className="w-full"
                >
                  <LineChart className="mr-2 h-4 w-4" />
                  ExplainerDashboard 재생성
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="dashboard" className="space-y-4">
          {explainerUrl ? (
            <div className="bg-background rounded-lg p-6">
              <Alert>
                <ExternalLink className="h-4 w-4" />
                <AlertTitle>ExplainerDashboard 준비 완료</AlertTitle>
                <AlertDescription className="space-y-4">
                  <p>ExplainerDashboard가 별도 포트에서 실행 중입니다.</p>
                  <div className="flex gap-4 items-center mt-3">
                    <Button 
                      onClick={() => {
                        // URL을 서버 주소로 변경 (0.0.0.0을 현재 호스트로)
                        const url = explainerUrl.replace('0.0.0.0', window.location.hostname).replace('localhost', window.location.hostname);
                        window.open(url, '_blank');
                      }}
                      className="flex items-center gap-2"
                    >
                      <ExternalLink className="h-4 w-4" />
                      새 창에서 열기 (포트 8050)
                    </Button>
                    <div className="text-sm text-muted-foreground">
                      주의: 외부 접근 시 8050 포트가 방화벽에서 허용되어야 합니다
                    </div>
                  </div>
                </AlertDescription>
              </Alert>
              
              {/* 로컬에서만 iframe 표시 */}
              {window.location.hostname === 'localhost' && (
                <div className="mt-6 bg-background rounded-lg overflow-hidden" style={{ height: 'calc(100vh - 300px)' }}>
                  <iframe
                    src={explainerUrl}
                    className="w-full h-full border-0"
                    title="ExplainerDashboard"
                    allow="fullscreen"
                    style={{ minHeight: '600px' }}
                  />
                </div>
              )}
            </div>
          ) : (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>대시보드 없음</AlertTitle>
              <AlertDescription>
                ExplainerDashboard가 아직 생성되지 않았습니다. 
                '개요' 탭에서 대시보드를 생성해주세요.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};