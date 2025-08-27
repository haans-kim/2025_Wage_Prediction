import React, { useState, useCallback, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  Upload, 
  FileSpreadsheet, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
  Download,
  Eye,
  Trash2,
  Database,
  Zap,
  Loader2,
  Info
} from 'lucide-react';
import { apiClient, DataUploadResponse, SampleDataResponse } from '../lib/api';

export const DataUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [loadingDefault, setLoadingDefault] = useState(false);
  const [augmenting, setAugmenting] = useState(false);
  const [uploadResult, setUploadResult] = useState<DataUploadResponse | null>(null);
  const [sampleData, setSampleData] = useState<SampleDataResponse | null>(null);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showDataPreview, setShowDataPreview] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  }, []);

  const checkDataStatus = async () => {
    try {
      const status = await apiClient.getDataStatus();
      setDataStatus(status);
      
      // 마스터 데이터가 있는 경우 현재 데이터도 확인
      if (status.status.has_master_data) {
        try {
          const response = await apiClient.getCurrentData(5, false);
          if (response) {
            setSampleData(response);
            setShowDataPreview(true);
          }
        } catch (error) {
          console.log("Current data check failed, but default data exists");
        }
      }
    } catch (error) {
      console.error("Failed to check data status:", error);
    }
  };

  const loadDefaultData = async () => {
    setLoadingDefault(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await apiClient.loadDefaultData();
      setSuccess("마스터 데이터가 성공적으로 로드되었습니다.");
      setSampleData({
        message: result.message,
        source: "pickle",
        data: [],
        summary: result.summary
      });
      setShowDataPreview(true);
      await checkDataStatus(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '마스터 데이터 로드 중 오류가 발생했습니다.');
    } finally {
      setLoadingDefault(false);
    }
  };

  const augmentData = async () => {
    setAugmenting(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await apiClient.augmentData(1100, 0.01); // 10x multiplier with 1% noise
      if (result.augmentation_applied) {
        setSuccess(`데이터가 ${result.original_size}개에서 ${result.augmented_size}개로 증강되었습니다.`);
      } else {
        setSuccess(result.message);
      }
      
      // 증강 후 현재 데이터 새로고침
      const currentData = await apiClient.getCurrentData(10, false);
      setSampleData(currentData);
      await checkDataStatus(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '데이터 증강 중 오류가 발생했습니다.');
    } finally {
      setAugmenting(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);
    
    try {
      const result = await apiClient.uploadFile(file);
      setUploadResult(result);
      setShowDataPreview(true);
    } catch (error) {
      setError(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다.');
    } finally {
      setUploading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoadingSample(true);
    setError(null);
    
    try {
      const result = await apiClient.getSampleData(20);
      setSampleData(result);
      setShowDataPreview(true);
    } catch (error) {
      setError(error instanceof Error ? error.message : '마스터 데이터 로드 중 오류가 발생했습니다.');
    } finally {
      setLoadingSample(false);
    }
  };

  const clearData = async () => {
    try {
      await apiClient.clearCurrentData();
      setUploadResult(null);
      setSampleData(null);
      setFile(null);
      setShowDataPreview(false);
      setError(null);
      setSuccess(null);
      await checkDataStatus(); // 상태 새로고침
    } catch (error) {
      setError('데이터 삭제 중 오류가 발생했습니다.');
    }
  };

  // 컴포넌트 마운트 시 데이터 상태 확인
  useEffect(() => {
    checkDataStatus();
  }, []);

  const renderValidationStatus = (validation: any) => {
    if (validation.is_valid) {
      return (
        <Alert variant="success">
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>데이터 검증 완료</AlertTitle>
          <AlertDescription>
            모델링을 위한 데이터 준비가 완료되었습니다.
          </AlertDescription>
        </Alert>
      );
    } else {
      return (
        <Alert variant="warning">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>데이터 검증 이슈</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside space-y-1">
              {validation.issues.map((issue: string, index: number) => (
                <li key={index}>{issue}</li>
              ))}
            </ul>
            {validation.recommendations.length > 0 && (
              <div className="mt-2">
                <strong>권고사항:</strong>
                <ul className="list-disc list-inside space-y-1">
                  {validation.recommendations.map((rec: string, index: number) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </AlertDescription>
        </Alert>
      );
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">데이터 업로드</h1>
          <p className="text-muted-foreground">Excel 파일을 업로드하여 분석을 시작하세요.</p>
        </div>
        {(uploadResult || sampleData) && (
          <Button variant="outline" onClick={clearData}>
            <Trash2 className="mr-2 h-4 w-4" />
            데이터 초기화
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

      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>성공</AlertTitle>
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Data Status Information */}
      {dataStatus && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Info className="mr-2 h-5 w-5" />
              데이터 상태
            </CardTitle>
            <CardDescription>
              현재 로드된 데이터와 마스터 데이터 정보
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm">마스터 데이터 파일:</span>
              <span className={`text-sm font-medium ${dataStatus.status.has_master_data ? 'text-green-600' : 'text-muted-foreground'}`}>
                {dataStatus.status.has_master_data ? '사용 가능' : '없음'}
              </span>
            </div>
            {dataStatus.status.data_shape && (
              <div className="flex items-center justify-between">
                <span className="text-sm">현재 데이터 크기:</span>
                <span className="text-sm font-medium">
                  {dataStatus.status.data_shape[0]} × {dataStatus.status.data_shape[1]}
                </span>
              </div>
            )}
            <div className="flex space-x-2">
              <Button 
                variant="outline" 
                size="sm"
                onClick={loadDefaultData}
                disabled={loadingDefault || !dataStatus.status.has_master_data}
                className="flex-1"
              >
                {loadingDefault ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    로딩 중...
                  </>
                ) : (
                  <>
                    <Database className="mr-2 h-4 w-4" />
                    마스터 데이터 로드
                  </>
                )}
              </Button>
              <Button 
                variant="outline" 
                size="sm"
                onClick={augmentData}
                disabled={augmenting || !dataStatus.status.data_shape}
                className="flex-1"
              >
                {augmenting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    증강 중...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2 h-4 w-4" />
                    데이터 증강 (10배수)
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* File Upload Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Upload className="mr-2 h-5 w-5" />
              파일 업로드
            </CardTitle>
            <CardDescription>
              Excel (.xlsx, .xls) 또는 CSV 파일을 선택하세요. (최대 10MB)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div 
              className={`border-2 border-dashed rounded-lg p-8 text-center relative transition-colors ${
                dragActive 
                  ? 'border-primary bg-primary/10' 
                  : 'border-border hover:border-primary/50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {file ? (
                <div className="space-y-2">
                  <FileSpreadsheet className="mx-auto h-12 w-12 text-primary" />
                  <p className="text-sm font-medium">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
                  <p className="text-sm text-muted-foreground">
                    파일을 드래그하거나 클릭하여 선택하세요
                  </p>
                </div>
              )}
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={handleFileSelect}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={uploading}
              />
            </div>
            
            {file && (
              <Button 
                onClick={handleUpload} 
                disabled={uploading}
                className="w-full"
              >
                {uploading ? '업로드 중...' : '업로드'}
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Master Data Card */}
        <Card>
          <CardHeader>
            <CardTitle>마스터 데이터</CardTitle>
            <CardDescription>
              기본 제공되는 마스터 데이터를 사용하여 바로 시작할 수 있습니다.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              variant="outline" 
              className="w-full"
              onClick={handleLoadSample}
              disabled={loadingSample}
            >
              <Download className="mr-2 h-4 w-4" />
              {loadingSample ? '로드 중...' : '마스터 데이터 사용'}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">업로드 결과</h2>
          
          {renderValidationStatus(uploadResult.validation)}
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">데이터 크기</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">
                  {uploadResult.summary.shape[0].toLocaleString()} × {uploadResult.summary.shape[1]}
                </p>
                <p className="text-xs text-muted-foreground">행 × 열</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">수치형 컬럼</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">{uploadResult.summary.numeric_columns.length}</p>
                <p className="text-xs text-muted-foreground">개</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">결측값 비율</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-2xl font-bold">
                  {uploadResult.summary.missing_data_percentage.toFixed(1)}%
                </p>
                <p className="text-xs text-muted-foreground">전체 데이터 대비</p>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Master Data Result */}
      {sampleData && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">마스터 데이터</h2>
          
          <Alert>
            <Download className="h-4 w-4" />
            <AlertTitle>마스터 데이터 로드됨</AlertTitle>
            <AlertDescription>
              {sampleData.source === 'generated' 
                ? '시연용 가상 데이터가 생성되었습니다.'
                : `${sampleData.source} 파일에서 데이터를 로드했습니다.`
              }
            </AlertDescription>
          </Alert>
          
          {sampleData.shape && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">데이터 정보</CardTitle>
              </CardHeader>
              <CardContent>
                <p><strong>크기:</strong> {sampleData.shape[0]} × {sampleData.shape[1]}</p>
                <p><strong>컬럼:</strong> {sampleData.columns?.join(', ')}</p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Data Preview */}
      {showDataPreview && (uploadResult || sampleData) && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center">
                <Eye className="mr-2 h-5 w-5" />
                데이터 미리보기
              </span>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowDataPreview(!showDataPreview)}
              >
                {showDataPreview ? '숨기기' : '보기'}
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {showDataPreview && (
              <div className="overflow-x-auto">
                <table className="w-full border-collapse border border-border">
                  <thead>
                    <tr className="bg-background border-b">
                      {(uploadResult?.data_analysis.sample_data[0] 
                        ? Object.keys(uploadResult.data_analysis.sample_data[0])
                        : sampleData?.data[0] 
                        ? Object.keys(sampleData.data[0])
                        : []
                      ).map((column) => (
                        <th key={column} className="border border-border p-2 text-left text-sm font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(uploadResult?.data_analysis.sample_data || sampleData?.data || [])
                      .slice(0, uploadResult ? undefined : 10)
                      .map((row, index) => (
                        <tr key={index} className="even:bg-background/50 border-b border-border/50">
                          {Object.values(row).map((cell, cellIndex) => (
                            <td key={cellIndex} className="border border-border p-2 text-sm">
                              {String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};