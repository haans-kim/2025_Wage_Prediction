import React, { useState, useCallback, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Slider } from '../components/ui/slider';
import { Label } from '../components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { 
  Upload, 
  FileSpreadsheet, 
  CheckCircle, 
  XCircle, 
  AlertTriangle,
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
  const [loadingDefault, setLoadingDefault] = useState(false);
  const [augmenting, setAugmenting] = useState(false);
  const [uploadResult, setUploadResult] = useState<DataUploadResponse | null>(null);
  const [sampleData, setSampleData] = useState<SampleDataResponse | null>(null);
  const [dataStatus, setDataStatus] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [showDataPreview, setShowDataPreview] = useState(false);
  const [showAugmentOptions, setShowAugmentOptions] = useState(false);
  
  // 증강 옵션 상태
  const [augmentMethod, setAugmentMethod] = useState<string>('auto');
  const [augmentFactor] = useState<number>(10);
  const [targetSize, setTargetSize] = useState<number>(200);
  const [noiseLevel, setNoiseLevel] = useState<number>(0.02);

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
      
      // 기본 데이터 또는 마스터 데이터가 있는 경우 현재 데이터 확인
      if (status.status.has_default_data || status.status.has_master || status.status.has_current) {
        try {
          const response = await apiClient.getCurrentData(5, false);
          if (response) {
            setSampleData(response);
            setShowDataPreview(true);
          }
        } catch (error) {
          console.log("Current data check failed, trying to get available data");
          // 현재 데이터가 없으면 마스터 데이터라도 보여주기
          if (status.status.has_master) {
            setShowDataPreview(true);
          }
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
      setSuccess("기본 데이터가 성공적으로 로드되었습니다.");
      
      // 로드 후 실제 데이터 가져오기
      const currentData = await apiClient.getCurrentData(10, false);
      setSampleData(currentData);
      setShowDataPreview(true);
      
      await checkDataStatus(); // 상태 새로고침
    } catch (error) {
      setError(error instanceof Error ? error.message : '기본 데이터 로드 중 오류가 발생했습니다.');
    } finally {
      setLoadingDefault(false);
    }
  };

  const augmentData = async () => {
    setAugmenting(true);
    setError(null);
    setSuccess(null);

    try {
      const result = await apiClient.augmentDataAdvanced({
        method: augmentMethod,
        target_size: targetSize,
        factor: augmentFactor,
        noise_level: noiseLevel,
        preserve_distribution: true
      });
      
      if (result.augmentation_info) {
        setSuccess(`데이터가 ${result.augmentation_info.original_size}개에서 ${result.augmentation_info.augmented_size}개로 증강되었습니다. (방법: ${result.augmentation_info.method})`);
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
  
  const resetToMaster = async () => {
    try {
      await apiClient.resetToMaster();
      setSuccess('마스터 데이터로 리셋되었습니다.');
      setShowDataPreview(true);
      await checkDataStatus();
    } catch (error) {
      setError(error instanceof Error ? error.message : '리셋 중 오류가 발생했습니다.');
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
      
      // 업로드 후 데이터 상태 확인 및 샘플 데이터 로드
      await checkDataStatus();
    } catch (error) {
      setError(error instanceof Error ? error.message : '업로드 중 오류가 발생했습니다.');
    } finally {
      setUploading(false);
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
    <div className="space-y-3">
      <div className="flex justify-between items-center mb-2">
        <div>
          <h1 className="text-xl font-bold text-foreground">데이터 업로드</h1>
          <p className="text-xs text-muted-foreground">Excel 파일을 업로드하여 분석을 시작하세요</p>
        </div>
        {(uploadResult || sampleData) && (
          <Button variant="outline" size="sm" onClick={clearData}>
            <Trash2 className="mr-1 h-3 w-3" />
            초기화
          </Button>
        )}
      </div>

      {error && (
        <Alert variant="destructive" className="py-2">
          <XCircle className="h-3 w-3" />
          <AlertDescription className="text-xs">{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="py-2">
          <CheckCircle className="h-3 w-3" />
          <AlertDescription className="text-xs">{success}</AlertDescription>
        </Alert>
      )}

      {/* 저장된 데이터 섹션 */}
      <Card>
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm flex items-center">
            <Database className="mr-1.5 h-4 w-4" />
            저장된 데이터
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-3">
          {dataStatus && (dataStatus.status?.has_master_data || dataStatus.has_master_data) ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between p-2 bg-muted/50 rounded">
                <div>
                  <p className="text-sm font-medium">마스터 데이터 사용 가능</p>
                  {(dataStatus.status?.master_data_shape || dataStatus.master_data_shape) && (
                    <p className="text-xs text-muted-foreground">
                      {dataStatus.status?.master_data_shape?.[0] || dataStatus.master_data_shape?.[0]}개 행 × {dataStatus.status?.master_data_shape?.[1] || dataStatus.master_data_shape?.[1]}개 열
                    </p>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={loadDefaultData}
                    disabled={loadingDefault}
                  >
                    {loadingDefault ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Database className="h-3 w-3" />
                    )}
                    <span className="ml-1">불러오기</span>
                  </Button>
                  {(dataStatus.status?.master_data_shape || dataStatus.master_data_shape) && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => setShowAugmentOptions(!showAugmentOptions)}
                    >
                      <Zap className="h-3 w-3" />
                      <span className="ml-1">데이터 증강</span>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center py-4">
              <Info className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">저장된 데이터가 없습니다</p>
              <p className="text-xs text-muted-foreground mt-1">새 파일을 업로드하거나 기본 데이터를 로드하세요</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 데이터 증강 옵션 */}
      {showAugmentOptions && (dataStatus?.status?.has_master_data || dataStatus?.has_master_data) && (
        <Card>
          <CardHeader className="pb-2 pt-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <span className="flex items-center">
                <Zap className="mr-1.5 h-4 w-4" />
                데이터 증강 설정
              </span>
              {dataStatus?.status?.is_augmented && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetToMaster}
                  className="h-6 text-xs"
                >
                  마스터 데이터로 리셋
                </Button>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 space-y-4">
            {/* 증강 방법 선택 */}
            <div className="space-y-2">
              <Label className="text-xs">증강 방법</Label>
              <Select value={augmentMethod} onValueChange={setAugmentMethod}>
                <SelectTrigger className="h-8">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">자동 선택</SelectItem>
                  <SelectItem value="noise">가우시안 노이즈</SelectItem>
                  <SelectItem value="interpolation">보간법 (시계열)</SelectItem>
                  <SelectItem value="mixup">Mixup (혼합)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* 목표 크기 */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-xs">목표 데이터 크기</Label>
                <span className="text-xs text-muted-foreground">{targetSize}개</span>
              </div>
              <Slider
                value={[targetSize]}
                onValueChange={([value]: number[]) => setTargetSize(value)}
                min={100}
                max={1000}
                step={50}
                className="w-full"
              />
            </div>

            {/* 노이즈 수준 (노이즈 방법일 때만) */}
            {augmentMethod === 'noise' && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-xs">노이즈 수준</Label>
                  <span className="text-xs text-muted-foreground">{(noiseLevel * 100).toFixed(1)}%</span>
                </div>
                <Slider
                  value={[noiseLevel * 100]}
                  onValueChange={([value]: number[]) => setNoiseLevel(value / 100)}
                  min={0.5}
                  max={10}
                  step={0.5}
                  className="w-full"
                />
              </div>
            )}

            {/* 현재 상태 */}
            {dataStatus?.status && (
              <div className="p-2 bg-muted/50 rounded text-xs space-y-1">
                <div className="flex justify-between">
                  <span>마스터 데이터:</span>
                  <span className="font-medium">
                    {dataStatus.status.master_data_shape?.[0] || 0} 행
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>현재 작업 데이터:</span>
                  <span className="font-medium">
                    {dataStatus.status.working_data_shape?.[0] || 0} 행
                    {dataStatus.status.is_augmented && " (증강됨)"}
                  </span>
                </div>
              </div>
            )}

            {/* 증강 버튼 */}
            <Button
              onClick={augmentData}
              disabled={augmenting}
              className="w-full"
              size="sm"
            >
              {augmenting ? (
                <>
                  <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                  증강 중...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-3 w-3" />
                  데이터 증강 실행
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* File Upload Card */}
      <Card>
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm flex items-center">
            <Upload className="mr-1.5 h-4 w-4" />
            파일 업로드
          </CardTitle>
        </CardHeader>
        <CardContent className="pb-3">
          <div 
            className={`border-2 border-dashed rounded-lg p-6 text-center relative transition-colors ${
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
              <div className="space-y-1">
                <FileSpreadsheet className="mx-auto h-10 w-10 text-primary" />
                <p className="text-sm font-medium">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                <Upload className="mx-auto h-10 w-10 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  파일을 드래그하거나 클릭하여 선택
                </p>
                <p className="text-xs text-muted-foreground">
                  지원 형식: .xlsx, .xls, .csv
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
              className="w-full mt-2"
              size="sm"
            >
              {uploading ? '업로드 중...' : '업로드'}
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Upload Result */}
      {uploadResult && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold">업로드 결과</h2>
          
          {renderValidationStatus(uploadResult.validation)}
          
          <div className="grid grid-cols-3 gap-2">
            <Card>
              <CardHeader className="pb-0 pt-2">
                <CardTitle className="text-xs text-muted-foreground">데이터 크기</CardTitle>
              </CardHeader>
              <CardContent className="pt-1 pb-2">
                <p className="text-sm font-bold">
                  {uploadResult.summary?.shape[0].toLocaleString()} × {uploadResult.summary?.shape[1]}
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-0 pt-2">
                <CardTitle className="text-xs text-muted-foreground">수치형 컬럼</CardTitle>
              </CardHeader>
              <CardContent className="pt-1 pb-2">
                <p className="text-sm font-bold">{uploadResult.summary?.numeric_columns.length}개</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader className="pb-0 pt-2">
                <CardTitle className="text-xs text-muted-foreground">결측값</CardTitle>
              </CardHeader>
              <CardContent className="pt-1 pb-2">
                <p className="text-sm font-bold">
                  {uploadResult.summary?.missing_data_percentage.toFixed(1)}%
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Data Preview */}
      {showDataPreview && (uploadResult || sampleData) && (
        <Card>
          <CardHeader className="pb-2 pt-3">
            <CardTitle className="text-sm flex items-center justify-between">
              <span className="flex items-center">
                <Eye className="mr-1.5 h-4 w-4" />
                데이터 미리보기
                {(uploadResult?.data_analysis.sample_data || sampleData?.data) && (
                  <span className="ml-2 text-xs text-muted-foreground">
                    ({(uploadResult?.data_analysis.sample_data || sampleData?.data || []).length}행)
                  </span>
                )}
              </span>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowDataPreview(!showDataPreview)}
                className="h-6 text-xs"
              >
                {showDataPreview ? '숨기기' : '보기'}
              </Button>
            </CardTitle>
          </CardHeader>
          {showDataPreview && (
            <CardContent className="pb-3">
              <div className="overflow-auto max-h-96">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b">
                      {(uploadResult?.data_analysis.sample_data[0] 
                        ? Object.keys(uploadResult.data_analysis.sample_data[0])
                        : sampleData?.data[0] 
                        ? Object.keys(sampleData.data[0])
                        : []
                      ).map((column) => (
                        <th key={column} className="p-1 text-left font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(uploadResult?.data_analysis.sample_data || sampleData?.data || [])
                      .map((row: any, index: number) => (
                        <tr key={index} className="border-b">
                          {Object.values(row).map((cell, cellIndex) => (
                            <td key={cellIndex} className="p-1">
                              {String(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
};