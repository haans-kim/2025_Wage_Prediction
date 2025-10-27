const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface DataUploadResponse {
  message: string;
  filename: string;
  file_path: string;
  data_analysis: {
    basic_stats: {
      shape: [number, number];
      columns: string[];
      dtypes: Record<string, string>;
      memory_usage: number;
    };
    missing_analysis: {
      missing_counts: Record<string, number>;
      missing_percentages: Record<string, number>;
      total_missing: number;
    };
    numeric_stats: Record<string, any>;
    categorical_stats: Record<string, any>;
    sample_data: any[];
    numeric_columns: string[];
    categorical_columns: string[];
  };
  validation: {
    is_valid: boolean;
    issues: string[];
    recommendations: string[];
  };
  summary: {
    shape: [number, number];
    columns: string[];
    numeric_columns: string[];
    categorical_columns: string[];
    missing_data_percentage: number;
    memory_usage_mb: number;
  };
}

export interface SampleDataResponse {
  message: string;
  source: string;
  data: any[];
  summary?: any;
  analysis?: any;
  shape?: [number, number];
  columns?: string[];
}

export class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  // Generic HTTP methods
  async get(path: string, options?: { params?: Record<string, any> }): Promise<any> {
    const url = new URL(`${this.baseUrl}${path}`);
    
    if (options?.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    const response = await fetch(url.toString());

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return { data: await response.json() };
  }

  async post(path: string, data?: any): Promise<any> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return { data: await response.json() };
  }

  async uploadFile(file: File): Promise<DataUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/data/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getSampleData(rows: number = 10): Promise<SampleDataResponse> {
    const response = await fetch(`${this.baseUrl}/api/data/master?rows=${rows}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getCurrentData(rows: number = 20, includeAnalysis: boolean = false) {
    const response = await fetch(
      `${this.baseUrl}/api/data/current?rows=${rows}&include_analysis=${includeAnalysis}`
    );
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async validateData() {
    const response = await fetch(`${this.baseUrl}/api/data/validate`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getDataInfo() {
    const response = await fetch(`${this.baseUrl}/api/data/info`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async clearCurrentData() {
    const response = await fetch(`${this.baseUrl}/api/data/current`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // Modeling API methods
  async setupModeling(targetColumn: string, trainSize?: number): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/setup`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        target_column: targetColumn,
        train_size: trainSize,
        session_id: 42
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async compareModels(nSelect: number = 3): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/compare?n_select=${nSelect}`, {
      method: 'POST',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async trainModel(modelName: string, tuneHyperparameters: boolean = true): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model_name: modelName,
        tune_hyperparameters: tuneHyperparameters
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async evaluateModel(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/evaluate`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async predictWithModel(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/predict`, {
      method: 'POST',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getAvailableModels(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/available-models`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getModelingStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/status`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getModelingRecommendations(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/recommendations`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async clearModels(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/modeling/clear`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // Analysis API methods
  async getShapAnalysis(sampleIndex?: number, topN: number = 10): Promise<any> {
    const params = new URLSearchParams();
    if (sampleIndex !== undefined) params.append('sample_index', sampleIndex.toString());
    params.append('top_n', topN.toString());

    const response = await fetch(`${this.baseUrl}/api/analysis/shap?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getFeatureImportance(method: string = 'shap', topN: number = 15): Promise<any> {
    const params = new URLSearchParams();
    params.append('method', method);
    params.append('top_n', topN.toString());

    const response = await fetch(`${this.baseUrl}/api/analysis/feature-importance?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getLimeAnalysis(sampleIndex: number, numFeatures: number = 10): Promise<any> {
    const params = new URLSearchParams();
    params.append('sample_index', sampleIndex.toString());
    params.append('num_features', numFeatures.toString());

    const response = await fetch(`${this.baseUrl}/api/analysis/lime?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getModelPerformance(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analysis/model-performance`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getPartialDependence(featureName: string, numGridPoints: number = 50): Promise<any> {
    const params = new URLSearchParams();
    params.append('feature_name', featureName);
    params.append('num_grid_points', numGridPoints.toString());

    const response = await fetch(`${this.baseUrl}/api/analysis/partial-dependence?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getResidualAnalysis(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analysis/residual-analysis`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getPredictionIntervals(confidenceLevel: number = 0.95): Promise<any> {
    const params = new URLSearchParams();
    params.append('confidence_level', confidenceLevel.toString());

    const response = await fetch(`${this.baseUrl}/api/analysis/prediction-intervals?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }


  // Dashboard API methods
  async predictWageIncrease(inputData: Record<string, number>, confidenceLevel: number = 0.95): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input_data: inputData,
        confidence_level: confidenceLevel
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async runScenarioAnalysis(scenarios: Array<{scenario_name: string, variables: Record<string, number>, description?: string}>): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/scenario-analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(scenarios),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getAvailableVariables(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/variables`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getHistoricalTrends(years: number = 10, includeForecast: boolean = true): Promise<any> {
    const params = new URLSearchParams();
    params.append('years', years.toString());
    params.append('include_forecast', includeForecast.toString());

    const response = await fetch(`${this.baseUrl}/api/dashboard/historical-trends?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getEconomicIndicators(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/economic-indicators`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getScenarioTemplates(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/scenario-templates`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async runSensitivityAnalysis(baseScenario: Record<string, number>, variableName: string, variationRange: number = 0.2): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/sensitivity-analysis`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        base_scenario: baseScenario,
        variable_name: variableName,
        variation_range: variationRange
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getForecastAccuracy(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/forecast-accuracy`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async runMonteCarloSimulation(
    baseScenario: Record<string, number>, 
    uncertaintyRanges: Record<string, number>, 
    numSimulations: number = 1000
  ): Promise<any> {
    const params = new URLSearchParams();
    params.append('num_simulations', numSimulations.toString());

    const response = await fetch(`${this.baseUrl}/api/dashboard/monte-carlo?${params}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        base_scenario: baseScenario,
        uncertainty_ranges: uncertaintyRanges
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getMarketConditions(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/market-conditions`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async createCustomScenario(
    scenarioName: string, 
    variables: Record<string, number>, 
    saveTemplate: boolean = false
  ): Promise<any> {
    const params = new URLSearchParams();
    params.append('scenario_name', scenarioName);
    params.append('save_template', saveTemplate.toString());

    const response = await fetch(`${this.baseUrl}/api/dashboard/custom-scenario?${params}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(variables),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getPredictionBreakdown(scenarioVariables?: Record<string, number>): Promise<any> {
    const params = new URLSearchParams();
    if (scenarioVariables) {
      params.append('scenario_variables', JSON.stringify(scenarioVariables));
    }

    const response = await fetch(`${this.baseUrl}/api/dashboard/prediction-breakdown?${params}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  // Data augmentation and status methods
  async getDataStatus(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/data/status`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async loadDefaultData(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/data/load-default`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async augmentDataAdvanced(options: {
    method?: string;
    target_size?: number;
    factor?: number;
    noise_level?: number;
    preserve_distribution?: boolean;
  }): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/data/augment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }
  
  async resetToMaster(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/data/reset-to-master`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async augmentData(targetSize: number = 120, noiseFactor: number = 0.02): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/data/augment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        target_size: targetSize,
        noise_factor: noiseFactor
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getTrendData(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/dashboard/trend-data`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getExplainerDashboard(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analysis/explainer-dashboard`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async generateExplainerDashboard(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analysis/explainer-dashboard`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async getContributionPlot(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/analysis/contribution-plot`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();