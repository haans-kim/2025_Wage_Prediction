import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';

// Chart.js 등록
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

export const Effects: React.FC = () => {
  // 평균값 기준으로 정렬된 회사 데이터 (인건비율) - 엑셀 데이터 기반
  const laborCostSorted = [
    { name: 'Lonza', value: 34.7, color: 'rgb(168, 85, 247)' },
    { name: 'GSK', value: 29.6, color: 'rgb(250, 204, 21)' },
    { name: 'Wuxi', value: 28.7, color: 'rgb(34, 197, 94)' },
    { name: 'Roche', value: 26.5, color: 'rgb(251, 146, 60)' },
    { name: 'AstraZeneca', value: 26.4, color: 'rgb(239, 68, 68)' },
    { name: 'CSL', value: 26.0, color: 'rgb(220, 38, 127)' },
    { name: 'Bayer', value: 25.2, color: 'rgb(245, 158, 11)' },
    { name: 'SBL', value: 13.6, color: 'rgb(59, 130, 246)' },
    { name: 'Celltrion', value: 8.5, color: 'rgb(129, 140, 248)' }
  ];

  // 평균값 기준으로 정렬된 회사 데이터 (영업이익률)
  const profitSorted = [
    { name: 'SBL', value: 32.3, color: 'rgb(59, 130, 246)' },
    { name: 'Celltrion', value: 30.5, color: 'rgb(129, 140, 248)' },
    { name: 'Wuxi', value: 29.8, color: 'rgb(34, 197, 94)' },
    { name: 'Roche', value: 25, color: 'rgb(251, 146, 60)' },
    { name: 'Lonza', value: 24.3, color: 'rgb(168, 85, 247)' },
    { name: 'GSK', value: 23.3, color: 'rgb(250, 204, 21)' },
    { name: 'CSL', value: 18, color: 'rgb(220, 38, 127)' },
    { name: 'Bayer', value: 16.3, color: 'rgb(245, 158, 11)' },
    { name: 'AstraZeneca', value: 12.3, color: 'rgb(239, 68, 68)' }
  ];

  // 인당 영업이익 데이터 (단위: 백만원)
  const profitPerEmployeeData = {
    labels: ['2021', '2022', '2023', '2024', '4개년 평균'],
    datasets: [
      {
        label: 'SBL',
        data: [156, 245, 302, 289, 248],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        tension: 0.4,
        borderWidth: 4,
        pointRadius: 2,
        pointHoverRadius: 3,
        order: 0,
      },
      {
        label: 'Celltrion',
        data: [178, 201, 189, 215, 196],
        borderColor: 'rgb(129, 140, 248)',
        backgroundColor: 'rgba(129, 140, 248, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
      },
      {
        label: 'Wuxi',
        data: [145, 167, 178, 186, 169],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
      },
      {
        label: 'Lonza',
        data: [134, 142, 158, 165, 150],
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
      },
      {
        label: 'Roche',
        data: [123, 135, 142, 148, 137],
        borderColor: 'rgb(251, 146, 60)',
        backgroundColor: 'rgba(251, 146, 60, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
      },
    ]
  };

  // 인건비/매출액 비중 데이터
  const laborCostRatioData = {
    labels: ['2021', '2022', '2023', '2024', '4개년 평균'],
    datasets: [
      {
        label: 'SBL',
        data: [17.9, 13.1, 12.2, 11.3, 13.6],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        tension: 0.4,
        borderWidth: 4,  // 더 굵은 선
        pointRadius: 2,  // 작은 도트
        pointHoverRadius: 3,
        spanGaps: false,
        order: 0,  // 다른 선들 위에 그려지도록
      },
      {
        label: 'Lonza',
        data: [34.9, 34.4, 33.5, 35.9, 34.7],
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Wuxi',
        data: [34.7, 26.4, 26.3, 27.2, 28.7],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Celltrion',
        data: [9.0, 7.8, 9.3, 7.8, 8.5],
        borderColor: 'rgb(129, 140, 248)',
        backgroundColor: 'rgba(129, 140, 248, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Roche',
        data: [26.7, 24.5, 27.5, 27.2, 26.5],
        borderColor: 'rgb(251, 146, 60)',
        backgroundColor: 'rgba(251, 146, 60, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'GSK',
        data: [36.5, 26.2, 27.9, 27.9, 29.6],
        borderColor: 'rgb(250, 204, 21)',
        backgroundColor: 'rgba(250, 204, 21, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'CSL',
        data: [26.7, 26.7, 25.4, 25.0, 26.0],
        borderColor: 'rgb(220, 38, 127)',
        backgroundColor: 'rgba(220, 38, 127, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'AstraZeneca',
        data: [27.5, 26.0, 26.9, 25.4, 26.4],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Bayer',
        data: [26.8, 24.9, 22.4, 26.7, 25.2],
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Bayer',
        data: [27, 26, 22, 25, 25],
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      }
    ]
  };

  // 영업이익/매출액 비중 데이터
  const operatingProfitRatioData = {
    labels: ['2021', '2022', '2023', '2024', '4개년 평균'],
    datasets: [
      {
        label: 'SBL',
        data: [39, 31, 30, 29, 32.3],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        tension: 0.4,
        borderWidth: 4,  // 더 굵은 선
        pointRadius: 2,  // 작은 도트
        pointHoverRadius: 3,
        spanGaps: false,
        order: 0,  // 다른 선들 위에 그려지도록
      },
      {
        label: 'Lonza',
        data: [22, 27, 26, 22, 24.3],
        borderColor: 'rgb(168, 85, 247)',
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Wuxi',
        data: [34, 28, 30, 27, 29.8],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Celltrion',
        data: [32, 31, 30, 29, 30.5],
        borderColor: 'rgb(129, 140, 248)',
        backgroundColor: 'rgba(129, 140, 248, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Roche',
        data: [29, 28, 24, 19, 25],
        borderColor: 'rgb(251, 146, 60)',
        backgroundColor: 'rgba(251, 146, 60, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'GSK',
        data: [25, 22, 25, 21, 23.3],
        borderColor: 'rgb(250, 204, 21)',
        backgroundColor: 'rgba(250, 204, 21, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'CSL',
        data: [19, 22, 17, 14, 18],
        borderColor: 'rgb(220, 38, 127)',
        backgroundColor: 'rgba(220, 38, 127, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'AstraZeneca',
        data: [15, 14, 13, 7, 12.3],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      },
      {
        label: 'Bayer',
        data: [16, 14, 17, 18, 16.3],
        borderColor: 'rgb(245, 158, 11)',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 2,
        pointHoverRadius: 3,
        spanGaps: false,
      }
    ]
  };

  // 차트 옵션 (인건비율)
  const laborCostOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: '인건비/매출액 비중 2021~2024',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (context: any) => {
            const label = context.dataIndex === 4 ? '4개년 평균' : context.dataset.label;
            return `${label}: ${context.parsed.y}%`;
          }
        }
      },
      datalabels: {
        display: true, // 모든 포인트에 라벨 표시
        align: function(context: any) {
          const value = context.parsed?.y || context.dataset.data[context.dataIndex];
          // 값에 따라 라벨 위치 조정
          if (value > 30) {
            return 'top';
          } else if (value < 15) {
            return 'bottom';
          } else {
            return 'top';
          }
        } as any,
        anchor: 'end' as any,
        offset: 4,
        formatter: function(value: any, context: any) {
          const numValue = typeof value === 'number' ? value : parseFloat(value);
          return `${numValue.toFixed(1)}%`;
        },
        color: (context: any) => {
          // SBL은 진한 파란색, 나머지는 회색
          return context.dataset.label === 'SBL' ? 'rgb(59, 130, 246)' : '#666';
        },
        font: (context: any) => {
          // SBL은 굵고 크게, 나머지는 보통 크기
          if (context.dataset.label === 'SBL') {
            return {
              size: 14,
              weight: 'bold' as const
            };
          }
          return {
            size: 12,
            weight: 'normal' as const
          };
        }
      },
      annotation: {
        annotations: {
          // 4개년 평균 섹션 배경
          avgSection: {
            type: 'box' as const,
            xMin: 3.5,
            xMax: 4.5,
            backgroundColor: 'rgba(229, 231, 235, 0.5)',
            borderColor: 'rgba(156, 163, 175, 0.3)',
            borderWidth: 1,
            drawTime: 'beforeDatasetsDraw' as const
          },
          // 4개년 평균 섹션 구분선
          verticalLine: {
            type: 'line' as const,
            xMin: 3.5,
            xMax: 3.5,
            borderColor: 'rgba(156, 163, 175, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            drawTime: 'beforeDatasetsDraw' as const
          },
        }
      }
    },
    layout: {
      padding: {
        left: 10,
        right: 10,  // 오른쪽 공간 축소
        top: 20,
        bottom: 20
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 45,
        title: {
          display: false
        },
        ticks: {
          callback: (value: any) => `${value}%`,
          stepSize: 5
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        offset: true,  // 첫 번째와 마지막 포인트를 위한 오프셋 추가
        title: {
          display: false
        },
        grid: {
          display: false,
          offset: true  // 그리드에도 오프셋 적용
        },
        ticks: {
          padding: 10,  // 틱과 축 사이의 패딩
          callback: function(value: any, index: number): string {
            const labels = ['2021', '2022', '2023', '2024', '4개년 평균'];
            return labels[index] || '';
          }
        }
      }
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    }
  };

  // 차트 옵션 (영업이익률)
  const operatingProfitOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: '영업이익/매출액 비중 2021~2024',
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: (context: any) => {
            const label = context.dataIndex === 4 ? '4개년 평균' : context.dataset.label;
            return `${label}: ${context.parsed.y}%`;
          }
        }
      },
      datalabels: {
        display: true, // 모든 포인트에 라벨 표시
        align: function(context: any) {
          const value = context.parsed?.y || context.dataset.data[context.dataIndex];
          // 값에 따라 라벨 위치 조정
          if (value > 30) {
            return 'top';
          } else if (value < 15) {
            return 'bottom';
          } else {
            return 'top';
          }
        } as any,
        anchor: 'end' as any,
        offset: 4,
        formatter: function(value: any, context: any) {
          const numValue = typeof value === 'number' ? value : parseFloat(value);
          return `${numValue.toFixed(1)}%`;
        },
        color: (context: any) => {
          // SBL은 진한 파란색, 나머지는 회색
          return context.dataset.label === 'SBL' ? 'rgb(59, 130, 246)' : '#666';
        },
        font: (context: any) => {
          // SBL은 굵고 크게, 나머지는 보통 크기
          if (context.dataset.label === 'SBL') {
            return {
              size: 14,
              weight: 'bold' as const
            };
          }
          return {
            size: 12,
            weight: 'normal' as const
          };
        }
      },
      annotation: {
        annotations: {
          // 4개년 평균 섹션 배경
          avgSection: {
            type: 'box' as const,
            xMin: 3.5,
            xMax: 4.5,
            backgroundColor: 'rgba(229, 231, 235, 0.5)',
            borderColor: 'rgba(156, 163, 175, 0.3)',
            borderWidth: 1,
            drawTime: 'beforeDatasetsDraw' as const
          },
          // 4개년 평균 섹션 구분선
          verticalLine: {
            type: 'line' as const,
            xMin: 3.5,
            xMax: 3.5,
            borderColor: 'rgba(156, 163, 175, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            drawTime: 'beforeDatasetsDraw' as const
          },
        }
      }
    },
    layout: {
      padding: {
        left: 10,
        right: 10,  // 오른쪽 공간 축소
        top: 20,
        bottom: 20
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 45,
        title: {
          display: false
        },
        ticks: {
          callback: (value: any) => `${value}%`,
          stepSize: 5
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        offset: true,  // 첫 번째와 마지막 포인트를 위한 오프셋 추가
        title: {
          display: false
        },
        grid: {
          display: false,
          offset: true  // 그리드에도 오프셋 적용
        },
        ticks: {
          padding: 10,  // 틱과 축 사이의 패딩
          callback: function(value: any, index: number): string {
            const labels = ['2021', '2022', '2023', '2024', '4개년 평균'];
            return labels[index] || '';
          }
        }
      }
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    }
  };

  // 인당 영업이익 차트 옵션
  const profitPerEmployeeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        align: 'center' as const,
        labels: {
          padding: 10,
          usePointStyle: true,
          pointStyle: 'circle',
          font: {
            size: 11
          }
        }
      },
      title: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            return `${label}: ${value}백만원`;
          }
        },
        mode: 'index' as const,
        intersect: false,
      },
      datalabels: {
        display: true,  // 모든 데이터 포인트에 표시
        align: 'end' as const,
        anchor: 'end' as const,
        offset: 3,
        formatter: (value: any, context: any) => {
          // 모든 데이터셋에 대해 값 표시
          return `${value}`;
        },
        color: (context: any) => {
          // SBL은 진한 파란색, 나머지는 회색
          return context.dataset.label === 'SBL' ? 'rgb(59, 130, 246)' : '#666';
        },
        font: (context: any) => {
          // SBL은 굵고 크게, 나머지는 작고 보통
          if (context.dataset.label === 'SBL') {
            return {
              size: 14,
              weight: 'bold' as const
            };
          }
          return {
            size: 12,
            weight: 'normal' as const
          };
        }
      },
      annotation: {
        annotations: {
          avgSection: {
            type: 'box' as const,
            xMin: 3.5,
            xMax: 4.5,
            backgroundColor: 'rgba(229, 231, 235, 0.5)',
            borderColor: 'rgba(156, 163, 175, 0.3)',
            borderWidth: 1,
            drawTime: 'beforeDatasetsDraw' as const
          },
          verticalLine: {
            type: 'line' as const,
            xMin: 3.5,
            xMax: 3.5,
            borderColor: 'rgba(156, 163, 175, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            drawTime: 'beforeDatasetsDraw' as const
          },
        }
      }
    },
    layout: {
      padding: {
        left: 10,
        right: 10,
        top: 20,
        bottom: 20
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 350,
        title: {
          display: false
        },
        ticks: {
          callback: (value: any) => `${value}`,
          stepSize: 50
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      },
      x: {
        offset: true,
        title: {
          display: false
        },
        grid: {
          display: false,
          offset: true
        },
        ticks: {
          padding: 10,
          callback: function(value: any, index: number): string {
            const labels = ['2021', '2022', '2023', '2024', '4개년 평균'];
            return labels[index] || '';
          }
        }
      }
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">기대효과</h1>
        <p className="text-muted-foreground">경쟁사 대비 분석 및 기대효과</p>
      </div>

      {/* 인당 영업이익 차트와 종합분석 - 맨 위에 추가 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>인당 영업이익 추이</CardTitle>
            <CardDescription>주요 경쟁사 대비 인당 영업이익 비교 (단위: 백만원)</CardDescription>
          </CardHeader>
          <CardContent>
            <div style={{ height: '400px', position: 'relative' }}>
              <Line
                key="profit-per-employee-chart"
                data={profitPerEmployeeData}
                options={profitPerEmployeeOptions}
                plugins={[ChartDataLabels]}
              />
            </div>
          </CardContent>
        </Card>

        {/* 종합 분석 */}
        <Card>
          <CardHeader>
            <CardTitle>인당 영업이익 종합분석</CardTitle>
            <CardDescription>2024년 기준 경쟁력 평가</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 핵심 지표 */}
            <div className="bg-blue-50 dark:bg-blue-950/20 p-4 rounded-lg">
              <h4 className="font-semibold text-sm mb-3 text-blue-900 dark:text-blue-200">
                📊 핵심 성과 지표
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">SBL 인당 영업이익</span>
                  <span className="font-bold text-lg">289백만원</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">업계 평균 대비</span>
                  <span className="font-bold text-green-600">+47.4%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">전년 대비 성장률</span>
                  <span className="font-bold text-blue-600">-4.3%</span>
                </div>
              </div>
            </div>

            {/* 경쟁력 분석 */}
            <div className="bg-green-50 dark:bg-green-950/20 p-4 rounded-lg">
              <h4 className="font-semibold text-sm mb-3 text-green-900 dark:text-green-200">
                💡 경쟁력 분석
              </h4>
              <ul className="space-y-2 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>높은 자동화율과 효율적 생산체계로 업계 최고 수준 생산성 달성</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-600 mt-0.5">✓</span>
                  <span>CDMO 고부가가치 제품 비중 확대로 수익성 개선</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-blue-600 mt-0.5">→</span>
                  <span>2025년 신규 라인 증설로 추가 생산성 향상 예상</span>
                </li>
              </ul>
            </div>

            {/* 시사점 */}
            <div className="border-l-4 border-blue-500 pl-4 py-2">
              <p className="text-sm font-medium mb-1">💰 임금인상 여력</p>
              <p className="text-sm text-muted-foreground">
                업계 최고 수준의 인당 영업이익은 충분한 임금인상 여력을 시사하며,
                생산성 기반 보상 체계 구축이 가능함을 보여줍니다.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 경쟁사 비교 차트 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>인건비율 비교</CardTitle>
            <CardDescription>주요 경쟁사 대비 인건비/매출액 비중 추이</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <div style={{ height: '600px', flex: 1, position: 'relative' }}>
                <Line 
                  key="labor-cost-chart"
                  data={laborCostRatioData} 
                  options={laborCostOptions}
                  plugins={[ChartDataLabels]}
                />
              </div>
              <div className="ml-2" style={{ width: '180px' }}>
                <div className="bg-gray-50 rounded-lg p-3">
                  <h4 className="text-xs font-semibold mb-2 text-gray-700">4개년 평균 순위</h4>
                  <div className="space-y-1">
                    {laborCostSorted.map((company, index) => (
                      <div key={company.name} className="flex items-center justify-between text-xs">
                        <div className="flex items-center">
                          <span className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: company.color }}></span>
                          <span className={company.name === 'SBL' ? 'font-bold' : ''}>{company.name}</span>
                        </div>
                        <span className={company.name === 'SBL' ? 'font-bold' : ''}>{company.value.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-900">
                <strong>SBL 현황:</strong> 업계 최저 수준의 인건비율 (4개년 평균 13.6%)을 유지하며 우수한 비용 효율성 달성
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>영업이익률 비교</CardTitle>
            <CardDescription>주요 경쟁사 대비 영업이익/매출액 비중 추이</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center">
              <div style={{ height: '600px', flex: 1, position: 'relative' }}>
                <Line 
                  key="operating-profit-chart"
                  data={operatingProfitRatioData} 
                  options={operatingProfitOptions}
                  plugins={[ChartDataLabels]}
                />
              </div>
              <div className="ml-2" style={{ width: '180px' }}>
                <div className="bg-gray-50 rounded-lg p-3">
                  <h4 className="text-xs font-semibold mb-2 text-gray-700">4개년 평균 순위</h4>
                  <div className="space-y-1">
                    {profitSorted.map((company, index) => (
                      <div key={company.name} className="flex items-center justify-between text-xs">
                        <div className="flex items-center">
                          <span className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: company.color }}></span>
                          <span className={company.name === 'SBL' ? 'font-bold' : ''}>{company.name}</span>
                        </div>
                        <span className={company.name === 'SBL' ? 'font-bold' : ''}>{company.value.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-green-50 rounded-lg">
              <p className="text-sm text-green-900">
                <strong>SBL 현황:</strong> 업계 최고 수준의 영업이익률 (4개년 평균 32.3%)을 유지하며 안정적 수익성 확보
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 종합 분석 */}
      <Card>
        <CardHeader>
          <CardTitle>종합 분석</CardTitle>
          <CardDescription>2026년 임금인상 전략 권고사항</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 border rounded-lg">
              <h3 className="font-semibold mb-2">핵심 인사이트</h3>
              <ul className="space-y-2 text-sm">
                <li>• SBL은 업계 최저 수준의 인건비율(13.6%)과 최상위 영업이익률(32.3%)을 동시에 달성</li>
                <li>• 경쟁사 대비 우수한 비용 효율성으로 임금 인상 여력 충분</li>
                <li>• 2026년 예상 인상률 5.4%는 업계 평균을 상회하지만 재무 건전성 유지 가능</li>
                <li>• 인재 확보 경쟁력 강화를 위한 전략적 임금 인상 필요</li>
              </ul>
            </div>
            
            <div className="p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold mb-2 text-blue-900">권고사항</h3>
              <ul className="space-y-2 text-sm text-blue-800">
                <li>✓ Base-up 3.3% + 성과급 2.1% 구조로 탄력적 운영</li>
                <li>✓ 핵심 인재 대상 차별화된 보상 전략 수립</li>
                <li>✓ 생산성 향상과 연계한 성과 보상 체계 강화</li>
                <li>✓ 분기별 시장 동향 모니터링 및 유연한 대응</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};