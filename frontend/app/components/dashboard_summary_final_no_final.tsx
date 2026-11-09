"use client";
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ChatAssistant from "./ChatAssistant";
import SummaryCard from "./SummaryCard";
const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";
type Regional = {
  regional_id: string;
};
type Compartment = {
  regional_id: string;
  compartment_id: number;
  compartment_code: string;
};
type FiltersResponse = {
  success: boolean;
  filters: {
    regional: Regional[];
    compartment: Compartment[];
    species: string[];
    clone: string[];
  };
};
type TrajectoryRow = {
  age_months: number;
  growth_period: string;
  regional_id: string;
  compartment_id: number;
  compartment_code: string;
  species: string;
  clone: string;
  dbh_cm: number;
  height_m: number;
  volume_m3_tree: number;
  mai_m3_tree_year: number;
  volume_m3_ha: number;
  total_volume_m3: number;
  stand_mai_m3_ha_year: number;
  tha_m?: number;
};
type TrajectoryResponse = {
  success: boolean;
  model: string;
  prediction_cohort: number;
  trajectory_scope?: string;
  planting_year_range?: {
  min: number | null;
  max: number | null;
  };
  filters: {
    regional_id: string;
    compartment_id: number | string;
    species: string;
    clone: string;
  };
  trajectory: TrajectoryRow[];
  row_count: number;
};
type SummaryResponse = {
  success: boolean;
  model?: string;
  prediction_cohort?: number;
  filters?: {
    regional_id?: string;
    compartment_id?: number | string;
    species?: string;
    clone?: string;
  };
  attributes?: {
    area_ha?: number;
    spacing_m?: string;
    spacing_x_m?: string;
    spacing_y_m?: string;
    plant_spacing_m?: string;
    trees_per_ha?: number;
    estimated_trees_compartment?: number;
  };
  summary?: {
    average_dbh_cm?: number;
    average_height_m?: number;
    average_volume_m3_tree?: number;
    average_mai_m3_tree_year?: number;
    average_volume_m3_ha?: number;
    average_total_volume_m3?: number;
    average_stand_mai_m3_ha_year?: number;
    tha_note?: string;
  };
};
type AggregateCompartmentForecast = {
  compartmentCount: number;
  totalEstimatedTrees: number;
  weightedAverageDbh: number | null;
  weightedAverageHeight: number | null;
  accumulatedVolume: number | null;
  predictionCohort: number | null;
};
type MetricKey =
  | "dbh_cm"
  | "height_m"
  | "volume_m3_tree"
  | "mai_m3_tree_year";
type ComparisonRow = {
  growth_period: string;
  age_months: number;
  rf_dbh_cm: number;
  pinns_dbh_cm: number;
  rf_height_m: number;
  pinns_height_m: number;
  rf_volume_m3_tree: number;
  pinns_volume_m3_tree: number;
  rf_mai_m3_tree_year: number;
  pinns_mai_m3_tree_year: number;
};
type ActualTrajectoryRow = {
  growth_period: string;
  age_months_mean: number;
  dbh_cm: number;
  height_m: number;
  volume_m3_tree: number;
  actual_rows: number;
};
type ActualTrajectoryResponse = {
  success: boolean;
  source: string;
  filters: {
    regional_id: string | null;
    compartment_id: number | null;
    species: string;
    clone: string;
  };
  periods: ActualTrajectoryRow[];
};
const DISPLAY_PERIODS = [
  "P6",
  "P12",
  "P18",
  "P24",
  "P30",
  "P36",
  "P42",
  "P48",
  "P54",
  "P60",
  "P66",
];
function getDashboardPeriods(
  rows: TrajectoryRow[]
): TrajectoryRow[] {
  const result: TrajectoryRow[] = [];
  const normalPeriods = [
    "P6",
    "P12",
    "P18",
    "P24",
    "P30",
    "P36",
    "P42",
    "P48",
    "P54",
    "P60",
    "P66",
  ];
  for (const period of normalPeriods) {
    const age = Number(
      period.replace("P", "")
    );
    const row = rows.find(
      (item) =>
        item.growth_period === period &&
        item.age_months === age
    );
    if (row) {
      result.push(row);
    }
  }
  return result;
}
function getMaiPeriods(
  rows: TrajectoryRow[]
): TrajectoryRow[] {
  return rows.filter(
    (row) =>
      row.growth_period !== "P6"
  );
}
function formatNumber(
  value: number,
  decimals = 2
): string {
  if (
    !Number.isFinite(value)
  ) {
    return "-";
  }
  return value.toFixed(
    decimals
  );
}
function getMetricUnit(
  metric: MetricKey
): string {
  switch (metric) {
    case "dbh_cm":
      return "cm";
    case "height_m":
      return "m";
    case "volume_m3_tree":
      return "m³/tree";
    case "mai_m3_tree_year":
      return "m³/tree/year";
  }
}
function getMetricLabel(
  metric: MetricKey
): string {
  switch (metric) {
    case "dbh_cm":
      return "DBH";
    case "height_m":
      return "Height";
    case "volume_m3_tree":
      return "Volume / Tree";
    case "mai_m3_tree_year":
      return "MAI";
  }
}
type ChartSeries = {
  label: string;
  values: number[];
  type:
    | "rf"
    | "pinns"
    | "single";
};
type PeriodLineChartProps = {
  title: string;
  description: string;
  rows: TrajectoryRow[];
  series: ChartSeries[];
  metric: MetricKey;
};
function getSeriesColor(
  type: ChartSeries["type"]
): string {
  if (type === "pinns") {
    return "#0057D9";
  }
  if (type === "single") {
    return "#DC2626";
  }
  return "#008A5B";
}
type ChartView = "line" | "area" | "columns" | "points";
type RechartsPoint = {
  period: string;
  age: number;
  [key: string]: string | number;
};
function PeriodLineChart({
  title,
  description,
  rows,
  series,
  metric,
}: PeriodLineChartProps) {
  const chartRows = rows.filter((row) =>
    DISPLAY_PERIODS.includes(row.growth_period) ||
    row.growth_period === "P66"
  );
  const [view, setView] = useState<ChartView>(() => {
    if (series.length > 1) return "line";
    if (metric === "volume_m3_tree") return "area";
    if (metric === "mai_m3_tree_year") return "columns";
    return "line";
  });
  const chartData: RechartsPoint[] = chartRows.map((row, index) => {
    const point: RechartsPoint = {
      period: row.growth_period,
      age: row.age_months,
    };
    series.forEach((item, seriesIndex) => {
      point[`series_${seriesIndex}`] =
        Number.isFinite(item.values[index]) ? item.values[index] : NaN;
    });
    return point;
  });
  const numberDecimals =
  metric === "volume_m3_tree" || metric === "mai_m3_tree_year"
    ? 4
    : 3;
  const yLabel = `${getMetricLabel(metric)} (${getMetricUnit(metric)})`;
  const tooltipFormatter = (value: unknown, _name: unknown, item: any) => {
    const numeric = Number(value);
    const index = Number(item?.dataKey?.toString().replace("series_", ""));
    const label = Number.isInteger(index)
      ? series[index]?.label ?? "Value"
      : "Value";
        return [
      Number.isFinite(numeric)
        ? `${numeric.toFixed(numberDecimals)} ${getMetricUnit(metric)}`
        : "Data tidak tersedia",
      label,
    ];
  };
  if (chartRows.length === 0) {
    return (
      <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
        <p className="mt-1 text-sm text-slate-500">{description}</p>
        <div className="mt-8 flex h-48 items-center justify-center text-sm text-slate-400">
          No data available.
        </div>
      </div>
    );
  }
  const commonAxisProps = {
    tick: { fontSize: 11, fill: "#475569" },
    axisLine: { stroke: "#94a3b8" },
    tickLine: { stroke: "#cbd5e1" },
  };
  const renderSeries = () =>
  series
    .map((item, index) => ({
      item,
      originalIndex: index,
    }))
    .sort((a, b) => {
  const order = {
    single: 0,
    pinns: 1,
    rf: 2,
  } as const;
  return order[a.item.type] - order[b.item.type];
})
    .map(({ item, originalIndex }) => {
      const dataKey = `series_${originalIndex}`;
      const color = getSeriesColor(item.type);
      if (view === "area") {
        return (
          <Area
            key={`${item.label}-${dataKey}`}
            type="monotone"
            dataKey={dataKey}
            name={item.label}
            stroke={color}
            fill={color}
            fillOpacity={item.type === "single" ? 0.12 : 0.16}
            strokeWidth={
            item.type === "rf"
                ? 4
                : item.type === "single"
                ? 3
                : 3
            }
            dot={{
              r: 3,
              strokeWidth: 2,
              fill: color,
            }}
            activeDot={{
              r: 6,
              strokeWidth: 2,
              stroke: "#fff",
            }}
            connectNulls
            isAnimationActive
            animationDuration={500}
            animationEasing="ease-out"
          />
        );
      }
      if (view === "columns") {
        return (
          <Bar
            key={`${item.label}-${dataKey}`}
            dataKey={dataKey}
            name={item.label}
            fill={color}
            radius={[4, 4, 0, 0]}
            maxBarSize={28}
            opacity={0.86}
            isAnimationActive
            animationDuration={450}
            animationEasing="ease-out"
          />
        );
      }
      return (
  <React.Fragment key={`${item.label}-${dataKey}`}>
    {item.type === "rf" && (
      <Line
        type="monotone"
        dataKey={dataKey}
        name={item.label}
        stroke="#ffffff"
        strokeWidth={7}
        strokeLinecap="round"
        strokeLinejoin="round"
        dot={false}
        activeDot={false}
        connectNulls
        isAnimationActive={false}
      />
    )}
    <Line
      type="monotone"
      dataKey={dataKey}
      name={item.label}
      stroke={color}
      strokeWidth={
        item.type === "rf"
          ? 3
          : item.type === "single"
            ? 4
            : 3
      }
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeDasharray={
        item.type === "rf"
            ? "1 7"
            : item.type === "single"
            ? undefined
            : undefined
        }
      dot={
  item.type === "rf"
    ? {
        r: 2.5,
        strokeWidth: 1,
        fill: color,
      }
    : view === "points"
      ? {
          r: 5,
          strokeWidth: 2,
          fill: color,
        }
      : {
          r: 3,
          strokeWidth: 2,
          fill: color,
        }
}
      activeDot={{
        r: 6,
        strokeWidth: 2,
        stroke: "#fff",
      }}
      connectNulls
      isAnimationActive
      animationDuration={500}
      animationEasing="ease-out"
    />
  </React.Fragment>
);
});
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md">
      <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-slate-500">
              Interactive
            </span>
          </div>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-slate-500">
            {description}
          </p>
        </div>
        <div className="shrink-0">
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Chart view
          </p>
          <div className="flex rounded-lg border border-slate-200 bg-slate-50 p-1">
            {(
              [
                ["line", "Trend"],
                ["area", "Area"],
                ["columns", "Bars"],
                ["points", "Points"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setView(value)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                  view === value
                    ? "bg-white text-slate-900 shadow-sm ring-1 ring-slate-200"
                    : "text-slate-500 hover:text-slate-800"
                }`}
                aria-pressed={view === value}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2 text-xs">
        {series.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <span
              className="h-2.5 w-7 rounded-full"
              style={{ backgroundColor: getSeriesColor(item.type) }}
            />
            <span className="font-medium text-slate-600">{item.label}</span>
          </div>
        ))}
      </div>
      <div className="h-430px w-full">
        <ResponsiveContainer width="100%" height="100%">
          {view === "columns" ? (
            <BarChart
              data={chartData}
              margin={{ top: 12, right: 18, left: 10, bottom: 18 }}
              barCategoryGap="20%"
            >
              <CartesianGrid
                stroke="#DDEBE1"
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="period"
                {...commonAxisProps}
                tickMargin={10}
              />
              <YAxis
                {...commonAxisProps}
                width={72}
                tickFormatter={(value) => Number(value).toFixed(numberDecimals)}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: 5,
                  style: { fill: "#475569", fontSize: 12 },
                }}
              />
              <Tooltip
                cursor={{ fill: "#E3F4EA" }}
                formatter={tooltipFormatter}
                labelFormatter={(label, payload) => {
                  const age = payload?.[0]?.payload?.age;
                  return `${label} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· ${Number(age).toFixed(0)} months`;
                }}
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid #A9D3B8",
                  boxShadow: "0 8px 24px rgba(6,48,31,0.10)",
                }}
              />
              {renderSeries()}
            </BarChart>
          ) : view === "area" ? (
            <AreaChart
              data={chartData}
              margin={{ top: 12, right: 18, left: 10, bottom: 18 }}
            >
              <defs>
                {series.map((item, index) => {
                  const color = getSeriesColor(item.type);
                  return (
                    <linearGradient
                      key={`gradient-${index}`}
                      id={`gradient-${metric}-${index}`}
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="0%" stopColor={color} stopOpacity={0.28} />
                      <stop offset="100%" stopColor={color} stopOpacity={0.02} />
                    </linearGradient>
                  );
                })}
              </defs>
              <CartesianGrid
                stroke="#DDEBE1"
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="period"
                {...commonAxisProps}
                tickMargin={10}
              />
              <YAxis
                {...commonAxisProps}
                width={72}
                tickFormatter={(value) => Number(value).toFixed(numberDecimals)}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: 5,
                  style: { fill: "#475569", fontSize: 12 },
                }}
              />
              <Tooltip
                formatter={tooltipFormatter}
                labelFormatter={(label, payload) => {
                  const age = payload?.[0]?.payload?.age;
                  return `${label} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· ${Number(age).toFixed(0)} months`;
                }}
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid #A9D3B8",
                  boxShadow: "0 8px 24px rgba(6,48,31,0.10)",
                }}
              />
              {series.map((item, index) => {
                const color = getSeriesColor(item.type);
                return (
                  <Area
                    key={`${item.label}-area`}
                    type="monotone"
                    dataKey={`series_${index}`}
                    name={item.label}
                    stroke={color}
                    fill={`url(#gradient-${metric}-${index})`}
                    strokeWidth={3}
                    dot={{ r: 3, strokeWidth: 2, fill: color }}
                    activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
                    connectNulls
                    isAnimationActive
                    animationDuration={500}
                    animationEasing="ease-out"
                  />
                );
              })}
            </AreaChart>
          ) : (
            <LineChart
              data={chartData}
              margin={{ top: 12, right: 18, left: 10, bottom: 18 }}
            >
              <CartesianGrid
                stroke="#DDEBE1"
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="period"
                {...commonAxisProps}
                tickMargin={10}
              />
              <YAxis
                {...commonAxisProps}
                width={72}
                tickFormatter={(value) => Number(value).toFixed(numberDecimals)}
                label={{
                  value: yLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: 5,
                  style: { fill: "#475569", fontSize: 12 },
                }}
              />
              <Tooltip
                formatter={tooltipFormatter}
                labelFormatter={(label, payload) => {
                  const age = payload?.[0]?.payload?.age;
                  return `${label} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· ${Number(age).toFixed(0)} months`;
                }}
                contentStyle={{
                  borderRadius: 10,
                  border: "1px solid #A9D3B8",
                  boxShadow: "0 8px 24px rgba(6,48,31,0.10)",
                }}
              />
              {renderSeries()}
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
      <div className="mt-3 flex flex-col gap-1.5 rounded-lg border border-[#A9D3B8] bg-[#F3FAF5] px-4 py-3 text-xs text-[#4A6556] sm:flex-row sm:items-center sm:justify-between">
        <span>
          <strong className="text-[#0B4F34]">Interactive chart:</strong> hover a
          period to inspect exact values. Curves use monotone interpolation.
        </span>
        <span>
          Periods:{" "}
          <strong className="text-slate-600">
            P6 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P12 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P18 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P24 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P30 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P36 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P42 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P48 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P54 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P60 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ P66
          </strong>
        </span>
      </div>
    </div>
  );
}
export default function TreeGrowthDashboard({
  initialView = "comparison",
  showAssistant = true,
}: {
  initialView?: "comparison" | "actual" | "random_forest" | "pinns";
  showAssistant?: boolean;
}) {
  const [
    filters,
    setFilters,
  ] =
    useState<FiltersResponse | null>(
      null
    );
  const [
    selectedRegional,
    setSelectedRegional,
  ] =
    useState("All");
  const [
    selectedCompartment,
    setSelectedCompartment,
  ] =
    useState("All");
  const [
    selectedSpecies,
    setSelectedSpecies,
  ] =
    useState("All");
  const [
    selectedClone,
    setSelectedClone,
  ] =
    useState("All");
  type DashboardView =
    | "comparison"
    | "actual"
    | "random_forest"
    | "pinns";
  const [
    viewMode,
    setViewMode,
  ] = useState<DashboardView>(initialView
  );
  const [
    selectedModel,
    setSelectedModel,
  ] =
    useState<
      "random_forest" |
      "pinns"
    >(
      "random_forest"
    );
  const [
    showFilterDetails,
    setShowFilterDetails,
  ] = useState(false);
  const [
    trajectory,
    setTrajectory,
  ] =
    useState<TrajectoryResponse | null>(
      null
    );
  const [
    comparisonRF,
    setComparisonRF,
  ] =
    useState<TrajectoryResponse | null>(
      null
    );
  const [
    comparisonPINNs,
    setComparisonPINNs,
  ] =
    useState<TrajectoryResponse | null>(
      null
    );
  const [
    actualTrajectory,
    setActualTrajectory,
  ] =
    useState<ActualTrajectoryResponse | null>(
      null
    );
  const [
    cloneTrajectories,
    setCloneTrajectories,
  ] =
    useState<TrajectoryResponse[]>([]);
  const [
    loadingCloneTable,
    setLoadingCloneTable,
  ] =
    useState(false);
  const [
  compartmentSummary,
  setCompartmentSummary,
] =
  useState<SummaryResponse | null>(
    null
  );
  const [
    aggregateCompartmentForecast,
    setAggregateCompartmentForecast,
  ] = useState<AggregateCompartmentForecast | null>(null);
  const [
    loadingAggregateForecast,
    setLoadingAggregateForecast,
  ] = useState(false);
  const [
    scopeTotalEstimatedTrees,
    setScopeTotalEstimatedTrees,
  ] = useState<number | null>(null);
  const [
    loadingScopeTotalTrees,
    setLoadingScopeTotalTrees,
  ] = useState(false);
  const [
    loadingFilters,
    setLoadingFilters,
  ] =
    useState(true);
  const [
    loadingTrajectory,
    setLoadingTrajectory,
  ] =
    useState(false);
  const [
    error,
    setError,
  ] =
    useState<string | null>(
      null
    );
  const trajectoryRequestId = useRef(0);
  useEffect(() => {
    async function loadFilters() {
      try {
        setLoadingFilters(
          true
        );
        setError(null);
        const response =
          await fetch(
            `${API_URL}/dashboard/filters`
          );
        if (
          !response.ok
        ) {
          throw new Error(
            `Filter API returned ${response.status}`
          );
        }
        const result =
          (await response.json()) as FiltersResponse;
        setFilters(
          result
        );
      } catch (
        err
      ) {
        setError(
          err instanceof
            Error
            ? err.message
            : "Failed to load dashboard filters."
        );
      } finally {
        setLoadingFilters(
          false
        );
      }
    }
    loadFilters();
  }, []);
  const availableCompartments =
    useMemo(() => {
      if (
        !filters
      ) {
        return [];
      }
      if (
        selectedRegional ===
        "All"
      ) {
        return (
          filters.filters
            .compartment
        );
      }
      return (
        filters.filters
          .compartment
          .filter(
            (item) =>
              item.regional_id ===
              selectedRegional
          )
      );
    }, [
      filters,
      selectedRegional,
    ]);
  useEffect(() => {
    if (
      selectedCompartment !==
        "All" &&
      !availableCompartments.some(
        (item) =>
          String(
            item.compartment_id
          ) ===
          selectedCompartment
      )
    ) {
      setSelectedCompartment(
        "All"
      );
    }
  }, [
    availableCompartments,
    selectedCompartment,
  ]);
  const compatibleCloneOptions =
    useMemo(() => {
      return cloneTrajectories
        .map(
          (item) =>
            item.filters.clone ||
            item.trajectory[0]?.clone
        )
        .filter(
          (
            item
          ): item is string =>
            Boolean(item)
        )
        .filter(
          (
            item,
            index,
            array
          ) =>
            array.indexOf(item) ===
            index
        );
    }, [
      cloneTrajectories,
    ]);
  useEffect(() => {
    if (
      !filters
    ) {
      return;
    }
    let cancelled =
      false
    async function loadCompatibleClones() {
      try {
        setLoadingCloneTable(
          true
        );
        const clonePool =
          filters?.filters.clone ?? [];
        const results =
          await Promise.all(
            clonePool.map(
              (clone) =>
                fetchCloneTrajectory(
                  selectedModel,
                  clone
                ).catch(
                  () => null
                )
            )
          );
        if (
          cancelled
        ) {
          return;
        }
        const validResults =
          results.filter(
            (
              item
            ): item is TrajectoryResponse =>
              Boolean(
                item &&
                item.success &&
                item.trajectory &&
                item.trajectory.length > 0
              )
          );
        setCloneTrajectories(
          validResults
        );
      } finally {
        if (
          !cancelled
        ) {
          setLoadingCloneTable(
            false
          );
        }
      }
    }
    loadCompatibleClones();
    return () => {
      cancelled = true;
    };
  }, [
    filters,
    selectedRegional,
    selectedCompartment,
    selectedSpecies,
    selectedModel,
  ]);
  useEffect(() => {
    if (
      selectedClone !==
        "All" &&
      compatibleCloneOptions.length > 0 &&
      !compatibleCloneOptions.includes(
        selectedClone
      )
    ) {
      setSelectedClone(
        "All"
      );
    }
  }, [
    compatibleCloneOptions,
    selectedClone,
  ]);
  function buildQuery(
    model:
      | "random_forest"
      | "pinns"
  ): string {
    const params =
      new URLSearchParams();
    if (
      selectedRegional !==
      "All"
    ) {
      params.set(
        "regional_id",
        selectedRegional
      );
    }
    if (
      selectedCompartment !==
      "All"
    ) {
      params.set(
        "compartment_id",
        selectedCompartment
      );
    }
    if (
      selectedSpecies !==
      "All"
    ) {
      params.set(
        "species",
        selectedSpecies
      );
    }
    if (
      selectedClone !==
      "All"
    ) {
      params.set(
        "clone",
        selectedClone
      );
    }
    params.set(
      "model",
      model
    );
    return params.toString();
  }
async function fetchCompartmentSummary(
  model:
    | "random_forest"
    | "pinns"
): Promise<SummaryResponse> {
  const query =
    buildQuery(model);
  const response =
    await fetch(
      `${API_URL}/dashboard/summary?${query}`
    );
  if (!response.ok) {
    const body =
      await response.text();
    throw new Error(
      `Summary API returned ${response.status}: ${body}`
    );
  }
  return (
    await response.json()
  ) as SummaryResponse;
}
  async function fetchTrajectory(
    model:
      | "random_forest"
      | "pinns"
  ): Promise<TrajectoryResponse> {
    const query =
      buildQuery(
        model
      );
    const response =
      await fetch(
        `${API_URL}/dashboard/trajectory?${query}`
      );
    if (
      !response.ok
    ) {
      const body =
        await response.text();
      throw new Error(
        `Trajectory API returned ${response.status}: ${body}`
      );
    }
    return (
      await response.json()
    ) as TrajectoryResponse;
  }
  async function fetchCloneTrajectory(
    model:
      | "random_forest"
      | "pinns",
    clone: string
  ): Promise<TrajectoryResponse> {
    const params =
      new URLSearchParams();
    if (
      selectedRegional !==
      "All"
    ) {
      params.set(
        "regional_id",
        selectedRegional
      );
    }
    if (
      selectedCompartment !==
      "All"
    ) {
      params.set(
        "compartment_id",
        selectedCompartment
      );
    }
    if (
      selectedSpecies !==
      "All"
    ) {
      params.set(
        "species",
        selectedSpecies
      );
    }
    params.set(
      "clone",
      clone
    );
    params.set(
      "model",
      model
    );
    const response =
      await fetch(
        `${API_URL}/dashboard/trajectory?${params.toString()}`
      );
    if (
      !response.ok
    ) {
      const body =
        await response.text();
      throw new Error(
        `Clone trajectory API returned ${response.status}: ${body}`
      );
    }
    return (
      await response.json()
    ) as TrajectoryResponse;
  }
async function fetchActualTrajectory(): Promise<ActualTrajectoryResponse> {
  const params = new URLSearchParams();
  if (selectedRegional !== "All") {
    params.set(
      "regional_id",
      selectedRegional
    );
  }
  if (selectedCompartment !== "All") {
    params.set(
      "compartment_id",
      selectedCompartment
    );
  }
  if (selectedSpecies !== "All") {
    params.set(
      "species",
      selectedSpecies
    );
  }
  if (selectedClone !== "All") {
    params.set(
      "clone",
      selectedClone
    );
  }
  const response = await fetch(
    `${API_URL}/dashboard/actual-trajectory?${params.toString()}`
  );
  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `Actual trajectory API returned ${response.status}: ${body}`
    );
  }
  return (
    await response.json()
  ) as ActualTrajectoryResponse;
}
  async function loadAggregateCompartmentForecast(
    model: "random_forest" | "pinns"
  ): Promise<AggregateCompartmentForecast | null> {
    if (selectedCompartment !== "All") {
      return null;
    }

    if (availableCompartments.length === 0) {
      return null;
    }
    const results = await Promise.all(
      availableCompartments.map(async (compartment) => {
        const params = new URLSearchParams();
        params.set("compartment_id", String(compartment.compartment_id));
        params.set("model", model);
        if (selectedRegional !== "All") {
          params.set("regional_id", selectedRegional);
        }
        if (selectedSpecies !== "All") {
          params.set("species", selectedSpecies);
        }
        if (selectedClone !== "All") {
          params.set("clone", selectedClone);
        }
        try {
          const [trajectoryResponse, summaryResponse] = await Promise.all([
            fetch(`${API_URL}/dashboard/trajectory?${params.toString()}`),
            fetch(`${API_URL}/dashboard/summary?${params.toString()}`),
          ]);
          if (!trajectoryResponse.ok || !summaryResponse.ok) {
            return null;
          }
          const compartmentTrajectory =
            (await trajectoryResponse.json()) as TrajectoryResponse;
          const compartmentSummary =
            (await summaryResponse.json()) as SummaryResponse;
          const p66 = getDashboardPeriods(
            compartmentTrajectory.trajectory
          ).find((row) => row.growth_period === "P66");
          const estimatedTrees =
            compartmentSummary?.attributes?.estimated_trees_compartment;
          if (
            !p66 ||
            estimatedTrees == null ||
            !Number.isFinite(estimatedTrees) ||
            estimatedTrees <= 0
          ) {
            return null;
          }
          return {
            estimatedTrees,
            dbh: Number(p66.dbh_cm),
            height: Number(p66.height_m),
             volume: Number(p66.volume_m3_tree),
            predictionCohort: compartmentTrajectory.prediction_cohort ?? null,
          };
        } catch {
          return null;
        }
      })
    );
    const valid = results.filter(
      (item) => item !== null
    );
    if (valid.length === 0) {
      return null;
    }
    const totalEstimatedTrees = valid.reduce(
      (sum, item) => sum + item.estimatedTrees,
      0
    );
    const weightedDbh = valid.reduce(
      (sum, item) => sum + item.dbh * item.estimatedTrees,
      0
    );
    const weightedHeight = valid.reduce(
      (sum, item) => sum + item.height * item.estimatedTrees,
      0
    );
    const accumulatedVolume = valid.reduce(
      (sum, item) => sum + item.volume * item.estimatedTrees,
      0
    );
    return {
      compartmentCount: valid.length,
      totalEstimatedTrees,
      weightedAverageDbh:
        totalEstimatedTrees > 0
          ? weightedDbh / totalEstimatedTrees
          : null,
      weightedAverageHeight:
        totalEstimatedTrees > 0
          ? weightedHeight / totalEstimatedTrees
          : null,
      accumulatedVolume: Number.isFinite(accumulatedVolume)
        ? accumulatedVolume
        : null,
      predictionCohort: valid.find((item) => item.predictionCohort != null)
        ?.predictionCohort ?? null,
    };
  }
  async function loadScopeTotalEstimatedTrees(): Promise<number | null> {
    if (!filters) {
      return null;
    }
    const candidates = filters.filters.compartment.filter((item) => {
      const regionalMatch =
        selectedRegional === "All" || item.regional_id === selectedRegional;
      const compartmentMatch =
        selectedCompartment === "All" ||
        String(item.compartment_id) === selectedCompartment;
      return regionalMatch && compartmentMatch;
    });
    if (candidates.length === 0) {
      return null;
    }
    const results = await Promise.all(
      candidates.map(async (item) => {
        const params = new URLSearchParams();
        params.set("regional_id", item.regional_id);
        params.set("compartment_id", String(item.compartment_id));
        params.set("model", selectedModel);
        if (selectedSpecies !== "All") {
          params.set("species", selectedSpecies);
        }
        if (selectedClone !== "All") {
          params.set("clone", selectedClone);
        }
        try {
          const response = await fetch(
            `${API_URL}/dashboard/summary?${params.toString()}`
          );
          if (!response.ok) {
            return null;
          }
          const data = (await response.json()) as SummaryResponse;
          const trees = data.attributes?.estimated_trees_compartment;
          return trees != null && Number.isFinite(trees) && trees > 0
            ? trees
            : null;
        } catch {
          return null;
        }
      })
    );
    const valid = results.filter(
      (value): value is number => value != null && Number.isFinite(value)
    );
    return valid.length > 0
      ? valid.reduce((sum, value) => sum + value, 0)
      : null;
  }
  async function loadTrajectory() {
    const requestId = ++trajectoryRequestId.current;
    try {
      setLoadingTrajectory(true);
      setError(null);
      const [rfResult, pinnsResult, actualResult] = await Promise.all([
        fetchTrajectory("random_forest"),
        fetchTrajectory("pinns"),
        fetchActualTrajectory(),
      ]);
      if (requestId !== trajectoryRequestId.current) {
        return;
      }
      setComparisonRF(rfResult);
      setComparisonPINNs(pinnsResult);
      setActualTrajectory(actualResult);
      const selectedResult =
        selectedModel === "random_forest" ? rfResult : pinnsResult;
      setTrajectory(selectedResult);
      const summaryResult = await fetchCompartmentSummary(selectedModel);
      if (requestId !== trajectoryRequestId.current) {
        return;
      }
      setCompartmentSummary(summaryResult);
      if (selectedCompartment === "All") {
        setLoadingAggregateForecast(true);
        const aggregateResult =
          await loadAggregateCompartmentForecast(selectedModel);
        if (requestId !== trajectoryRequestId.current) {
          return;
        }
        setAggregateCompartmentForecast(aggregateResult);
        setLoadingAggregateForecast(false);
      } else {
        setAggregateCompartmentForecast(null);
      }
      setLoadingScopeTotalTrees(true);
      const totalTrees = await loadScopeTotalEstimatedTrees();
      if (requestId !== trajectoryRequestId.current) {
        return;
      }
      setScopeTotalEstimatedTrees(totalTrees);
      setLoadingScopeTotalTrees(false);
    } catch (err) {
      if (requestId !== trajectoryRequestId.current) {
        return;
      }
      setTrajectory(null);
      setComparisonRF(null);
      setComparisonPINNs(null);
      setCompartmentSummary(null);
      setAggregateCompartmentForecast(null);
      setLoadingAggregateForecast(false);
      setScopeTotalEstimatedTrees(null);
      setLoadingScopeTotalTrees(false);
      setActualTrajectory(null);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load trajectory."
      );
    } finally {
      if (requestId === trajectoryRequestId.current) {
        setLoadingTrajectory(false);
      }
    }
  }
  useEffect(() => {
    if (
      !filters
    ) {
      return;
    }
    loadTrajectory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    filters,
    selectedRegional,
    selectedCompartment,
    selectedSpecies,
    selectedClone,
    selectedModel,
  ]);
  useEffect(() => {
    setViewMode("comparison");
    setShowFilterDetails(false);
  }, [
    selectedRegional,
    selectedCompartment,
    selectedSpecies,
    selectedClone,
  ]);
  function resetFilters() {
    setSelectedRegional(
      "All"
    );
    setSelectedCompartment(
      "All"
    );
    setSelectedSpecies(
      "All"
    );
    setSelectedClone(
      "All"
    );
    setSelectedModel(
      "random_forest"
    );
    setViewMode(
      "comparison"
    );
    setShowFilterDetails(false);
  }
  const selectedRows =
    useMemo(
      () =>
        getDashboardPeriods(
          trajectory?.trajectory ?? []
        ),
      [trajectory]
    );
const actualComparisonRows =
  useMemo(() => {

    if (
      !actualTrajectory ||
      !comparisonRF ||
      !comparisonPINNs
    ) {
      return [];
    }
    const actualRows =
      actualTrajectory.periods;
    const rfRows =
      getDashboardPeriods(
        comparisonRF.trajectory
      );
    const pinnsRows =
      getDashboardPeriods(
        comparisonPINNs.trajectory
      );
    const rfMap =
      new Map(
        rfRows.map(
          (row) => [
            row.growth_period,
            row,
          ]
        )
      );
    const pinnsMap =
      new Map(
        pinnsRows.map(
          (row) => [
            row.growth_period,
            row,
          ]
        )
      );
    return actualRows
      .filter(
        (actual) =>
          DISPLAY_PERIODS.includes(actual.growth_period) ||
          actual.growth_period === "P66"
      )
      .map(
        (actual) => {
          const canonicalPeriod =
            actual.growth_period;
          const rf =
            rfMap.get(canonicalPeriod);
          const pinns =
            pinnsMap.get(canonicalPeriod);
          if (
            !rf ||
            !pinns
          ) {
            return null;
          }
          return {
            growth_period:
              canonicalPeriod,
            age_months:
              actual.age_months_mean,
            actual_dbh_cm:
              actual.dbh_cm,
            actual_height_m:
              actual.height_m,
            actual_volume_m3_tree:
              actual.volume_m3_tree,
            rf_dbh_cm:
              rf.dbh_cm,
            pinns_dbh_cm:
              pinns.dbh_cm,
            rf_height_m:
              rf.height_m,
            pinns_height_m:
              pinns.height_m,
            rf_volume_m3_tree:
              rf.volume_m3_tree,
            pinns_volume_m3_tree:
              pinns.volume_m3_tree,
            rf_mai_m3_tree_year:
              rf.age_months > 0
                ? rf.volume_m3_tree / (rf.age_months / 12)
                : 0,
            pinns_mai_m3_tree_year:
              pinns.age_months > 0
                ? pinns.volume_m3_tree / (pinns.age_months / 12)
                : 0,
          };
        }
      )
      .filter(
        (
          row
        ): row is NonNullable<typeof row> =>
          row !== null
      );
  }, [
    actualTrajectory,
    comparisonRF,
    comparisonPINNs,
  ]);
  function actualComparisonChartRows() {
  return actualComparisonRows.map(
    (row) => ({
      age_months:
        row.age_months,
      growth_period:
        row.growth_period,
      regional_id:
        "",
      compartment_id:
        0,
      compartment_code:
        "",
      species:
        "",
      clone:
        "",
      dbh_cm:
        row.actual_dbh_cm,
      height_m:
        row.actual_height_m,
      volume_m3_tree:
        row.actual_volume_m3_tree,
      mai_m3_tree_year:
        row.age_months > 0
          ? row.actual_volume_m3_tree /
            (row.age_months / 12)
          : 0,
      volume_m3_ha:
        0,
      total_volume_m3:
        0,
      stand_mai_m3_ha_year:
        0,
    })
  );
}
  function volumeComparisonChartRows() {
    return actualComparisonChartRows().filter(
      (row) => row.age_months >= 18
    );
  }
  const latestRow =
    selectedRows.length >
    0
      ? selectedRows[
          selectedRows.length -
            1
        ]
      : null;
  function comparisonChartRows() {
    return actualComparisonRows.map(
      (row) => ({
        age_months:
          row.age_months,
        growth_period:
          row.growth_period,
        regional_id:
          "",
        compartment_id:
          0,
        compartment_code:
          "",
        species:
          "",
        clone:
          "",
        dbh_cm:
          row.rf_dbh_cm,
        height_m:
          row.rf_height_m,
        volume_m3_tree:
          row.rf_volume_m3_tree,
        mai_m3_tree_year:
          row.rf_mai_m3_tree_year,
        volume_m3_ha:
          0,
        total_volume_m3:
          0,
        stand_mai_m3_ha_year:
          0,
      })
    );
  }
  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-7xl px-6 py-6">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Tree Growth Analytics
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Compartment growth prediction dashboard
              </p>
            </div>
          <div className="rounded-lg bg-emerald-50 px-4 py-2 text-sm text-emerald-700">
          <p 
          className="font-medium">
            Prediction Data Range
          </p>
          <strong>
            {trajectory?.planting_year_range?.min != null &&
            trajectory?.planting_year_range?.max != null
              ? `${trajectory.planting_year_range.min}ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ${trajectory.planting_year_range.max}`
              : trajectory?.prediction_cohort ?? "-"}
          </strong>
          <p className="mt-1 text-xs text-emerald-600">
            Aggregated across selected planting cohorts
          </p>
        </div>
          </div>
        </div>
      </header>
      <div className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        <section className="rounded-2xl border border-[#A9D3B8] bg-white p-6 shadow-sm">
          <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-lg font-semibold">
                Dashboard Filters
              </h2>
              <p className="text-sm text-slate-500">
                Select the analysis scope.
              </p>
            </div>
            <button
              onClick={
                resetFilters
              }
              className="rounded-lg border px-4 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-100"
            >
              Reset Filters
            </button>
          </div>
          {loadingFilters ? (
            <div className="rounded-lg bg-slate-50 p-5 text-sm text-slate-500">
              Loading filters...
            </div>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Regional
                </label>
                <select
                  value={
                    selectedRegional
                  }
                  onChange={(
                    event
                  ) =>
                    setSelectedRegional(
                      event.target.value
                    )
                  }
                  className="w-full rounded-lg border border-[#A9D3B8] bg-white px-3 py-2.5 text-sm outline-none focus:border-[#0F7A4A] focus:ring-2 focus:ring-[#2BBF73]/20"
                >
                  <option value="All">
                    All Regional
                  </option>

                  {filters?.filters.regional.map(
                    (item) => (
                      <option
                        key={
                          item.regional_id
                        }
                        value={
                          item.regional_id
                        }
                      >
                        {
                          item.regional_id
                        }
                      </option>
                    )
                  )}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Compartment
                </label>
                <select
                  value={
                    selectedCompartment
                  }
                  onChange={(
                    event
                  ) =>
                    setSelectedCompartment(
                      event.target.value
                    )
                  }
                  className="w-full rounded-lg border border-[#A9D3B8] bg-white px-3 py-2.5 text-sm outline-none focus:border-[#0F7A4A] focus:ring-2 focus:ring-[#2BBF73]/20"
                >
                  <option value="All">
                    All Compartment
                  </option>
                  {availableCompartments.map(
                    (item) => (
                      <option
                        key={
                          item.compartment_id
                        }
                        value={String(
                          item.compartment_id
                        )}
                      >
                        {
                          item.compartment_code
                        }
                      </option>
                    )
                  )}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Species
                </label>
                <select
                  value={
                    selectedSpecies
                  }
                  onChange={(
                    event
                  ) =>
                    setSelectedSpecies(
                      event.target.value
                    )
                  }
                  className="w-full rounded-lg border border-[#A9D3B8] bg-white px-3 py-2.5 text-sm outline-none focus:border-[#0F7A4A] focus:ring-2 focus:ring-[#2BBF73]/20"
                >
                  <option value="All">
                    All Species
                  </option>
                  {filters?.filters.species.map(
                    (item) => (
                      <option
                        key={item}
                        value={item}
                      >
                        {item}
                      </option>
                    )
                  )}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Clone
                </label>
                <select
                  value={
                    selectedClone
                  }
                  onChange={(
                    event
                  ) =>
                    setSelectedClone(
                      event.target.value
                    )
                  }
                  className="w-full rounded-lg border border-[#A9D3B8] bg-white px-3 py-2.5 text-sm outline-none focus:border-[#0F7A4A] focus:ring-2 focus:ring-[#2BBF73]/20"
                >
                  <option value="All">
                    All Clone
                  </option>
                  {compatibleCloneOptions.map(
                    (item) => (
                      <option
                        key={item}
                        value={item}
                      >
                        {item}
                      </option>
                    )
                  )}
                </select>
              </div>
            </div>
          )}
        </section>
        <section className="rounded-2xl border border-[#A9D3B8] bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Analysis View
              </p>
              <p className="mt-1 text-sm text-slate-500">
                Comparison is the default view after applying filters. Open a specific view when you need more detail.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 rounded-xl border border-slate-200 bg-slate-50 p-1.5 sm:grid-cols-4">
              {[
                ["comparison", "Comparison", "All models + actual"],
                ["actual", "Actual", "Historical data"],
                ["random_forest", "Random Forest", "RF forecast"],
                ["pinns", "PINNs V2", "PINNs V2 forecast"],
              ].map(([value, label, hint]) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => {
                    setViewMode(value as DashboardView);
                    if (value === "random_forest" || value === "pinns") {
                      setSelectedModel(value as "random_forest" | "pinns");
                    }
                  }}
                  className={`min-w-132px rounded-lg px-4 py-2.5 text-left transition ${
                    viewMode === value
                      ? value === "comparison"
                        ? "bg-[#06301F] text-white shadow-sm"
                        : value === "actual"
                        ? "bg-[#4A6556] text-white shadow-sm"
                        : value === "random_forest"
                        ? "bg-[#0F7A4A] text-white shadow-sm"
                        : "bg-[#12935A] text-white shadow-sm"
                      : "text-[#4A6556] hover:bg-[#E3F4EA]"
                  }`}
                  aria-pressed={viewMode === value}
                >
                  <span className="block text-sm font-semibold">
                    {label}
                  </span>
                  <span className={`mt-0.5 block text-[10px] ${
                    viewMode === value ? "text-white/75" : "text-slate-400"
                  }`}>
                    {hint}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </section>
        <section className="grid gap-4 md:grid-cols-4">
          <SummaryCard
            label="Regional"
            value={
              selectedRegional
            }
          />
          <SummaryCard
            label="Compartment"
            value={
              selectedCompartment
            }
          />
          <SummaryCard
            label="Species / Clone"
            value={`${selectedSpecies} / ${selectedClone}`}
          />
          <SummaryCard
            label="Current View"
            value={
              viewMode === "comparison"
                ? "Comparison"
                : viewMode === "actual"
                ? "Actual"
                : viewMode === "random_forest"
                ? "Random Forest"
                : "PINNs"
            }
          />
        </section>
<section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
    <div>
      <h2 className="text-xl font-bold text-slate-900">
        Selected Filter Information
      </h2>
      <p className="mt-1 text-sm text-slate-500">
        General information based on the active regional,
        compartment, species, and clone filters.
      </p>
    </div>
    <button
      type="button"
      onClick={() =>
        setShowFilterDetails((current) => !current)
      }
      className="inline-flex w-fit items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-emerald-500 hover:text-emerald-700"
      aria-expanded={showFilterDetails}
    >
      <span>
        {showFilterDetails ? "Hide Detail" : "View Detail"}
      </span>
      <span
        className={`text-xs transition-transform ${
          showFilterDetails ? "rotate-180" : ""
        }`}
        aria-hidden="true"
      >
        ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¼
      </span>
    </button>
  </div>
  {showFilterDetails && (
    compartmentSummary || scopeTotalEstimatedTrees != null ? (
    <>
    <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50/70 p-3">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
        <span className="font-semibold text-slate-700">Active scope:</span>
        <span className="rounded bg-white px-2 py-1 border border-slate-200">
          Regional: {selectedRegional === "All" ? "All Regional" : selectedRegional}
        </span>
        <span className="rounded bg-white px-2 py-1 border border-slate-200">
          Compartment: {selectedCompartment === "All" ? "All Compartment" : selectedCompartment}
        </span>
        <span className="rounded bg-white px-2 py-1 border border-slate-200">
          Species: {selectedSpecies === "All" ? "All Species" : selectedSpecies}
        </span>
        <span className="rounded bg-white px-2 py-1 border border-slate-200">
          Clone: {selectedClone === "All" ? "All Clone" : selectedClone}
        </span>
      </div>
    </div>
    <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {selectedCompartment === "All"
            ? "Selected Area"
            : "Compartment Area"}
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {selectedCompartment === "All"
            ? "Multiple"
            : compartmentSummary?.attributes?.area_ha != null
            ? formatNumber(
                compartmentSummary?.attributes?.area_ha,
                2
              )
            : "-"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          {selectedCompartment === "All"
            ? "multiple selected compartments"
            : "hectares"}
        </p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Plant Spacing
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {compartmentSummary?.attributes?.plant_spacing_m ??
            compartmentSummary?.attributes?.spacing_m ??
            "-"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          spacing between trees
        </p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">

        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Spacing X
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {compartmentSummary?.attributes?.spacing_x_m ?? "-"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          meters
        </p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Spacing Y
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {compartmentSummary?.attributes?.spacing_y_m ?? "-"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          meters
        </p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Trees / ha
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {compartmentSummary?.attributes?.trees_per_ha != null
            ? formatNumber(
                compartmentSummary?.attributes?.trees_per_ha,
                0
              )
            : "-"}
        </p>

        <p className="mt-1 text-xs text-slate-500">
          trees per hectare
        </p>
      </div>
      <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Total Estimated Trees
        </p>
        <p className="mt-3 text-2xl font-bold text-slate-900">
          {loadingScopeTotalTrees
            ? "ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦"
            : scopeTotalEstimatedTrees != null
            ? formatNumber(scopeTotalEstimatedTrees, 0)
            : "-"}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          across the selected filter scope
        </p>
      </div>
      <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-emerald-600">
          Forecast Planting Year
        </p>
        <p className="mt-3 text-2xl font-bold text-emerald-800">
          {compartmentSummary?.prediction_cohort != null
            ? compartmentSummary.prediction_cohort
            : trajectory?.prediction_cohort != null
            ? trajectory.prediction_cohort
            : "-"}
        </p>
        <p className="mt-1 text-xs text-emerald-600">
          next planting cohort
        </p>
      </div>
      <div className="rounded-xl border border-blue-200 bg-blue-50 p-5">
        <p className="text-xs font-semibold uppercase tracking-wider text-blue-600">
          Active Model
        </p>
        <p className="mt-3 text-xl font-bold text-blue-800">
          {viewMode === "comparison"
            ? "Comparison"
            : viewMode === "actual"
            ? "Actual"
            : selectedModel === "random_forest"
            ? "Random Forest"
            : "PINNs V2"}
        </p>
        <p className="mt-1 text-xs text-blue-600">
          current dashboard view
        </p>
      </div>
    </div>
    <p className="mt-4 text-xs text-slate-500">
      Total estimated trees are summed from the compartment-level tree estimates within the active Regional and Compartment scope. This provides context for the accumulated P66 volume shown below.
    </p>
    {selectedCompartment === "All" && (
      <p className="mt-1 text-xs text-slate-400">
        Area, spacing, and trees/ha are compartment-level attributes; when multiple compartments are selected, the detailed cards identify the scope rather than presenting one compartment's attributes as a universal value.
      </p>
    )}
    </>
  ) : (
    <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-8 text-center">
      <p className="text-sm text-slate-500">
        Loading filter details...
      </p>
    </div>
  ))}
  {!showFilterDetails && (
    <div className="mt-4 rounded-lg border border-dashed border-slate-200 bg-slate-50/60 px-4 py-3 text-xs text-slate-500">
      Supporting plantation details are hidden to keep the dashboard compact. Select <span className="font-semibold text-slate-700">View Detail</span> to see area, spacing, trees/ha, estimated trees, forecast cohort, active model, and the active filter scope.
    </div>
  )}
</section>
        {(() => {
          if (selectedCompartment !== "All" && trajectory) {
            const p66
             = getDashboardPeriods(trajectory.trajectory).find(
              (row) => row.growth_period === "P66"
            );
            const estimatedTrees =
              compartmentSummary?.attributes?.estimated_trees_compartment ?? 0;
            const predictedDbh = p66?.dbh_cm;
            const predictedHeight = p66?.height_m;
            const accumulatedVolume =
              p66 && Number.isFinite(p66.volume_m3_tree) && estimatedTrees > 0
                ? p66.volume_m3_tree * estimatedTrees
                : undefined;
            return (
              <section className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex flex-col gap-1 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
                      Compartment Forecast
                    </p>
                    <h2 className="mt-1 text-base font-semibold text-slate-900">
                      Predicted Growth Summary ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Compartment {selectedCompartment}
                    </h2>
                    <p className="mt-1 text-xs text-slate-500">
                      P66 forecast using {selectedModel === "random_forest" ? "Random Forest" : "PINNs V2"}.
                      DBH and height are per-tree predictions; accumulated volume uses P66 volume/tree ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â estimated compartment trees.
                    </p>
                  </div>
                  <span className="w-fit rounded bg-slate-100 px-2 py-1 text-[10px] font-medium text-slate-600">
                    Forecast {trajectory.prediction_cohort} ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· P66
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average predicted DBH</p>
                    <p className="mt-1.5 text-xl font-bold text-slate-900">
                      {predictedDbh != null ? `${formatNumber(predictedDbh, 2)} cm` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-slate-500">At P66ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· per tree</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average predicted Height</p>
                    <p className="mt-1.5 text-xl font-bold text-slate-900">
                      {predictedHeight != null ? `${formatNumber(predictedHeight, 2)} m` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-slate-500">At P66 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· per tree</p>
                  </div>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">Accumulated Volume</p>
                    <p className="mt-1.5 text-xl font-bold text-emerald-800">
                      {accumulatedVolume != null ? `${formatNumber(accumulatedVolume, 2)} mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-emerald-700">P66 volume/tree ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â estimated trees</p>
                  </div>
                </div>
              </section>
            );
          }
          if (selectedCompartment === "All" && aggregateCompartmentForecast) {
            const aggregate = aggregateCompartmentForecast;
            return (
              <section className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex flex-col gap-1 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
                      Forecast Summary
                    </p>
                    <h2 className="mt-1 text-base font-semibold text-slate-900">
                      Predicted Growth Summary ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â All Selected Compartments
                    </h2>
                    <p className="mt-1 text-xs text-slate-500">
                      P66 forecast across {aggregate.compartmentCount} selected compartments using {selectedModel === "random_forest" ? "Random Forest" : "PINNs V2"}.
                      DBH and height use estimated-tree weighting; accumulated volume is the total across the selected scope.
                    </p>
                  </div>
                  <span className="w-fit rounded bg-slate-100 px-2 py-1 text-[10px] font-medium text-slate-600">
                    Forecast cohort {aggregate.predictionCohort ?? "-"} Â· P66
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average predicted DBH</p>
                    <p className="mt-1.5 text-xl font-bold text-slate-900">
                      {aggregate.weightedAverageDbh != null ? `${formatNumber(aggregate.weightedAverageDbh, 2)} cm` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-slate-500"> P66 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· weighted by estimated trees</p>
                  </div>
                  <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Average predicted Height</p>
                    <p className="mt-1.5 text-xl font-bold text-slate-900">
                      {aggregate.weightedAverageHeight != null ? `${formatNumber(aggregate.weightedAverageHeight, 2)} m` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-slate-500"> P66 ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· weighted by estimated trees</p>
                  </div>
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-emerald-700">Accumulated Volume</p>
                    <p className="mt-1.5 text-xl font-bold text-emerald-800">
                      {aggregate.accumulatedVolume != null ? `${formatNumber(aggregate.accumulatedVolume, 2)} mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³` : "-"}
                    </p>
                    <p className="mt-0.5 text-[10px] text-emerald-700"> P66 total across selected compartments</p>
                  </div>
                </div>
              </section>
            );
          }
          if (selectedCompartment === "All" && loadingAggregateForecast) {
            return (
              <section className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-700">Forecast Summary</p>
                    <h2 className="mt-1 text-base font-semibold text-slate-900">Calculating selected-compartment summary...</h2>
                  </div>
                  <span className="text-xs text-slate-500">P66</span>
                </div>
              </section>
            );
          }
          return null;
        })()}
        {(viewMode === "random_forest" ||
          viewMode === "pinns") &&
          selectedRows.length > 0 && (
          <section>
            <div className="mb-5">
              <h2 className="text-2xl font-bold">
                Growth Trajectory
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Prediction by period:
              </p>
            </div>
            <div className="space-y-6">
              <PeriodLineChart
                key={`dbh-${selectedModel}`}
                title={`Stand Yield & Growth Trajectory ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â DBH ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ${
                  selectedModel ===
                  "random_forest"
                    ? "Random Forest"
                    : "PINNs"
                }`}
                description="Diameter growth across the defined growth periods."
                rows={
                  selectedRows
                }
                metric="dbh_cm"
                series={[
                  {
                    label:
                      selectedModel ===
                      "random_forest"
                        ? "Random Forest"
                        : "PINNs V2",
                    type:
                      selectedModel ===
                      "random_forest"
                        ? "rf"
                        : "pinns",
                    values:
                      selectedRows.map(
                        (row) =>
                          row.dbh_cm
                      ),
                  },
                ]}
              />
              <PeriodLineChart
                key={`height-${selectedModel}`}
                title={`Stand Yield & Growth Trajectory ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Height ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ${
                  selectedModel ===
                  "random_forest"
                    ? "Random Forest"
                    : "PINNs"
                }`}
                description="Height growth across the defined growth periods."
                rows={
                  selectedRows
                }
                metric="height_m"
                series={[
                  {
                    label:
                      selectedModel ===
                      "random_forest"
                        ? "Random Forest"
                        : "PINNs V2",
                    type:
                      selectedModel ===
                      "random_forest"
                        ? "rf"
                        : "pinns",
                    values:
                      selectedRows.map(
                        (row) =>
                          row.height_m
                      ),
                  },
                ]}
              />
              <PeriodLineChart
                key={`volume-${selectedModel}`}
                title={`Stand Yield & Growth Trajectory ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Volume ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ${
                  selectedModel ===
                  "random_forest"
                    ? "Random Forest"
                    : "PINNs"
                }`}
                description="Predicted volume per tree across the defined growth periods."
                rows={
                  selectedRows
                }
                metric="volume_m3_tree"
                series={[
                  {
                    label:
                      selectedModel ===
                      "random_forest"
                        ? "Random Forest"
                        : "PINNs V2",
                    type:
                      selectedModel ===
                      "random_forest"
                        ? "rf"
                        : "pinns",
                    values:
                      selectedRows.map(
                        (row) =>
                          row.volume_m3_tree
                      ),
                  },
                ]}
              />
<PeriodLineChart
  key={`mai-${selectedModel}`}
  title={`Stand Yield & Growth Trajectory ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â MAI ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ${
    selectedModel ===
    "random_forest"
      ? "Random Forest"
      : "PINNs"
  }`}
  description="Mean annual increment from P12 through P66."
  rows={getMaiPeriods(
    selectedRows
  )}
  metric="mai_m3_tree_year"
  series={[
    {
      label:
        selectedModel ===
        "random_forest"
          ? "Random Forest"
          : "PINNs V2",
      type:
        selectedModel ===
        "random_forest"
          ? "rf"
          : "pinns",
      values:
        getMaiPeriods(
          selectedRows
        ).map(
          (row) =>
            row.mai_m3_tree_year
        ),
    },
  ]}
/>
            </div>
          </section>
        )}
                {viewMode === "actual" && (
          <section>
            <div className="mb-5">
              <h2 className="text-2xl font-bold text-slate-900">
                Historical Actual Growth
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Historical observations for the selected filters. This view contains actual data only.
              </p>
            </div>
            {actualComparisonRows.length > 0 ? (
              <div className="space-y-6">
                <PeriodLineChart
                  title="DBH ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Historical Actual"
                  description="Observed diameter by growth period."
                  rows={actualComparisonChartRows()}
                  metric="dbh_cm"
                  series={[{
                    label: "Actual",
                    type: "single",
                    values: actualComparisonRows.map((row) => row.actual_dbh_cm),
                  }]}
                />
                <PeriodLineChart
                  title="Height ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Historical Actual"
                  description="Observed height by growth period."
                  rows={actualComparisonChartRows()}
                  metric="height_m"
                  series={[{
                    label: "Actual",
                    type: "single",
                    values: actualComparisonRows.map((row) => row.actual_height_m),
                  }]}
                />
                <PeriodLineChart
                  title="Volume ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Historical Actual"
                  description="Observed volume per tree by growth period."
                  rows={actualComparisonChartRows()}
                  metric="volume_m3_tree"
                  series={[{
                    label: "Actual",
                    type: "single",
                    values: actualComparisonRows.map((row) => row.actual_volume_m3_tree),
                  }]}
                />
                <PeriodLineChart
                  title="MAI ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Historical Actual"
                  description="Mean annual increment calculated from observed volume, starting at P12."
                  rows={getMaiPeriods(actualComparisonChartRows())}
                  metric="mai_m3_tree_year"
                  series={[{
                    label: "Actual",
                    type: "single",
                    values: getMaiPeriods(actualComparisonChartRows()).map((row) => row.mai_m3_tree_year),
                  }]}
                />
              </div>
            ) : (
              <div className="rounded-xl border border-slate-200 bg-white p-8 text-center shadow-sm">
                <p className="text-sm text-slate-500">
                  Historical actual data is not available for the selected filters.
                </p>
              </div>
            )}
          </section>
        )}
        {viewMode === "comparison" && (
        <section>
          <div className="mb-5">
          </div>
          {actualComparisonRows.length >
          0 ? (
            <div className="space-y-6">
<section>
  <div className="mb-5">
    <h2 className="text-2xl font-bold">
      Model Performance Comparison
    </h2>
    <p className="mt-1 text-sm text-slate-500">
      Test-set performance of Random Forest and PINNs V2.
    </p>
  </div>
  <div className="overflow-x-auto rounded-xl border bg-white shadow-sm">
    <table className="w-full min-w-900px text-left text-sm">
      <thead>
        <tr className="border-b bg-slate-50">
          <th className="px-5 py-4 font-semibold">
            Target
          </th>
          <th className="px-5 py-4 font-semibold">
            Model
          </th>
          <th className="px-5 py-4 font-semibold">
            MAE
          </th>
          <th className="px-5 py-4 font-semibold">
            RMSE
          </th>
          <th className="px-5 py-4 font-semibold">
            MAPE
          </th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-b">
          <td
            rowSpan={2}
            className="px-5 py-4 font-semibold"
          >
            DBH
            <span className="block text-xs font-normal text-slate-400">
              cm
            </span>
          </td>
          <td className="px-5 py-4">
            Random Forest
          </td>
          <td className="px-5 py-4">
            0.423406
          </td>
          <td className="px-5 py-4">
            0.533821
          </td>
          <td className="px-5 py-4">
            0.988134
          </td>
          <td className="px-5 py-4">
            3.3188%
          </td>
        </tr>
        <tr className="border-b">
          <td className="px-5 py-4">
            PINNs V2
          </td>
          <td className="px-5 py-4">
            1.103119
          </td>
          <td className="px-5 py-4">
            1.359792
          </td>
          <td className="px-5 py-4">
            0.923007
          </td>
          <td className="px-5 py-4">
            12.6193%
          </td>
        </tr>
        <tr className="border-b">
          <td className="px-5 py-4">
            Random Forest
          </td>
          <td className="px-5 py-4">
            0.552280
          </td>
          <td className="px-5 py-4">
            0.697006
          </td>
          <td className="px-5 py-4">
            0.988414
          </td>
          <td className="px-5 py-4">
            3.1253%
          </td>
        </tr>
        <tr className="border-b">
          <td className="px-5 py-4">
            PINNs V2
          </td>
          <td className="px-5 py-4">
            1.576653
          </td>
          <td className="px-5 py-4">
            1.979054
          </td>
          <td className="px-5 py-4">
            0.906593
          </td>
          <td className="px-5 py-4">
            13.8617%
          </td>
        </tr>
        <tr className="border-b">
          <td
            rowSpan={2}
            className="px-5 py-4 font-semibold"
          >
            Volume
            <span className="block text-xs font-normal text-slate-400">
              mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³/tree
            </span>
          </td>
          <td className="px-5 py-4">
            Random Forest
          </td>
          <td className="px-5 py-4">
            0.026477
          </td>
          <td className="px-5 py-4">
            0.037866
          </td>
          <td className="px-5 py-4">
            0.868916
          </td>
          <td className="px-5 py-4">
            15.5522%
          </td>
        </tr>
        <tr>
          <td className="px-5 py-4">
            PINNs V2
          </td>
          <td className="px-5 py-4">
            0.033233
          </td>
          <td className="px-5 py-4">
            0.045006
          </td>
          <td className="px-5 py-4">
            0.814824
          </td>
          <td className="px-5 py-4">
            23.7649%
          </td>
        </tr>
      </tbody>
    </table>
  </div>
  <div className="mt-4 rounded-xl border border-blue-100 bg-blue-50 p-4">
    <p className="text-sm font-semibold text-blue-900">
      PINNs physical constraint
    </p>
    <p className="mt-1 text-sm leading-6 text-blue-800">
      The PINNs V2 model applies a post-processing positivity
      constraint to predicted tree volume: max(raw prediction, 0).
      Before this constraint, 296 of 35,640 test volume predictions
      were negative; all 296 occurred at P18.
    </p>
  </div>
</section>
               <PeriodLineChart
                title="Model Forecast Comparison ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â DBH"
                description="Historical actual data compared with Random Forest and PINNs V2 forecasts."
                rows={actualComparisonChartRows()}
                metric="dbh_cm"
                series={[
                  {
                    label: "Actual",
                    type: "single",
                    values: actualComparisonRows.map(
                      (row) => row.actual_dbh_cm
                    ),
                  },
                  {
                    label: "Random Forest",
                    type: "rf",
                    values: actualComparisonRows.map(
                      (row) => row.rf_dbh_cm
                    ),
                  },
                  {
                    label: "PINNs V2",
                    type: "pinns",
                    values: actualComparisonRows.map(
                      (row) => row.pinns_dbh_cm
                    ),
                  },
                ]}
              />
              <PeriodLineChart
                title="Model Forecast Comparison ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Height"
                description="Historical actual data compared with Random Forest and PINNs V2 forecasts."
                rows={actualComparisonChartRows()}
                metric="height_m"
                series={[
                  {
                    label: "Actual",
                    type: "single",
                    values: actualComparisonRows.map(
                      (row) => row.actual_height_m
                    ),
                  },
                  {
                    label: "Random Forest",
                    type: "rf",
                    values: actualComparisonRows.map(
                      (row) => row.rf_height_m
                    ),
                  },
                  {
                    label: "PINNs V2",
                    type: "pinns",
                    values: actualComparisonRows.map(
                      (row) => row.pinns_height_m
                    ),
                  },
                ]}
              />
              <PeriodLineChart
                title="Model Forecast Comparison ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â Volume"
                description="Historical actual data compared with Random Forest and PINNs V2 forecasts from P18 to P66."
                rows={volumeComparisonChartRows()}
                metric="volume_m3_tree"
                series={[
                  {
                    label: "Actual",
                    type: "single",
                    values: volumeComparisonChartRows().map(
                      (row) => row.volume_m3_tree
                    ),
                  },
                  {
                    label: "Random Forest",
                    type: "rf",
                    values: volumeComparisonChartRows().map(
                      (row) =>
                        actualComparisonRows.find(
                          (item) => item.growth_period === row.growth_period
                        )?.rf_volume_m3_tree ??NaN
                    ),
                  },
                  {
                    label: "PINNs V2",
                    type: "pinns",
                    values: volumeComparisonChartRows().map(
                      (row) =>
                        actualComparisonRows.find(
                          (item) => item.growth_period === row.growth_period
                        )?.pinns_volume_m3_tree ?? NaN
                    ),
                  },
                ]}
              />
              <PeriodLineChart
                title="Model Forecast Comparison ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â MAI"
                description="Model comparison from P12 through P66. MAI starts at P12."
                rows={getMaiPeriods(comparisonChartRows())}
                metric="mai_m3_tree_year"
                series={[
                  {
                    label: "Random Forest",
                    type: "rf",
                    values: getMaiPeriods(comparisonChartRows()).map(
                      (row) => row.mai_m3_tree_year
                    ),
                  },
                  {
                    label: "PINNs V2",
                    type: "pinns",
                    values: getMaiPeriods(comparisonChartRows()).map(
                      (row, index) =>
                        actualComparisonRows
                          .filter((item) => item.growth_period !== "P6")[index]
                          ?.pinns_mai_m3_tree_year ?? 0
                    ),
                  },
                ]}
              />
            </div>
          ) : (
            <div className="rounded-xl border bg-white p-8 text-center shadow-sm">
              {loadingTrajectory ? (
                <p className="text-sm text-slate-500">
                  Loading Random Forest and PINNs comparison...
                </p>
              ) : (
                <p className="text-sm text-slate-500">
                  Comparison data is not available.
                </p>
              )}
            </div>
          )}
        </section>
        )}
        {(viewMode === "random_forest" ||
          viewMode === "pinns") &&
          cloneTrajectories.length > 0 && (
          <section className="rounded-xl border border-slate-200 bg-white shadow-sm">
            <div className="border-b border-slate-200 px-5 py-4">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-700">
                    Growth Trajectory
                  </p>
                  <h2 className="mt-1 text-base font-semibold text-slate-900">
                    Forecast Growth by Clone
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    Comparison of DBH, height, and volume for each compatible clone.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-slate-500">
                  <span className="rounded bg-slate-100 px-2 py-1">
                    {cloneTrajectories.length} clone
                    {cloneTrajectories.length === 1 ? "" : "s"}
                  </span>
                  <span className="rounded bg-emerald-50 px-2 py-1 text-emerald-700">
                    Forecast cohort
                  </span>
                </div>
              </div>
            </div>
            {loadingCloneTable ? (
              <div className="px-5 py-8 text-center text-xs text-slate-500">
                Loading compatible clone trajectories...
              </div>
            ) : (
              <div className="overflow-x-auto">
                {(() => {
                  const cloneRows =
                    cloneTrajectories.map(
                      (item) => {
                        const rows =
                          getDashboardPeriods(
                            item.trajectory
                          );
                        const p66 =
                          rows.find(
                            (row) =>
                              row.growth_period ===
                              "P66"
                          );
                        return {
                            clone:
                                item.filters.clone ||
                                rows[0]?.clone ||
                                "-",
                            rows,
                            p66,
                            };
                      }
                    )
                    .filter(
                      (item) =>
                        item.rows.length > 0
                    );
                  const validP66 =
                    cloneRows.filter(
                      (item) =>
                        item.p66 &&
                        Number.isFinite(
                          item.p66.dbh_cm
                        ) &&
                        Number.isFinite(
                          item.p66.height_m
                        ) &&
                        Number.isFinite(
                          item.p66.volume_m3_tree
                        )
                    );
                  let bestClone: string | null =
                    null;
                  if (
                    selectedClone === "All" &&
                    validP66.length > 0
                  ) {
                    const ranges = {
                      dbh:
                        Math.max(
                          ...validP66.map(
                            (item) =>
                              item.p66!.dbh_cm
                          )
                        ) -
                        Math.min(
                          ...validP66.map(
                            (item) =>
                              item.p66!.dbh_cm
                          )
                        ),
                      height:
                        Math.max(
                          ...validP66.map(
                            (item) =>
                              item.p66!.height_m
                          )
                        ) -
                        Math.min(
                          ...validP66.map(
                            (item) =>
                              item.p66!.height_m
                          )
                        ),
                      volume:
                        Math.max(
                          ...validP66.map(
                            (item) =>
                              item.p66!.volume_m3_tree
                          )
                        ) -
                        Math.min(
                          ...validP66.map(
                            (item) =>
                              item.p66!.volume_m3_tree
                          )
                        ),
                    };
                    const score =
                      (
                        item: (typeof validP66)[number]
                      ) => {
                        const minDbh =
                          Math.min(
                            ...validP66.map(
                              (x) =>
                                x.p66!.dbh_cm
                            )
                          );
                        const minHeight =
                          Math.min(
                            ...validP66.map(
                              (x) =>
                                x.p66!.height_m
                            )
                          );
                        const minVolume =
                          Math.min(
                            ...validP66.map(
                              (x) =>
                                x.p66!.volume_m3_tree
                            )
                          );
                        const dbhScore =
                          ranges.dbh > 0
                            ? (
                                item.p66!.dbh_cm -
                                minDbh
                              ) /
                              ranges.dbh
                            : 1;
                        const heightScore =
                          ranges.height > 0
                            ? (
                                item.p66!.height_m -
                                minHeight
                              ) /
                              ranges.height
                            : 1;
                        const volumeScore =
                          ranges.volume > 0
                            ? (
                                item.p66!.volume_m3_tree -
                                minVolume
                              ) /
                              ranges.volume
                            : 1;
                        return (
                          dbhScore +
                          heightScore +
                          volumeScore
                        ) / 3;
                      };
                    bestClone =
                      [...validP66]
                        .sort(
                          (a, b) =>
                            score(b) -
                            score(a)
                        )[0]
                        ?.clone ??
                      null;
                  }
                  const displayedCloneRows =
                    selectedClone === "All"
                      ? cloneRows
                      : cloneRows.filter(
                          (item) =>
                            item.clone ===
                            selectedClone
                        );
                  return (
                    <>
                      <div className="border-b border-slate-100 bg-slate-50/70 px-5 py-2.5">
                        <p className="text-[10px] leading-4 text-slate-500">
                          {selectedClone === "All" &&
                          bestClone ? (
                            <>
                              <span className="font-semibold text-emerald-700">
                                Best clone: {bestClone}
                              </span>
                              {" ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â "}
                              determined from P66 DBH, height, and volume/tree
                              with equal weighting (33.33% each).
                            </>
                          ) : (
                            <>
                              <span className="font-semibold text-slate-700">
                                Selected clone: {selectedClone}
                              </span>
                              {" ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â "}
                              clone ranking is not applied when a specific clone is selected.
                            </>
                          )}
                        </p>
                      </div>
                      <table className="min-w-max w-full border-collapse text-[11px] text-slate-700">
                        <thead>
                          <tr className="bg-slate-100 text-slate-700">
                            <th
                              rowSpan={2}
                              className="sticky left-0 z-20 border-b border-r border-slate-200 bg-slate-100 px-3 py-2 text-left font-semibold"
                            >
                              Period
                            </th>
                            {displayedCloneRows.map(
                              (item) => {
                                const isBest =
                                  selectedClone === "All" &&
                                  item.clone === bestClone;
                                return (
                                  <th
                                    key={item.clone}
                                    colSpan={3}
                                    className={`border-b border-r border-slate-200 px-2 py-2 text-center font-semibold ${
                                      isBest
                                        ? "bg-emerald-100 text-emerald-900"
                                        : "bg-slate-100 text-slate-700"
                                    }`}
                                  >
                                    {item.clone}
                                    {isBest && (
                                      <span className="ml-1 text-[9px] font-medium text-emerald-700">
                                        BEST
                                      </span>
                                    )}
                                  </th>
                                );
                              }
                            )}
                          </tr>
                          <tr className="bg-white text-[10px] text-slate-500">
                            {displayedCloneRows.flatMap(
                              (item) => {
                                const isBest =
                                  selectedClone === "All" &&
                                  item.clone === bestClone;
                                return [
                                  <th
                                    key={`${item.clone}-dbh`}
                                    className={`border-b border-r border-slate-200 px-2 py-1.5 text-center font-medium ${
                                      isBest
                                        ? "bg-emerald-50 text-emerald-800"
                                        : "bg-white"
                                    }`}
                                  >
                                    DBH
                                  </th>,
                                  <th
                                    key={`${item.clone}-height`}
                                    className={`border-b border-r border-slate-200 px-2 py-1.5 text-center font-medium ${
                                      isBest
                                        ? "bg-emerald-50 text-emerald-800"
                                        : "bg-white"
                                    }`}
                                  >
                                    Height
                                  </th>,
                                  <th
                                    key={`${item.clone}-volume`}
                                    className={`border-b border-r border-slate-200 px-2 py-1.5 text-center font-medium ${
                                      isBest
                                        ? "bg-emerald-50 text-emerald-800"
                                        : "bg-white"
                                    }`}
                                  >
                                    Volume
                                  </th>,
                                ];
                              }
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {DISPLAY_PERIODS.map(
                            (period) => {
                              const periodRows =
                                displayedCloneRows.map(
                                  (item) => {
                                    const lookupPeriod =
                                      period === "P66"
                                        ? "P66"
                                        : period;
                                    const row =
                                      item.rows.find(
                                        (candidate) =>
                                          candidate.growth_period ===
                                          lookupPeriod
                                      );
                                    return {
                                      clone:
                                        item.clone,
                                      row,
                                    };
                                  }
                                );
                              const age =
                                periodRows.find(
                                  (item) =>
                                    item.row
                                )?.row?.age_months;
                              return (
                                <tr
                                  key={period}
                                  className="border-b border-slate-200 last:border-0"
                                >
                                  <td className="sticky left-0 z-10 border-r border-slate-200 bg-white px-3 py-1.5 font-semibold text-slate-800">
                                    <div>
                                      {period === "P66" ? "P66" : period}
                                    </div>
                                    <div className="text-[9px] font-normal text-slate-400">
                            {age ?? "-"} mo
                                    </div>
                                  </td>
                                  {periodRows.map(
                                    (item) => {

                                      const isBest =
                                        selectedClone === "All" &&
                                        item.clone === bestClone;
                                      const row =
                                        item.row;
                                      return (
                                        <React.Fragment
                                          key={`${period}-${item.clone}`}
                                        >
                                          <td
                                            className={`border-r border-slate-200 px-2 py-1.5 text-center tabular-nums ${
                                              isBest
                                                ? "bg-emerald-50/80"
                                                : "bg-white"
                                            }`}
                                          >
                                            {row
                                              ? formatNumber(
                                                  row.dbh_cm
                                                )
                                              : "-"}
                                          </td>
                                          <td
                                            className={`border-r border-slate-200 px-2 py-1.5 text-center tabular-nums ${
                                              isBest
                                                ? "bg-emerald-50/80"
                                                : "bg-white"
                                            }`}
                                          >
                                            {row
                                              ? formatNumber(
                                                  row.height_m
                                                )
                                              : "-"}
                                          </td>
                                          <td
                                            className={`border-r border-slate-200 px-2 py-1.5 text-center tabular-nums ${
                                              isBest
                                                ? "bg-emerald-50/80"
                                                : "bg-white"
                                            }`}
                                          >
                                            {row
                                              ? formatNumber(
                                                  row.volume_m3_tree,
                                                  4
                                                )
                                              : "-"}
                                          </td>
                                        </React.Fragment>
                                      );
                                    }
                                  )}
                                </tr>
                              );
                            }
                          )}
                        </tbody>
                      </table>
                      <div className="border-t border-slate-200 px-5 py-2 text-[9px] text-slate-400">
                        Units: DBH (cm) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· Height (m) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â· Volume/tree (mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³/tree) ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â·
                        P66 = P66 (66 months), the official terminal age.
                      </div>
                    </>
                  );
                })()}
              </div>
            )}
          </section>
        )}
        {/* AI GROWTH ASSISTANT */}
        {showAssistant && <ChatAssistant />}
      </div>
    </main>
  );
}



