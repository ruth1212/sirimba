"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import DatasetHistory from "../components/dataset/DatasetHistory";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type MaiYear = {
  planting_year: number;
  cohort_area_ha: number;
  compartment_count: number;
  mai_weighted_m3_ha_year: number;
  mai_min_m3_ha_year: number;
  mai_max_m3_ha_year: number;
  good_count: number;
  below_threshold_count: number;
};

type MaiDetail = {
  regional_id: string;
  regional_name: string;
  compartment_id: number;
  compartment_code: string;
  planting_year: number;
  area_ha: number;
  age_months: number;
  growth_period: string;
  species: string;
  clone: string;
  measurement_rows: number;
  mean_volume_m3_tree: number;
  trees_per_ha: number;
  estimated_volume_m3_ha: number;
  mai_m3_ha_year: number;
  status: string;
};

type MaiResponse = {
  status: string;
  source: string;
  filters: Record<string, unknown>;
  summary: {
    cohort_compartment_count: number;
    cohort_area_ha: number;
    mai_weighted_m3_ha_year: number;
    threshold_m3_ha_year: number;
    good_count: number;
    below_threshold_count: number;
    mai_min_m3_ha_year: number;
    mai_max_m3_ha_year: number;
  };
  by_planting_year: MaiYear[];
  details: MaiDetail[];
  methodology: {
    source_metric: string;
    estimated_volume_formula: string;
    mai_formula: string;
    weighted_formula: string;
    threshold_rule: string;
    density_note: string;
    area_note: string;
    prediction_used: boolean;
  };
};

const number = (value: number | null | undefined, digits = 2) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("id-ID", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "—";

const integer = (value: number | null | undefined) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("id-ID", { maximumFractionDigits: 0 })
    : "—";

export default function MaiReportPage() {
  const [report, setReport] = useState<MaiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [yearStart, setYearStart] = useState("");
  const [yearEnd, setYearEnd] = useState("");
  const [regional, setRegional] = useState("");
  const [compartment, setCompartment] = useState("");
  const [species, setSpecies] = useState("");
  const [clone, setClone] = useState("");
  const [search, setSearch] = useState("");

  const loadReport = useCallback(async () => {
    setLoading(true);
    setError("");

    const params = new URLSearchParams();
    if (yearStart) params.set("planting_year_start", yearStart);
    if (yearEnd) params.set("planting_year_end", yearEnd);
    if (regional) params.set("regional_id", regional);
    if (compartment) params.set("compartment_id", compartment);
    if (species) params.set("species", species);
    if (clone) params.set("clone", clone);

    try {
      const response = await fetch(
        `${API_URL}/dashboard/mai-report?${params.toString()}`,
        { cache: "no-store" }
      );

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`API ${response.status}: ${body.slice(0, 240)}`);
      }

      const data = (await response.json()) as MaiResponse;
      setReport(data);
    } catch (err) {
      setReport(null);
      setError(
        err instanceof Error
          ? err.message
          : "Gagal memuat laporan MAI."
      );
    } finally {
      setLoading(false);
    }
  }, [yearStart, yearEnd, regional, compartment, species, clone]);

  useEffect(() => {
    void loadReport();
  }, [loadReport]);

  const visibleDetails = useMemo(() => {
    const rows = report?.details ?? [];
    const q = search.trim().toLowerCase();

    if (!q) return rows;

    return rows.filter((row) =>
      [
        row.regional_id,
        row.regional_name,
        row.compartment_code,
        String(row.planting_year),
        row.species,
        row.clone,
        row.status,
      ]
        .join(" ")
        .toLowerCase()
        .includes(q)
    );
  }, [report, search]);

  const regionalOptions = useMemo(
    () => [...new Set((report?.details ?? []).map((row) => row.regional_id))].sort(),
    [report]
  );
  const compartmentOptions = useMemo(
    () =>
      [...new Map(
        (report?.details ?? []).map((row) => [
          String(row.compartment_id),
          row.compartment_code,
        ])
      ).entries()].sort((a, b) => a[1].localeCompare(b[1])),
    [report]
  );
  const speciesOptions = useMemo(
    () =>
      [...new Set(
        (report?.details ?? []).flatMap((row) =>
          row.species.split(", ").map((value) => value.trim())
        )
      )].sort(),
    [report]
  );
  const cloneOptions = useMemo(
    () =>
      [...new Set(
        (report?.details ?? []).flatMap((row) =>
          row.clone.split(", ").map((value) => value.trim())
        )
      )].sort(),
    [report]
  );

  function resetFilters() {
    setYearStart("");
    setYearEnd("");
    setRegional("");
    setCompartment("");
    setSpecies("");
    setClone("");
    setSearch("");
  }

  function exportCsv() {
    const rows = visibleDetails;
    if (!rows.length) return;

    const headers = [
      "Regional",
      "Kompartemen",
      "Tahun Tanam",
      "Luas (ha)",
      "Umur (bulan)",
      "Periode",
      "Jumlah Baris Pengukuran",
      "Rata-rata Volume per Pohon (m3)",
      "Pohon per ha Tercatat",
      "Estimasi Volume (m3/ha)",
      "MAI (m3/ha/tahun)",
      "Status",
    ];

    const escapeCsv = (value: unknown) =>
      `"${String(value ?? "").replace(/"/g, '""')}"`;

    const body = rows.map((row) =>
      [
        row.regional_id,
        row.compartment_code,
        row.planting_year,
        row.area_ha,
        row.age_months,
        row.growth_period,
        row.measurement_rows,
        row.mean_volume_m3_tree,
        row.trees_per_ha,
        row.estimated_volume_m3_ha,
        row.mai_m3_ha_year,
        row.status,
      ]
        .map(escapeCsv)
        .join(",")
    );

    const csv = "\uFEFF" + [headers.map(escapeCsv).join(","), ...body].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8;" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "laporan-mai-kompartemen.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-6 sm:flex-row sm:items-center sm:justify-between lg:px-8">
          <div>
            <a href="/" className="text-sm font-semibold text-emerald-700 hover:text-emerald-900">
              ← Kembali ke dashboard
            </a>
            <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">
              Laporan Mean Annual Increment
            </h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => void loadReport()}
              className="rounded-lg border border-slate-300 bg-white px-4 py-2.5 text-sm font-semibold hover:bg-slate-100"
            >
              Muat ulang
            </button>
            <button
              onClick={exportCsv}
              disabled={!visibleDetails.length}
              className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Ekspor CSV
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl space-y-6 px-5 py-7 lg:px-8">
        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold">Filter laporan</h2>
              <p className="mt-1 text-sm text-slate-500">
                Filter diterapkan ke endpoint laporan MAI. Tahun awal dan akhir bersifat inklusif.
              </p>
            </div>
            <button
              onClick={resetFilters}
              className="mt-2 self-start text-sm font-semibold text-emerald-700 hover:text-emerald-900 sm:mt-0"
            >
              Reset filter
            </button>
          </div>

          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
            <label className="text-sm font-medium text-slate-700">
              Tahun tanam awal
              <select
                value={yearStart}
                onChange={(event) => setYearStart(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua</option>
                {[2000, 2005, 2010, 2015, 2020].map((year) => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              Tahun tanam akhir
              <select
                value={yearEnd}
                onChange={(event) => setYearEnd(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua</option>
                {[2000, 2005, 2010, 2015, 2020].map((year) => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              Regional
              <select
                value={regional}
                onChange={(event) => setRegional(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua regional</option>
                {regionalOptions.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              Kompartemen
              <select
                value={compartment}
                onChange={(event) => setCompartment(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua kompartemen</option>
                {compartmentOptions.map(([id, code]) => (
                  <option key={id} value={id}>{code}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              Spesies
              <select
                value={species}
                onChange={(event) => setSpecies(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua spesies</option>
                {speciesOptions.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              Klon
              <select
                value={clone}
                onChange={(event) => setClone(event.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              >
                <option value="">Semua klon</option>
                {cloneOptions.map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
        </section>

        {loading && (
          <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-slate-600">
            Memuat data laporan MAI...
          </div>
        )}

        {!loading && error && (
          <section className="rounded-xl border border-red-200 bg-red-50 p-5">
            <h2 className="font-semibold text-red-800">Laporan belum dapat dimuat</h2>
            <p className="mt-2 [break-words] text-sm text-red-700">{error}</p>
            <p className="mt-2 text-sm text-red-700">
              Pastikan backend FastAPI berjalan dan NEXT_PUBLIC_API_URL mengarah ke server API.
            </p>
            <button
              onClick={() => void loadReport()}
              className="mt-4 rounded-lg bg-red-700 px-4 py-2 text-sm font-semibold text-white"
            >
              Coba lagi
            </button>
          </section>
        )}

        {!loading && report && (
          <>
            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="MAI berbobot luas"
                value={number(report.summary.mai_weighted_m3_ha_year)}
                unit="m³/ha/tahun"
                description="Rata-rata berbobot berdasarkan luas kohort."
              />
              <MetricCard
                label="Kombinasi kohort-kompartemen"
                value={integer(report.summary.cohort_compartment_count)}
                unit="kombinasi"
                description="Setiap kompartemen dihitung pada setiap tahun tanam."
              />
              <MetricCard
                label="Akumulasi luas kohort"
                value={number(report.summary.cohort_area_ha, 0)}
                unit="ha"
                description="Bukan luas fisik unik jika beberapa tahun tanam dipilih."
              />
              <MetricCard
                label="Memenuhi ambang MAI"
                value={`${integer(report.summary.good_count)} / ${integer(
                  report.summary.cohort_compartment_count
                )}`}
                unit={`ambang ≥ ${number(report.summary.threshold_m3_ha_year, 0)} m³/ha/tahun`}
                description={`${integer(report.summary.below_threshold_count)} kombinasi di bawah ambang.`}
              />
            </section>

            <section className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <div className="mb-5">
                  <h2 className="text-lg font-semibold">MAI berbobot menurut tahun tanam</h2>
                  <p className="mt-1 text-sm text-slate-500">
                    Satuan m³/ha/tahun. Garis ambang menunjukkan nilai minimum 25.
                  </p>
                </div>
                {report.by_planting_year.length ? (
                  <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={report.by_planting_year}
                        margin={{ top: 10, right: 12, left: 4, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="planting_year" />
                        <YAxis />
                        <Tooltip
                          formatter={(value) => [
                            `${number(Number(value))} m³/ha/tahun`,
                            "MAI berbobot",
                          ]}
                        />
                        <Bar
                          dataKey="mai_weighted_m3_ha_year"
                          name="MAI berbobot"
                          fill="#047857"
                          radius={[5, 5, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="rounded-lg bg-slate-50 p-6 text-sm text-slate-500">
                    Tidak ada data tahunan untuk filter ini.
                  </p>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h2 className="text-lg font-semibold">Ringkasan per tahun tanam</h2>
                <p className="mt-1 text-sm text-slate-500">
                  Nilai minimum, MAI berbobot, dan maksimum.
                </p>
                <div className="mt-4 overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-slate-200 text-xs uppercase text-slate-500">
                      <tr>
                        <th className="py-3 pr-3">Tahun</th>
                        <th className="py-3 pr-3">MAI</th>
                        <th className="py-3 pr-3">Min–maks</th>
                        <th className="py-3">Baik</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.by_planting_year.map((row) => (
                        <tr key={row.planting_year} className="border-b border-slate-100 last:border-0">
                          <td className="py-3 pr-3 font-semibold">{row.planting_year}</td>
                          <td className="py-3 pr-3">{number(row.mai_weighted_m3_ha_year)}</td>
                          <td className="py-3 pr-3 text-slate-600">
                            {number(row.mai_min_m3_ha_year)}–{number(row.mai_max_m3_ha_year)}
                          </td>
                          <td className="py-3">
                            <span className="rounded-full bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-800">
                              {row.good_count}/{row.compartment_count}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Detail MAI per kompartemen</h2>
                  <p className="mt-1 text-sm text-slate-500">
                    Menampilkan {integer(visibleDetails.length)} dari {integer(report.details.length)} baris pada hasil API.
                  </p>
                </div>
                <label className="w-full text-sm font-medium text-slate-700 md:max-w-sm">
                  Cari detail
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Kode kompartemen, tahun, regional, spesies..."
                    className="mt-1.5 w-full rounded-lg border border-slate-300 px-3 py-2.5"
                  />
                </label>
              </div>

              <div className="mt-5 overflow-x-auto">
                <table className="min-w-1200px w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                    <tr>
                      <th className="px-3 py-3">Regional</th>
                      <th className="px-3 py-3">Kompartemen</th>
                      <th className="px-3 py-3">Tahun tanam</th>
                      <th className="px-3 py-3">Luas (ha)</th>
                      <th className="px-3 py-3">Umur</th>
                      <th className="px-3 py-3">Baris ukur</th>
                      <th className="px-3 py-3">Volume rata-rata (m³/pohon)</th>
                      <th className="px-3 py-3">Pohon/ha tercatat</th>
                      <th className="px-3 py-3">Estimasi volume (m³/ha)</th>
                      <th className="px-3 py-3">MAI (m³/ha/tahun)</th>
                      <th className="px-3 py-3">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleDetails.map((row, index) => (
                      <tr
                        key={`${row.compartment_id}-${row.planting_year}-${index}`}
                        className="border-b border-slate-100 hover:bg-slate-50"
                      >
                        <td className="px-3 py-3">{row.regional_id}</td>
                        <td className="px-3 py-3 font-semibold">{row.compartment_code}</td>
                        <td className="px-3 py-3">{row.planting_year}</td>
                        <td className="px-3 py-3">{number(row.area_ha)}</td>
                        <td className="px-3 py-3">
                          {row.age_months} bln <span className="text-xs text-slate-500">({row.growth_period})</span>
                        </td>
                        <td className="px-3 py-3">{integer(row.measurement_rows)}</td>
                        <td className="px-3 py-3">{number(row.mean_volume_m3_tree, 4)}</td>
                        <td className="px-3 py-3">{integer(row.trees_per_ha)}</td>
                        <td className="px-3 py-3">{number(row.estimated_volume_m3_ha)}</td>
                        <td className="px-3 py-3 font-semibold">{number(row.mai_m3_ha_year)}</td>
                        <td className="px-3 py-3">
                          <span
                            className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${
                              row.status === "Baik"
                                ? "bg-emerald-50 text-emerald-800"
                                : "bg-amber-50 text-amber-800"
                            }`}
                          >
                            {row.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {!visibleDetails.length && (
                      <tr>
                        <td colSpan={11} className="px-3 py-8 text-center text-slate-500">
                          Tidak ada detail yang cocok dengan pencarian.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="rounded-xl border border-amber-200 bg-amber-50 p-5">
              <h2 className="text-lg font-semibold text-amber-950">Catatan interpretasi</h2>
              <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-6 text-amber-950">
                <li>{report.methodology.density_note}</li>
                <li>{report.methodology.area_note}</li>
                <li>
                  MAI dihitung dari estimasi volume per hektare dibagi umur pengamatan dalam tahun.
                  Nilai ini tidak sama dengan MAI per pohon.
                </li>
                <li>
                  Data yang ditampilkan berasal dari pengukuran aktual pada umur pengamatan terakhir
                  yang tersedia untuk setiap kombinasi kompartemen dan tahun tanam; bukan prediksi PINNs.
                </li>
              </ul>
            </section>

            <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="text-lg font-semibold">Metodologi dan sumber data</h2>
              <p className="mt-1 text-sm text-slate-500">
                Sumber: {report.source}
              </p>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                <Formula label="Metrik sumber" value={report.methodology.source_metric} />
                <Formula label="Estimasi volume per hektare" value={report.methodology.estimated_volume_formula} />
                <Formula label="Rumus MAI" value={report.methodology.mai_formula} />
                <Formula label="Rumus rata-rata berbobot" value={report.methodology.weighted_formula} />
                <Formula label="Aturan ambang" value={report.methodology.threshold_rule} />
                <Formula label="Model prediksi digunakan" value={report.methodology.prediction_used ? "Ya" : "Tidak"} />
              </div>
            </section>
                    </>
        )}
        
        {/* <DatasetHistory /> */}
      </div>
    </main>
  );
}

function MetricCard({
  label,
  value,
  unit,
  description,
}: {
  label: string;
  value: string;
  unit: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="mt-3 text-3xl font-bold tracking-tight text-slate-900">{value}</p>
      <p className="mt-1 text-xs font-semibold text-emerald-700">{unit}</p>
      <p className="mt-3 text-xs leading-5 text-slate-500">{description}</p>
    </div>
  );
}

function Formula({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <p className="text-xs font-bold uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-2 [break-words] text-sm text-slate-800">{value}</p>
    </div>
  );
}
