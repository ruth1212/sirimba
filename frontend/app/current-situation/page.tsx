"use client";

import { useEffect, useMemo, useState } from "react";
import {
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

const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

type Compartment = {
  regional_id: string;
  compartment_id: number;
  compartment_code: string;
};

type Filters = {
  regional: { regional_id: string }[];
  compartment: Compartment[];
  species: string[];
  clone: string[];
};

type ActualPeriod = {
  growth_period: string;
  age_months_mean: number;
  dbh_cm: number;
  height_m: number;
  volume_m3_tree: number;
  actual_rows: number;
};

const numberFormat = (value: number, digits = 2) =>
  new Intl.NumberFormat("id-ID", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(value);

export default function CurrentSituationPage() {
  const [filters, setFilters] = useState<Filters>({
    regional: [],
    compartment: [],
    species: [],
    clone: [],
  });

  const [regional, setRegional] = useState("");
  const [compartment, setCompartment] = useState("");
  const [species, setSpecies] = useState("");
  const [clone, setClone] = useState("");

  const [periods, setPeriods] = useState<ActualPeriod[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [compartmentPeriods, setCompartmentPeriods] = useState<
    { compartment: Compartment; periods: ActualPeriod[] }[]
  >([]);
  const [loadingCompartments, setLoadingCompartments] = useState(false);
  const [compartmentError, setCompartmentError] = useState("");

  const availableCompartments = useMemo(
    () =>
      filters.compartment.filter(
        (item) => !regional || item.regional_id === regional
      ),
    [filters.compartment, regional]
  );

  const availableClones = useMemo(() => {
    if (!species) return filters.clone;

    const clonePrefix: Record<string, string> = {
      "Eucalyptus grandis": "EG-",
      "Eucalyptus pellita": "EP-",
      "Eucalyptus urophylla": "EU-",
    };

    const prefix = clonePrefix[species];

    if (!prefix) return [];

    return filters.clone.filter((item) => item.startsWith(prefix));
  }, [filters.clone, species]);

  useEffect(() => {
    async function loadFilters() {
      try {
        const response = await fetch(`${API_URL}/dashboard/filters`);

        if (!response.ok) {
          throw new Error("Gagal mengambil pilihan filter.");
        }

        const result = await response.json();
        setFilters(result.filters);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Terjadi kesalahan."
        );
      }
    }

    loadFilters();
  }, []);

  useEffect(() => {
    let active = true;

    async function loadActualData() {
      setLoading(true);
      setError("");

      try {
        const params = new URLSearchParams();

        if (regional) params.set("regional_id", regional);
        if (compartment) params.set("compartment_id", compartment);
        if (species) params.set("species", species);
        if (clone) params.set("clone", clone);

        const response = await fetch(
          `${API_URL}/dashboard/actual-trajectory?${params.toString()}`
        );

        if (!response.ok) {
          throw new Error("Gagal mengambil data aktual.");
        }

        const result = await response.json();

        if (!result.success) {
          throw new Error("Respons data aktual tidak berhasil.");
        }

        if (active) {
          setPeriods(result.periods || []);
        }
      } catch (err) {
        if (active) {
          setError(
            err instanceof Error ? err.message : "Terjadi kesalahan."
          );
          setPeriods([]);
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadActualData();

    return () => {
      active = false;
    };
  }, [regional, compartment, species, clone]);

  const [regionalPeriods, setRegionalPeriods] = useState<
    { regional_id: string; periods: ActualPeriod[] }[]
  >([]);
  const [loadingRegional, setLoadingRegional] = useState(false);
  const [regionalError, setRegionalError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadRegionalComparison() {
      if (regional || compartment) {
        setRegionalPeriods([]);
        setRegionalError("");
        setLoadingRegional(false);
        return;
      }

      setLoadingRegional(true);
      setRegionalError("");

      try {
        const regionalIds = ["R01", "R02", "R03"];

        const results = await Promise.all(
          regionalIds.map(async (regionalId) => {
            const params = new URLSearchParams({
              regional_id: regionalId,
            });

            if (species) params.set("species", species);
            if (clone) params.set("clone", clone);

            const response = await fetch(
              `${API_URL}/dashboard/actual-trajectory?${params.toString()}`
            );

            if (!response.ok) {
              throw new Error(`Gagal mengambil data ${regionalId}.`);
            }

            const result = await response.json();

            if (!result.success || !Array.isArray(result.periods)) {
              throw new Error(`Respons data ${regionalId} tidak valid.`);
            }

            return {
              regional_id: regionalId,
              periods: result.periods as ActualPeriod[],
            };
          })
        );

        if (active) {
          setRegionalPeriods(results);
        }
      } catch (err) {
        if (active) {
          setRegionalPeriods([]);
          setRegionalError(
            err instanceof Error
              ? err.message
              : "Gagal mengambil perbandingan regional."
          );
        }
      } finally {
        if (active) setLoadingRegional(false);
      }
    }

    loadRegionalComparison();

    return () => {
      active = false;
    };
  }, [regional, compartment, species, clone]);

  useEffect(() => {
    let active = true;

    async function loadCompartmentComparison() {
      if (!regional || compartment) {
        setCompartmentPeriods([]);
        setCompartmentError("");
        setLoadingCompartments(false);
        return;
      }

      if (availableCompartments.length === 0) {
        setCompartmentPeriods([]);
        setCompartmentError("");
        setLoadingCompartments(false);
        return;
      }

      setLoadingCompartments(true);
      setCompartmentError("");

      try {
        const results = await Promise.all(
          availableCompartments.map(async (item) => {
            const params = new URLSearchParams({
              regional_id: item.regional_id,
              compartment_id: String(item.compartment_id),
            });

            if (species) params.set("species", species);
            if (clone) params.set("clone", clone);

            const response = await fetch(
              `${API_URL}/dashboard/actual-trajectory?${params.toString()}`
            );

            if (!response.ok) {
              throw new Error(
                `Gagal mengambil data ${item.compartment_code}.`
              );
            }

            const result = await response.json();

            if (!result.success || !Array.isArray(result.periods)) {
              throw new Error(
                `Respons data ${item.compartment_code} tidak valid.`
              );
            }

            return {
              compartment: item,
              periods: result.periods as ActualPeriod[],
            };
          })
        );

        if (active) {
          setCompartmentPeriods(results);
        }
      } catch (err) {
        if (active) {
          setCompartmentPeriods([]);
          setCompartmentError(
            err instanceof Error
              ? err.message
              : "Gagal mengambil perbandingan compartment."
          );
        }
      } finally {
        if (active) setLoadingCompartments(false);
      }
    }

    loadCompartmentComparison();

    return () => {
      active = false;
    };
  }, [
    regional,
    compartment,
    species,
    clone,
    availableCompartments,
  ]);

  const latest = periods.length ? periods[periods.length - 1] : null;

  const cards = [
    {
      label: "Rata-rata DBH",
      value: latest ? numberFormat(latest.dbh_cm) : "-",
      unit: "cm",
    },
    {
      label: "Rata-rata Tinggi",
      value: latest ? numberFormat(latest.height_m) : "-",
      unit: "m",
    },
    {
      label: "Rata-rata Volume",
      value: latest ? numberFormat(latest.volume_m3_tree, 4) : "-",
      unit: "m³/pohon",
    },
    {
      label: "Jumlah Observasi",
      value: latest
        ? new Intl.NumberFormat("id-ID").format(latest.actual_rows)
        : "-",
      unit: latest ? latest.growth_period : "observasi",
    },
  ];

  const regionalComparison = regionalPeriods.map((item) => {
    const latestRegionalPeriod = item.periods[item.periods.length - 1];

    return {
      regional_id: item.regional_id,
      growth_period: latestRegionalPeriod?.growth_period ?? "-",
      dbh_cm: latestRegionalPeriod?.dbh_cm ?? 0,
      height_m: latestRegionalPeriod?.height_m ?? 0,
      volume_m3_tree: latestRegionalPeriod?.volume_m3_tree ?? 0,
      actual_rows: latestRegionalPeriod?.actual_rows ?? 0,
    };
  });

  const compartmentComparison = compartmentPeriods.map((item) => {
    const latestCompartmentPeriod =
      item.periods[item.periods.length - 1];

    return {
      compartment_code: item.compartment.compartment_code,
      growth_period: latestCompartmentPeriod?.growth_period ?? "-",
      dbh_cm: latestCompartmentPeriod?.dbh_cm ?? 0,
      height_m: latestCompartmentPeriod?.height_m ?? 0,
      volume_m3_tree: latestCompartmentPeriod?.volume_m3_tree ?? 0,
      actual_rows: latestCompartmentPeriod?.actual_rows ?? 0,
    };
  });

  return (
    <div className="space-y-6">
      <header>
        <p className="text-sm font-semibold tracking-wide text-emerald-700">
          SIRIMBA / MONITORING
        </p>
        <h1 className="mt-2 text-3xl font-bold text-slate-900">
          Current Situation
        </h1>
        <p className="mt-2 text-sm text-slate-500">
          Kondisi pertumbuhan berdasarkan observasi aktual dataset.
        </p>
      </header>

      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="mb-4">
          <h2 className="font-semibold text-slate-900">Filter Data</h2>
          <p className="mt-1 text-xs text-slate-500">
            Pilih cakupan data yang ingin dianalisis.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <label className="text-sm text-slate-600">
            Regional
            <select
              value={regional}
              onChange={(event) => {
                setRegional(event.target.value);
                setCompartment("");
              }}
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white p-3"
            >
              <option value="">Semua Regional</option>
              {filters.regional.map((item) => (
                <option key={item.regional_id} value={item.regional_id}>
                  {item.regional_id}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-slate-600">
            Compartment
            <select
              value={compartment}
              onChange={(event) => setCompartment(event.target.value)}
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white p-3"
            >
              <option value="">Semua Compartment</option>
              {availableCompartments.map((item) => (
                <option
                  key={`${item.regional_id}-${item.compartment_id}`}
                  value={item.compartment_id}
                >
                  {item.compartment_code}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-slate-600">
            Species
            <select
              value={species}
              onChange={(event) => {
                setSpecies(event.target.value);
                setClone("");
              }}
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white p-3"
            >
              <option value="">Semua Species</option>
              {filters.species.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-slate-600">
            Clone
            <select
              value={clone}
              onChange={(event) => setClone(event.target.value)}
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white p-3"
            >
              <option value="">Semua Clone</option>
              {availableClones.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      <section>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-900">
            Ringkasan Kondisi Aktual
          </h2>
          <span className="text-xs text-slate-500">
            {latest
              ? `Periode terakhir tersedia: ${latest.growth_period}`
              : "Belum ada periode tersedia"}
          </span>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {cards.map((card) => (
            <div
              key={card.label}
              className="rounded-xl border border-slate-200 bg-white p-5"
            >
              <p className="text-sm text-slate-500">{card.label}</p>
              <p className="mt-3 text-2xl font-bold text-slate-900">
                {loading ? "..." : card.value}
              </p>
              <p className="mt-1 text-xs text-slate-400">{card.unit}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-semibold text-slate-900">
            Pertumbuhan DBH
          </h2>
          <p className="mb-4 mt-1 text-xs text-slate-500">
            Rata-rata diameter batang berdasarkan periode umur.
          </p>
          <div className="h-72">
            {loading ? (
              <p className="pt-20 text-center text-sm text-slate-400">
                Memuat data...
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={periods}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="growth_period" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="dbh_cm"
                    name="DBH (cm)"
                    stroke="#059669"
                    strokeWidth={2.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-semibold text-slate-900">
            Pertumbuhan Tinggi
          </h2>
          <p className="mb-4 mt-1 text-xs text-slate-500">
            Rata-rata tinggi pohon berdasarkan periode umur.
          </p>
          <div className="h-72">
            {loading ? (
              <p className="pt-20 text-center text-sm text-slate-400">
                Memuat data...
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={periods}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="growth_period" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="height_m"
                    name="Tinggi (m)"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 xl:col-span-2">
          <h2 className="font-semibold text-slate-900">
            Volume Rata-rata per Pohon
          </h2>
          <p className="mb-4 mt-1 text-xs text-slate-500">
            Volume aktual berdasarkan periode umur pohon.
          </p>
          <div className="h-72">
            {loading ? (
              <p className="pt-20 text-center text-sm text-slate-400">
                Memuat data...
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={periods}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="growth_period" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="volume_m3_tree"
                    name="Volume (m³/pohon)"
                    stroke="#d97706"
                    strokeWidth={2.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </section>

      {regional && !compartment && (
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Compartment Performance Comparison
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Perbandingan rata-rata aktual antar-compartment pada {regional},
              berdasarkan periode terakhir yang tersedia.
            </p>
          </div>

          {loadingCompartments && (
            <div className="rounded-xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
              Memuat perbandingan compartment...
            </div>
          )}

          {compartmentError && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {compartmentError}
            </div>
          )}

          {!loadingCompartments &&
            !compartmentError &&
            compartmentComparison.length > 0 && (
              <>
                <div className="grid gap-4 xl:grid-cols-3">
                  <div className="rounded-xl border border-slate-200 bg-white p-5">
                    <h3 className="font-semibold text-slate-900">
                      Perbandingan DBH
                    </h3>
                    <p className="mb-4 mt-1 text-xs text-slate-500">
                      Rata-rata diameter batang (cm).
                    </p>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={compartmentComparison}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="compartment_code" />
                          <YAxis />
                          <Tooltip />
                          <Bar
                            dataKey="dbh_cm"
                            name="DBH (cm)"
                            fill="#059669"
                            radius={[5, 5, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-5">
                    <h3 className="font-semibold text-slate-900">
                      Perbandingan Tinggi
                    </h3>
                    <p className="mb-4 mt-1 text-xs text-slate-500">
                      Rata-rata tinggi pohon (m).
                    </p>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={compartmentComparison}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="compartment_code" />
                          <YAxis />
                          <Tooltip />
                          <Bar
                            dataKey="height_m"
                            name="Tinggi (m)"
                            fill="#2563eb"
                            radius={[5, 5, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-5">
                    <h3 className="font-semibold text-slate-900">
                      Perbandingan Volume
                    </h3>
                    <p className="mb-4 mt-1 text-xs text-slate-500">
                      Rata-rata volume per pohon (m³/pohon).
                    </p>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={compartmentComparison}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="compartment_code" />
                          <YAxis />
                          <Tooltip />
                          <Bar
                            dataKey="volume_m3_tree"
                            name="Volume (m³/pohon)"
                            fill="#d97706"
                            radius={[5, 5, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white p-5">
                  <h3 className="mb-3 font-semibold text-slate-900">
                    Cakupan Data Compartment
                  </h3>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                    {compartmentComparison.map((item) => (
                      <div
                        key={item.compartment_code}
                        className="rounded-lg border border-slate-200 p-3"
                      >
                        <p className="font-semibold text-slate-900">
                          {item.compartment_code}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          Periode: {item.growth_period}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          Observasi:{" "}
                          {new Intl.NumberFormat("id-ID").format(
                            item.actual_rows
                          )}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

          {!loadingCompartments &&
            !compartmentError &&
            compartmentComparison.length === 0 && (
              <div className="rounded-xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
                Belum tersedia data perbandingan compartment.
              </div>
            )}
        </section>
      )}

      {!regional && !compartment && (
        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Regional Performance Comparison
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Perbandingan rata-rata aktual pada periode terakhir yang tersedia
              untuk setiap regional.
            </p>
          </div>

          {loadingRegional && (
            <div className="rounded-xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
              Memuat perbandingan regional...
            </div>
          )}

          {regionalError && (
            <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {regionalError}
            </div>
          )}

          {!loadingRegional && !regionalError && regionalComparison.length > 0 && (
            <>
              <div className="grid gap-4 xl:grid-cols-3">
                <div className="rounded-xl border border-slate-200 bg-white p-5">
                  <h3 className="font-semibold text-slate-900">
                    Perbandingan DBH
                  </h3>
                  <p className="mb-4 mt-1 text-xs text-slate-500">
                    Rata-rata diameter batang (cm).
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={regionalComparison}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="regional_id" />
                        <YAxis />
                        <Tooltip />
                        <Bar
                          dataKey="dbh_cm"
                          name="DBH (cm)"
                          fill="#059669"
                          radius={[5, 5, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white p-5">
                  <h3 className="font-semibold text-slate-900">
                    Perbandingan Tinggi
                  </h3>
                  <p className="mb-4 mt-1 text-xs text-slate-500">
                    Rata-rata tinggi pohon (m).
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={regionalComparison}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="regional_id" />
                        <YAxis />
                        <Tooltip />
                        <Bar
                          dataKey="height_m"
                          name="Tinggi (m)"
                          fill="#2563eb"
                          radius={[5, 5, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white p-5">
                  <h3 className="font-semibold text-slate-900">
                    Perbandingan Volume
                  </h3>
                  <p className="mb-4 mt-1 text-xs text-slate-500">
                    Rata-rata volume per pohon (m³/pohon).
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={regionalComparison}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="regional_id" />
                        <YAxis />
                        <Tooltip />
                        <Bar
                          dataKey="volume_m3_tree"
                          name="Volume (m³/pohon)"
                          fill="#d97706"
                          radius={[5, 5, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <p className="text-xs text-slate-400">
                Periode data: {regionalComparison.map((item) =>
                  `${item.regional_id}: ${item.growth_period}`
                ).join(" | ")}
              </p>
            </>
          )}

          {!loadingRegional && !regionalError && regionalComparison.length === 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-6 text-center text-sm text-slate-500">
              Belum tersedia data perbandingan regional.
            </div>
          )}
        </section>
      )}

      <p className="text-xs text-slate-400">
        Sumber: dataset aktual melalui dashboard/actual-trajectory.
        Nilai merupakan rata-rata observasi per periode umur, bukan
        pengukuran real-time setiap pohon.
      </p>
    </div>
  );
}







