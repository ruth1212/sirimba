"use client";

import { useCallback, useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type DatasetRecord = {
  upload_id: string;
  original_filename: string;
  stored_filename: string;
  rows: number;
  columns: number;
  status: string;
  dataset_activated: boolean;
  uploaded_at: string;
};

type RegistryResponse = {
  success: boolean;
  total: number;
  datasets: DatasetRecord[];
};

function formatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString("id-ID", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export default function DatasetHistoryPage() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const loadHistory = useCallback(async () => {
    setError("");
    setRefreshing(true);

    try {
      const response = await fetch(`${API_URL}/dataset/registry`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Gagal mengambil Registry (${response.status}).`);
      }

      const result: RegistryResponse = await response.json();

      if (!result.success || !Array.isArray(result.datasets)) {
        throw new Error("Format respons Registry tidak sesuai.");
      }

      setDatasets(result.datasets);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan saat mengambil riwayat dataset."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  return (
    <main className="min-h-screen bg-slate-50 p-6 lg:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <section>
          <p className="text-sm font-semibold uppercase tracking-wider text-emerald-700">
            SIRIMBA Data Management
          </p>

          <div className="mt-2 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <h1 className="text-3xl font-bold text-slate-900">
                Dataset History
              </h1>
              <p className="mt-2 text-sm text-slate-500">
                Riwayat dataset yang telah diunggah dan tersimpan di Registry.
              </p>
            </div>

            <button
              type="button"
              onClick={() => void loadHistory()}
              disabled={refreshing}
              className="rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {refreshing ? "Memuat..." : "Refresh"}
            </button>
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-sm text-slate-500">Total Dataset</p>
            <p className="mt-2 text-3xl font-bold text-slate-900">
              {loading ? "..." : datasets.length}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-sm text-slate-500">Dataset Staged</p>
            <p className="mt-2 text-3xl font-bold text-amber-600">
              {loading
                ? "..."
                : datasets.filter((item) => !item.dataset_activated).length}
            </p>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-sm text-slate-500">Dataset Aktif</p>
            <p className="mt-2 text-3xl font-bold text-emerald-700">
              {loading
                ? "..."
                : datasets.filter((item) => item.dataset_activated).length}
            </p>
          </div>
        </section>

        <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-200 px-6 py-4">
            <h2 className="font-semibold text-slate-900">
              Daftar Dataset
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Informasi dataset berdasarkan Registry backend.
            </p>
          </div>

          {loading && (
            <div className="p-8 text-center text-slate-500">
              Memuat riwayat dataset...
            </div>
          )}

          {!loading && error && (
            <div className="m-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          {!loading && !error && datasets.length === 0 && (
            <div className="p-8 text-center text-slate-500">
              Belum ada dataset dalam Registry.
            </div>
          )}

          {!loading && !error && datasets.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-900px text-left text-sm">
                <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-6 py-4">Nama Dataset</th>
                    <th className="px-6 py-4">Tanggal Upload</th>
                    <th className="px-6 py-4">Ukuran Data</th>
                    <th className="px-6 py-4">Status</th>
                    <th className="px-6 py-4">Upload ID</th>
                    <th className="px-6 py-4">Aksi</th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-slate-100">
                  {datasets.map((dataset) => (
                    <tr
                      key={dataset.upload_id}
                      className="align-top hover:bg-slate-50"
                    >
                      <td className="px-6 py-4">
                        <p className="font-semibold text-slate-900">
                          {dataset.original_filename}
                        </p>
                        <p className="mt-1 text-xs text-slate-500">
                          {dataset.dataset_activated
                            ? "Dataset aktif"
                            : "Belum digunakan sebagai dataset aktif"}
                        </p>
                      </td>

                      <td className="whitespace-nowrap px-6 py-4 text-slate-600">
                        {formatDate(dataset.uploaded_at)}
                      </td>

                      <td className="whitespace-nowrap px-6 py-4 text-slate-600">
                        <p>{dataset.rows.toLocaleString("id-ID")} baris</p>
                        <p className="mt-1 text-xs text-slate-400">
                          {dataset.columns} kolom
                        </p>
                      </td>

                      <td className="px-6 py-4">
                        <span className="inline-flex rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-800">
                          {dataset.status}
                        </span>
                      </td>

                      <td className="px-6 py-4">
                        <details className="max-w-xs">
                          <summary className="cursor-pointer text-emerald-700">
                            Lihat Upload ID
                          </summary>
                          <p className="mt-2 break-all font-mono text-xs text-slate-500">
                            {dataset.upload_id}
                          </p>
                        </details>
                      </td>

                      <td className="px-6 py-4">
                        <a
                          href={`${API_URL}/dataset/${encodeURIComponent(dataset.upload_id)}/download`}
                          className="inline-flex whitespace-nowrap rounded-lg border border-emerald-700 px-3 py-2 text-xs font-semibold text-emerald-700 transition hover:bg-emerald-50"
                        >
                          Download CSV
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <p className="text-xs text-slate-500">
          Dataset pada halaman ini masih berada di staging. Halaman ini tidak
          mengubah dataset aktif maupun model Random Forest dan PINNs.
        </p>
      </div>
    </main>
  );
}
