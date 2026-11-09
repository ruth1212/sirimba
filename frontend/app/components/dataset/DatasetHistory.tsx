"use client";

import { useCallback, useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

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

const integer = (value: number) =>
  value.toLocaleString("id-ID");

export default function DatasetHistory() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadRegistry = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/dataset/registry`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`API ${response.status}`);
      }

      const data = (await response.json()) as RegistryResponse;

      if (!data.success || !Array.isArray(data.datasets)) {
        throw new Error("Format Registry tidak valid.");
      }

      setDatasets(data.datasets);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Gagal memuat Dataset History."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRegistry();
  }, [loadRegistry]);

  const stagedCount = datasets.filter(
    (dataset) => !dataset.dataset_activated
  ).length;

  const activeCount = datasets.filter(
    (dataset) => dataset.dataset_activated
  ).length;

  return (
    <section className="mt-8 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-900">
            Dataset History
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Riwayat dataset yang pernah diunggah dan divalidasi.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void loadRegistry()}
          disabled={loading}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
        >
          {loading ? "Memuat..." : "Muat ulang"}
        </button>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg bg-slate-50 p-4">
          <p className="text-sm text-slate-500">Total dataset</p>
          <p className="mt-1 text-2xl font-bold">
            {integer(datasets.length)}
          </p>
        </div>

        <div className="rounded-lg bg-amber-50 p-4">
          <p className="text-sm text-amber-800">Staged</p>
          <p className="mt-1 text-2xl font-bold text-amber-900">
            {integer(stagedCount)}
          </p>
        </div>

        <div className="rounded-lg bg-emerald-50 p-4">
          <p className="text-sm text-emerald-800">Aktif</p>
          <p className="mt-1 text-2xl font-bold text-emerald-900">
            {integer(activeCount)}
          </p>
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg bg-red-50 p-4 text-sm text-red-700">
          Gagal memuat Dataset History: {error}
        </div>
      )}

      {!error && loading && (
        <p className="mt-5 text-sm text-slate-500">
          Memuat riwayat dataset...
        </p>
      )}

      {!loading && !error && datasets.length === 0 && (
        <p className="mt-5 text-sm text-slate-500">
          Belum ada dataset yang terdaftar.
        </p>
      )}

      {!loading && !error && datasets.length > 0 && (
        <div className="mt-5 overflow-x-auto">
          <table className="min-w-850px w-full text-left text-sm">
            <thead className="border-b border-slate-200 bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-3">Nama dataset</th>
                <th className="px-3 py-3">Tanggal upload</th>
                <th className="px-3 py-3">Ukuran data</th>
                <th className="px-3 py-3">Status</th>
                <th className="px-3 py-3">Aksi</th>
              </tr>
            </thead>

            <tbody>
              {datasets.map((dataset) => (
                <tr
                  key={dataset.upload_id}
                  className="border-b border-slate-100"
                >
                  <td className="px-3 py-3">
                    <p className="font-medium text-slate-900">
                      {dataset.original_filename}
                    </p>
                    <details className="mt-1">
                      <summary className="cursor-pointer text-xs text-slate-500">
                        Lihat upload ID
                      </summary>
                      <p className="mt-1 break-all text-xs text-slate-600">
                        {dataset.upload_id}
                      </p>
                    </details>
                  </td>

                  <td className="px-3 py-3 text-slate-600">
                    {dataset.uploaded_at?.replace("T", " ") ?? "-"}
                  </td>

                  <td className="px-3 py-3 text-slate-600">
                    {integer(dataset.rows)} baris
                    <br />
                    {integer(dataset.columns)} kolom
                  </td>

                  <td className="px-3 py-3">
                    <span
                      className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                        dataset.dataset_activated
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-amber-100 text-amber-800"
                      }`}
                    >
                      {dataset.dataset_activated
                        ? "Aktif"
                        : "Staged"}
                    </span>
                  </td>

                  <td className="px-3 py-3">
                    <a
                      href={`${API_URL}/dataset/${encodeURIComponent(
                        dataset.upload_id
                      )}/download`}
                      className="font-medium text-blue-700 hover:text-blue-900"
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

      <p className="mt-4 text-xs leading-5 text-slate-500">
        Dataset staged belum digunakan sebagai dataset aktif dan tidak
        mengubah model RF maupun PINNs.
      </p>
    </section>
  );
}