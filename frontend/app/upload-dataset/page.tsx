"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type UploadResult = {
  status: string;
  message: string;
  upload_id?: string;
  original_filename?: string;
  rows?: number;
  columns?: number;
  missing_columns?: string[];
  preview?: Record<string, unknown>[];
  dataset_activated?: boolean;
  dataset_path?: string;
};

export default function UploadDatasetPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");

  async function handleUpload() {
    if (!file) {
      setError("Pilih file dataset terlebih dahulu.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/dataset/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail);

        throw new Error(detail || "Upload dataset gagal.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Terjadi kesalahan saat upload dataset."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="text-sm font-semibold uppercase tracking-widest text-emerald-700">
          SIRIMBA Data Management
        </p>

        <h1 className="mt-2 text-3xl font-bold text-slate-900">
          Upload Dataset
        </h1>

        <p className="mt-2 text-slate-500">
          Upload dataset pertumbuhan pohon untuk validasi sebelum digunakan
          oleh sistem.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5">
          <h2 className="text-lg font-semibold text-slate-900">
            Dataset Import
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Format yang didukung: CSV, XLSX, XLS. Maksimal 100 MB.
          </p>
        </div>

        <label
          htmlFor="dataset-file"
          className="flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-emerald-300 bg-emerald-50/50 p-6 text-center transition hover:bg-emerald-50"
        >
          <span className="text-3xl text-emerald-700">↑</span>

          <span className="mt-3 font-semibold text-slate-800">
            Pilih dataset dari komputer
          </span>

          <span className="mt-1 text-sm text-slate-500">
            Klik area ini untuk memilih file
          </span>

          <input
            id="dataset-file"
            type="file"
            accept=".csv,.xlsx,.xls"
            className="hidden"
            onChange={(event) => {
              setFile(event.target.files?.[0] ?? null);
              setResult(null);
              setError("");
            }}
          />
        </label>

        {file && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg bg-slate-50 p-4">
            <div>
              <p className="font-medium text-slate-800">{file.name}</p>
              <p className="text-sm text-slate-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>

            <button
              type="button"
              onClick={() => {
                setFile(null);
                setResult(null);
                setError("");
              }}
              className="text-sm font-medium text-red-600 hover:text-red-700"
            >
              Hapus pilihan
            </button>
          </div>
        )}

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={handleUpload}
            disabled={!file || loading}
            className="rounded-lg bg-emerald-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Mengupload dan memvalidasi..." : "Upload & Validasi"}
          </button>

          <p className="text-xs text-slate-500">
            Dataset lama tidak akan diubah pada tahap ini.
          </p>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-5">
          <h3 className="font-semibold text-red-800">Upload gagal</h3>
          <p className="mt-2 [break-words] text-sm text-red-700">{error}</p>
        </div>
      )}

      {result && (
        <div className="space-y-5 rounded-2xl border border-emerald-200 bg-white p-6 shadow-sm">
          <div>
            <span className="inline-flex rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800">
              Validasi berhasil
            </span>

            <h2 className="mt-3 text-xl font-bold text-slate-900">
              Dataset berhasil diupload
            </h2>

            <p className="mt-1 text-sm text-slate-500">{result.message}</p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Metric label="Jumlah Baris" value={result.rows?.toLocaleString("id-ID") ?? "-"} />
            <Metric label="Jumlah Kolom" value={result.columns ?? "-"} />
            <Metric label="Status" value={result.status ?? "-"} />
            <Metric
              label="Dataset Aktif"
              value={result.dataset_activated ? "Ya" : "Belum"}
            />
          </div>

          {result.upload_id && (
            <div className="rounded-lg bg-slate-50 p-4">
              <p className="text-xs font-medium text-slate-500">Upload ID</p>
              <p className="mt-1 break-all font-mono text-sm text-slate-800">
                {result.upload_id}
              </p>
            </div>
          )}

          {result.preview && result.preview.length > 0 && (
            <div>
              <h3 className="mb-3 font-semibold text-slate-800">
                Preview Dataset (5 baris pertama)
              </h3>

              <div className="overflow-x-auto rounded-lg border border-slate-200">
                <table className="min-w-full text-left text-xs">
                  <thead className="bg-slate-100 text-slate-600">
                    <tr>
                      {Object.keys(result.preview[0]).map((column) => (
                        <th key={column} className="whitespace-nowrap px-3 py-3 font-semibold">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody>
                    {result.preview.map((row, index) => (
                      <tr key={index} className="border-t border-slate-100">
                        {Object.keys(result.preview![0]).map((column) => (
                          <td key={column} className="whitespace-nowrap px-3 py-3 text-slate-700">
                            {String(row[column] ?? "-")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <p className="text-sm text-amber-700">
            Dataset masih berada di staging dan belum digunakan oleh dashboard,
            AI Assistant, Random Forest, atau PINNs.
          </p>
        </div>
      )}
    </div>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-lg bg-slate-50 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-1 text-xl font-bold text-slate-900">{value}</p>
    </div>
  );
}
