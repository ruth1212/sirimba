"use client";
import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
const API_URL = (
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");
type ChartPoint = {
  category: string;
  value: number;
};
type ChartPayload = {
  type: "bar";
  title: string;
  x_key: string;
  y_key: string;
  x_label: string;
  y_label: string;
  period: string | null;
  data: ChartPoint[];
};
type ChatResponse = {
  success: boolean;
  answer: string;
  mode?: string;
  data?: unknown;
  chart_offer?: {
    available: boolean;
    message: string;
    title: string;
    chart_type: string;
  } | null;
  chart?: ChartPayload | null;
};
type Message = {
  role: "assistant" | "user";
  content: string;
  chart?: ChartPayload | null;
  chartOffer?: boolean;
  chartVisible?: boolean;
  reportQuestion?: string;
  isMaiReport?: boolean;
};
const SUGGESTIONS = [
  "Berapa rata-rata tinggi pohon?",
  "Berapa rata-rata diameter pohon?",
  "Berapa rata-rata volume pohon?",
];
export default function ChatAssistant() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Halo Ruth, how can i help you ?",
    },
  ]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [downloadingPptx, setDownloadingPptx] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  // Tutup mode layar penuh dengan tombol Esc dan cegah halaman belakang ikut bergulir.
  useEffect(() => {
    if (!isFullscreen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setIsFullscreen(false);
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isFullscreen]);

  
async function sendQuestion(
  event?: FormEvent<HTMLFormElement>,
  suggestedQuestion?: string
) {
  event?.preventDefault();

  const text = (suggestedQuestion ?? question).trim();

  if (text.length < 3) {
    setError("Masukkan pertanyaan minimal 3 karakter.");
    return;
  }

  if (text.length > 1000) {
    setError("Pertanyaan maksimal 1000 karakter.");
    return;
  }

  if (loading) return;
  setError("");
  setQuestion("");
  setMessages((previous) => [
    ...previous,
    { role: "user", content: text },
  ]);

  setLoading(true);
  try {
    const response = await fetch(`${API_URL}/ai/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: text,
        history: messages
          .slice(-12)
          .map(({ role, content }) => ({ role, content })),
      }),
    });
    const result = (await response.json()) as ChatResponse;
    if (!response.ok) {
      throw new Error(
        result.answer || `Permintaan gagal (${response.status}).`
      );
    }
    if (!result.answer) {
      throw new Error("Backend belum memberikan jawaban.");
    }





    

    const isMaiReport =
      /\bMAI\b|mean annual increment|increment tahunan rata-rata/i.test(text);

    const isEmptyMaiReport =
      isMaiReport &&
      /tidak ada data yang sesuai dengan filter/i.test(result.answer);

    // Tambahkan jawaban assistant tepat satu kali.
    setMessages((previous) => [
  ...previous,
  {
    role: "assistant",
    content: result.answer,
    chart: result.chart ?? null,
    chartOffer: result.chart_offer?.available === true,
    chartVisible: false,
    reportQuestion:
    result.chart || (isMaiReport && !isEmptyMaiReport)
      ? text
      : undefined,
    isMaiReport,
  },
]);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Tidak dapat menghubungi backend."
    );
  } finally {
    setLoading(false);
  }
}

async function downloadPdfReport(question: string) {
  if (!question || downloadingPdf) return;

  setError("");
  setDownloadingPdf(true);

  try {
    const response = await fetch(`${API_URL}/ai/report/pdf`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let message = `Gagal membuat PDF (${response.status}).`;

      try {
        const result = await response.json();
        message = result.detail || result.answer || message;
      } catch {
        // Gunakan pesan default jika respons bukan JSON.
      }

      throw new Error(message);
    }

    const blob = await response.blob();

    if (blob.size === 0) {
      throw new Error("File PDF yang diterima kosong.");
    }

    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = objectUrl;
    link.download = "tree_growth_report.pdf";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Tidak dapat mengunduh laporan PDF."
    );
  } finally {
    setDownloadingPdf(false);
  }
}

async function downloadMaiPdfReport(question: string) {
  if (!question || downloadingPdf) return;

  setError("");
  setDownloadingPdf(true);

  try {
    const response = await fetch(`${API_URL}/ai/report/mai/pdf`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let message = `Gagal membuat PDF MAI (${response.status}).`;
      try {
        const result = await response.json();
        message = result.detail || result.answer || message;
      } catch {
        // Gunakan pesan default jika respons bukan JSON.
      }
      throw new Error(message);
    }

    const blob = await response.blob();
    if (blob.size === 0 || blob.type !== "application/pdf") {
      throw new Error("Respons bukan berkas PDF MAI yang valid.");
    }

    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = "laporan_mai.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Tidak dapat mengunduh laporan PDF MAI."
    );
  } finally {
    setDownloadingPdf(false);
  }
}

async function downloadPptxReport(question: string) {
  if (!question || downloadingPptx) return;

  setError("");
  setDownloadingPptx(true);

  try {
    const response = await fetch(`${API_URL}/ai/report/pptx`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      let message = `Gagal membuat PowerPoint (${response.status}).`;

      try {
        const result = await response.json();
        message = result.detail || result.answer || message;
      } catch {
        // Gunakan pesan default jika respons bukan JSON.
      }

      throw new Error(message);
    }

    const blob = await response.blob();

    if (blob.size === 0) {
      throw new Error("File PowerPoint yang diterima kosong.");
    }

    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = objectUrl;
    link.download = "tree_growth_report.pptx";

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(objectUrl);
  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Tidak dapat mengunduh laporan PowerPoint."
    );
  } finally {
    setDownloadingPptx(false);
  }
}
  return (
    <section
      id="ai-chat"
      aria-label="AI Growth Assistant"
      className={
        isFullscreen
          ? "fixed inset-0 z-[100] flex h-[100dvh] w-screen flex-col overflow-hidden rounded-none border-0 bg-white shadow-none"
          : "flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
      }
    >
      <header className="relative isolate overflow-hidden bg-gradient-to-r from-emerald-950 via-emerald-800 to-teal-700 px-4 py-4 text-white sm:px-6 sm:py-5">
        <div aria-hidden="true" className="pointer-events-none absolute -right-8 -top-16 h-44 w-44 rounded-full border border-white/10 bg-white/[0.04]" />
        <div aria-hidden="true" className="pointer-events-none absolute right-24 -top-20 h-36 w-36 rounded-full border border-white/10" />
        <div className="relative flex items-start gap-3 sm:items-center sm:gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-white/20 bg-white/10 text-2xl shadow-inner shadow-white/10 sm:h-14 sm:w-14">
            🌿
          </div>

          <div className="min-w-0 flex-1">
            <p className="mb-1 text-[10px] font-bold uppercase tracking-[0.2em] text-emerald-200 sm:text-xs">
              Halo, Ruth! 👋
            </p>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
              <h2 className="text-lg font-bold tracking-tight sm:text-2xl">
                Sirimba Assistant
              </h2>
              <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200/20 bg-emerald-300/10 px-2 py-1 text-[10px] font-medium text-emerald-100 sm:text-xs">
                <span className="h-1.5 w-1.5 rounded-full bg-lime-300 shadow-[0_0_8px_rgba(190,242,100,0.8)]" />
                Python Local
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setIsFullscreen((current) => !current)}
            aria-label={isFullscreen ? "Kembalikan ukuran AI Assistant" : "Perbesar AI Assistant ke layar penuh"}
            aria-pressed={isFullscreen}
            title={isFullscreen ? "Kembalikan ukuran (Esc)" : "Layar penuh"}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-white/25 bg-white/10 text-lg shadow-sm transition hover:border-white/40 hover:bg-white/20 focus:outline-none focus:ring-2 focus:ring-white/70 sm:h-11 sm:w-11"
          >
            {isFullscreen ? "⤢" : "⛶"}
          </button>
        </div>
      </header>
      <div
        className={`min-h-0 flex-1 space-y-5 overflow-y-auto bg-slate-50/70 p-4 sm:p-6 ${
          isFullscreen ? "max-h-none" : "max-h-[420px] min-h-[240px] flex-none"
        }`}
        aria-live="polite"
      >
        {messages.map((message, index) => (
          <div
            key={`${index}-${message.role}`}
            className={`flex w-full items-end gap-2 ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {message.role === "assistant" && (
              <div
                aria-hidden="true"
                className="mb-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-semibold text-emerald-800"
              >
                ✦
              </div>
            )}
            <div
              className={`min-w-0 max-w-[88%] whitespace-pre-wrap break-words rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm sm:max-w-[78%] ${
                message.role === "user"
                  ? "rounded-br-md bg-emerald-700 text-white"
                  : "rounded-bl-md border border-slate-200 bg-white text-slate-800"
              }`}
            >
              {message.content}
              
{message.role === "assistant" && message.chartOffer && message.chart && (
  <div className="mt-3 w-full max-w-full space-y-3">
    {!message.chartVisible ? (
      <div className="space-y-2">
        <p className="text-xs text-slate-600">
          Apakah kamu ingin melihat grafik perbandingannya?
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() =>
              setMessages((previous) =>
                previous.map((item, i) =>
                  i === index
                    ? { ...item, chartVisible: true }
                    : item
                )
              )
            }
            className="rounded-full bg-emerald-700 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-800"
          >
            Ya, tampilkan grafik
          </button>
          <button
            type="button"
            onClick={() =>
              setMessages((previous) =>
                previous.map((item, i) =>
                  i === index
                    ? {
                        ...item,
                        chartOffer: false,
                        chartVisible: false,
                      }
                    : item
                )
              )
            }
            className="rounded-full border border-slate-300 px-3 py-2 text-xs text-slate-600 hover:bg-slate-50"
          >
            Tidak, terima kasih
          </button>
        </div>
      </div>
    ) : (
      <div className="w-full min-w-0 rounded-xl border border-slate-200 bg-white p-3">
        <h3 className="mb-1 text-sm font-semibold text-slate-800">
          {message.chart.title}
        </h3>

        <p className="mb-3 text-xs text-slate-500">
          {message.chart.y_label}
          {message.chart.period
            ? ` · Periode ${message.chart.period}`
            : ""}
        </p>
        <div className="h-[260px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={message.chart.data}
              margin={{ top: 8, right: 8, left: 0, bottom: 28 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="category"
                angle={-20}
                textAnchor="end"
                interval={0}
                height={55}
                tick={{ fontSize: 10 }}
              />
              <YAxis
                tick={{ fontSize: 10 }}
                width={55}
              />
              <Tooltip
                formatter={(value) => [
                  Number(value).toFixed(4),
                  message.chart!.y_label,
                ]}
              />
              <Bar
                dataKey="value"
                fill="#047857"
                radius={[5, 5, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    )}
  </div>
)}

{message.role === "assistant" &&
  message.reportQuestion && (
    <div className="mt-4 flex flex-wrap justify-end gap-2 border-t border-slate-100 pt-3">
      <button
        type="button"
        onClick={() =>
          void (message.isMaiReport
            ? downloadMaiPdfReport(message.reportQuestion!)
            : downloadPdfReport(message.reportQuestion!))
        }
        disabled={downloadingPdf || downloadingPptx || loading}
        className="inline-flex items-center gap-2 rounded-xl bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {downloadingPdf
          ? "Membuat PDF..."
          : message.isMaiReport
            ? "Download Laporan MAI (PDF)"
            : "Download PDF"}
      </button>

      {!message.isMaiReport && (
        <button
          type="button"
          onClick={() =>
            void downloadPptxReport(message.reportQuestion!)
          }
          disabled={downloadingPptx || downloadingPdf || loading}
          className="inline-flex items-center gap-2 rounded-xl bg-slate-800 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {downloadingPptx
            ? "Membuat PowerPoint..."
            : "Download PowerPoint"}
        </button>
      )}
    </div>
  )}
            </div>
            
          </div>
        ))}
        {loading && (
          <div className="flex w-full items-end gap-2">
            <div
              aria-hidden="true"
              className="mb-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-100 text-sm font-semibold text-emerald-800"
            >
              ✦
            </div>
            <div className="rounded-2xl rounded-bl-md border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500 shadow-sm">
              <span className="mr-2 inline-flex h-2 w-2 animate-pulse rounded-full bg-emerald-600" />
              Sedang menganalisis data...
            </div>
          </div>
        )}
        <div ref={messagesEndRef} aria-hidden="true" />
      </div>
      <div className="shrink-0 border-t border-slate-100 bg-white px-4 py-4 sm:px-5">
        <p className="mb-3 text-xs font-medium text-slate-500">
          Coba tanyakan:
        </p>
        <div className="mb-4 flex flex-wrap gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              disabled={loading}
              onClick={() => void sendQuestion(undefined, suggestion)}
              className="rounded-full border border-emerald-200 px-3 py-2 text-xs text-emerald-800 transition hover:bg-emerald-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {suggestion}
            </button>
          ))}
        </div>
        <form
          onSubmit={(event) => void sendQuestion(event)}
          className="flex flex-col gap-3 sm:flex-row"
        >
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Tanyakan tentang pertumbuhan pohon..."
            maxLength={1000}
            disabled={loading}
            aria-label="Pertanyaan untuk AI Growth Assistant"
            className="min-w-0 flex-1 rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition focus:border-emerald-600 focus:ring-2 focus:ring-emerald-100 disabled:bg-slate-50"
          />
          <button
            type="submit"
            disabled={loading || question.trim().length < 3}
            className="rounded-xl bg-emerald-700 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Memproses..." : "Kirim"}
          </button>
        </form>


        {error && (
          <p
            role="alert"
            className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700"
          >
            {error}
          </p>
        )}
        <p className="mt-3 text-xs text-slate-400">
          Asisten ini bukan LLM eksternal. Jawaban berdasarkan kemampuan query Python lokal dan data
          yang tersedia. 
        </p>
      </div>
    </section>
  );
}
