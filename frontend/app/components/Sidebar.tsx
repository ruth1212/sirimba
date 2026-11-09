"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

const navigation = [
  { label: "Overview", href: "/dashboard" },
  { label: "Current Situation", href: "/current-situation" },
  { label: "AI Prediction", href: "/ai-prediction" },
  { label: "AI Assistant", href: "/ai-assistant" },
  { label: "Upload Dataset", href: "/upload-dataset" },
  { label: "Dataset History", href: "/dataset-history" },
  { label: "Reports", href: "/mai-report" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  function handleLogout() {
    sessionStorage.removeItem("sirimbo_logged_in");
    document.cookie =
      "sirimbo_logged_in=; path=/; max-age=0; SameSite=Lax";

    router.replace("/login");
    router.refresh();
  }

  return (
    <aside className="w-full shrink-0 border-b border-slate-200 bg-white lg:sticky lg:top-0 lg:h-screen lg:self-start lg:overflow-y-auto lg:w-[232px] lg:border-b-0">
      <div className="flex min-h-full flex-col bg-gradient-to-b from-[#137a51] to-[#06301F] text-white">
        <div className="border-b border-white/10 px-6 py-7">
          <h1 className="font-brand text-3xl font-bold tracking-wide text-white">
            SIRIMBA 🌴
          </h1>
          <p className="mt-1 text-xs tracking-wide text-emerald-100">
            Tree Growth Dashboard
          </p>
          <div className="mt-5 h-1 w-12 rounded-full bg-[#2BBF73]" />
        </div>

        <nav
          aria-label="Main navigation"
          className="flex flex-1 gap-2 overflow-x-auto p-4 lg:flex-col"
        >
          {navigation.map((item) => {
            const active =
              pathname === item.href ||
              (item.href === "/dashboard" && pathname === "/");

            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`shrink-0 rounded-xl border-l-4 px-4 py-3 text-sm font-medium transition ${
                  active
                    ? "border-[#2BBF73] bg-[#0F7A4A] text-white shadow-sm"
                    : "border-transparent text-emerald-100 hover:bg-white/10 hover:text-white"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto border-t border-white/10 p-4">
          <p className="mb-3 px-2 text-xs text-emerald-400">
            Tree Growth Analytics
          </p>

          <button
            type="button"
            onClick={handleLogout}
            className="flex w-full items-center gap-3 rounded-xl border border-red-300/30 bg-red-500/10 px-4 py-3 text-sm font-semibold text-red-100 transition hover:bg-red-500/25 hover:text-white"
          >
            <span aria-hidden="true">↪</span>
            Keluar
          </button>
        </div>
      </div>
    </aside>
  );
}
