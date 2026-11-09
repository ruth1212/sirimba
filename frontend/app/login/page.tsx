"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [message, setMessage] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (username.trim() === "ruth" && password === "ruth1212") {
      sessionStorage.setItem("sirimbo_logged_in", "true");
      document.cookie = "sirimbo_logged_in=true; path=/; SameSite=Lax";
      router.push("/dashboard");
    } else {
      setMessage("Username atau password salah. Silakan coba lagi.");
    }
  }

  return (
    <main className="min-h-screen bg-[#F4F7F2] text-[#183B2B]">
      <div className="grid min-h-screen lg:grid-cols-2">

        {/* BRAND PANEL */}
        <section className="relative hidden overflow-hidden bg-[#153D2B] px-12 py-10 text-white lg:flex lg:flex-col lg:justify-between xl:px-20">
          <div className="absolute -right-28 -top-28 h-96 w-96 rounded-full border border-white/10" />
          <div className="absolute -bottom-44 -left-24 h-[34rem] w-[34rem] rounded-full border border-white/10" />

          <div className="relative z-10 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#D9E9D8] text-[#153D2B]">
              <svg viewBox="0 0 48 48" className="h-8 w-8" fill="currentColor">
                <path d="M24 5 11 23h8l-5 10h20l-5-10h8L24 5Z" />
                <path d="M24 31v12" stroke="currentColor" strokeWidth="3" />
              </svg>
            </div>
            <div>
              <p className="text-xl font-extrabold tracking-[0.16em]">SIRIMBA</p>
              <p className="text-xs text-[#BBD3C1]">Sistem Informasi Rimba</p>
            </div>
          </div>

          <div className="relative z-10 max-w-xl pb-8">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-2 text-xs text-[#D2EADB]">
              <span className="h-2 w-2 rounded-full bg-[#9BC7A7]" />
              Forest Intelligence Platform
            </span>

            <h1 className="mt-6 font-[var(--font-outfit)] text-5xl font-semibold leading-tight xl:text-6xl">
              Understand the forest.
              <br />
              <span className="text-[#A9D1B4]">Grow the future.</span>
            </h1>

            <p className="mt-6 max-w-md text-base leading-7 text-[#C4D8CA]">
              In a forest canopy, tall trees naturally leave a little space between their crowns so everyone gets a fair share of sunlight.
            </p>
          </div>

          <p className="relative z-10 text-xs text-[#A9C4B1]">
            © 2026 SIRIMBA · Sistem Informasi Rimba
          </p>
        </section>

        {/* LOGIN FORM */}
        <section className="flex items-center justify-center px-5 py-10 sm:px-10 lg:px-12">
          <div className="w-full max-w-md">

            <div className="mb-10 flex items-center gap-3 lg:hidden">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#153D2B] text-white">
                <span className="text-xl font-bold">S</span>
              </div>
              <div>
                <p className="font-extrabold tracking-[0.14em]">SIRIMBA</p>
                <p className="text-xs text-[#718477]">Sistem Informasi Rimba</p>
              </div>
            </div>

            <div className="mb-8">
              <p className="text-sm font-semibold uppercase tracking-[0.18em] text-[#64836D]">
                Welcome ! 
              </p>
              <h2 className="mt-3 font-[var(--font-outfit)] text-3xl font-semibold sm:text-4xl">
                Masuk ke akunmu
              </h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-5 rounded-2xl border border-[#D7E2D7] bg-white p-6 shadow-xl shadow-[#183B2B]/10 sm:p-8">
              <div>
                <label htmlFor="username" className="mb-2 block text-sm font-semibold">
                  Username
                </label>
                <input
                  id="username"
                  type="text"
                  autoComplete="username"
                  required
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  placeholder="Masukkan username"
                  className="h-12 w-full rounded-xl border border-[#0b1a0b] bg-white px-4 text-sm outline-none transition placeholder:text-[#A5B2A7] focus:border-[#3D7953] focus:ring-4 focus:ring-[#3D7953]/10"
                />
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label htmlFor="password" className="text-sm font-semibold">
                    Password
                  </label>
                  {/* <button
                    type="button"
                    onClick={() =>
                      setMessage("Fitur reset password akan tersedia setelah integrasi autentikasi.")
                    }
                    className="text-xs font-bold text-[#39744D] hover:text-[#245236]"
                  >
                    Lupa password?
                  </button> */}
                </div>

                <div className="flex h-12 items-center rounded-xl border border-[#0b1a0b] bg-white px-4 focus-within:border-[#3D7953] focus-within:ring-4 focus-within:ring-[#3D7953]/10">
                  <input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    required
                    minLength={6}
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="Masukkan password"
                    className="w-full bg-transparent text-sm outline-none placeholder:text-[#A5B2A7]"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword((value) => !value)}
                    className="ml-2 text-xs font-semibold text-[#39744D]"
                  >
                    {showPassword ? "Sembunyikan" : "Lihat"}
                  </button>
                </div>
              </div>

              <label className="flex cursor-pointer items-center gap-2.5 text-sm text-[#637568]">
                <input
                  type="checkbox"
                  checked={remember}
                  onChange={(event) => setRemember(event.target.checked)}
                  className="h-4 w-4 accent-[#2F6844]"
                />
                Ingat saya
              </label>

              <button
                type="submit"
                className="flex h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#1E5035] px-4 text-sm font-semibold text-white shadow-lg shadow-[#1E5035]/15 transition hover:bg-[#163F29] focus:outline-none focus:ring-4 focus:ring-[#3D7953]/20"
              >
                Masuk ke Dashboard
                <span aria-hidden="true">→</span>
              </button>
              <button
                    type="button"
                    onClick={() =>
                      setMessage("Fitur reset password akan tersedia setelah integrasi autentikasi.")
                    }
                    className="text-xs font-bold item-center justify-center gap-2 text-[#39744D] hover:text-[#245236]"
                  >
                    Lupa password?
                </button>

              {message && (
                <div role="status" className="rounded-xl border border-[#D4E4D6] bg-[#EAF3EB] px-4 py-3 text-sm leading-6 text-[#315D3D]">
                  {message}
                </div>
              )}
            </form>

            <div className="mt-8 flex items-center gap-3">
              <div className="h-px flex-1 bg-[#E0E8E0]" />
              <span className="text-xs text-[#9AA89D]">SIRIMBA · Forest Intelligence</span>
              <div className="h-px flex-1 bg-[#E0E8E0]" />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}




