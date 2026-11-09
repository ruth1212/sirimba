import { NextRequest, NextResponse } from "next/server";

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const isLoggedIn = request.cookies.get("sirimbo_logged_in")?.value === "true";

  if (pathname === "/login") {
    if (isLoggedIn) {
      return NextResponse.redirect(new URL("/dashboard", request.url));
    }

    return NextResponse.next();
  }

  if (!isLoggedIn) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/",
    "/dashboard/:path*",
    "/overview/:path*",
    "/current-situation/:path*",
    "/ai-prediction/:path*",
    "/ai-assistant/:path*",
    "/upload-dataset/:path*",
    "/dataset-history/:path*",
    "/mai-report/:path*",
  ],
};
