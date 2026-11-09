"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import TreeGrowthDashboard from "../components/TreeGrowthDashboard";

export default function DashboardPage() {
  const router = useRouter();
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    const loggedIn = sessionStorage.getItem("sirimbo_logged_in");

    if (loggedIn === "true") {
      setAuthenticated(true);
    } else {
      router.replace("/login");
    }
  }, [router]);

  if (!authenticated) {
    return null;
  }

  return <TreeGrowthDashboard initialView="comparison" />;
}
