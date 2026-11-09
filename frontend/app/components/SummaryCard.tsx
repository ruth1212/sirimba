type SummaryCardProps = {
  label: string;
  value: string;
  unit?: string;
};

export default function SummaryCard({
  label,
  value,
  unit,
}: SummaryCardProps) {
  return (
    <div className="rounded-2xl border border-[#A9D3B8] bg-white p-5 shadow-sm transition hover:shadow-md">
      <p className="text-xs font-medium uppercase tracking-wide text-[#4A6556]">
        {label}
      </p>

      <p className="mt-2 text-xl font-bold text-[#0F2A1D]">
        {value}
      </p>

      {unit && (
        <p className="mt-1 text-xs text-[#4A6556]">
          {unit}
        </p>
      )}
    </div>
  );
}
