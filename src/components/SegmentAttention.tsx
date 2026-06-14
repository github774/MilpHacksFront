import { motion } from "framer-motion";
import type { SimulationResult } from "../api";
import { cn } from "../lib/utils";
import { fmtPct, getTranscriptSegments } from "../lib/simulation";

interface Props {
  result: SimulationResult;
  agentWeights?: number[];
  title?: string;
  compact?: boolean;
  scrollable?: boolean;
  selectedIndex?: number | null;
  highlightIndex?: number | null;
  onSelect?: (index: number) => void;
  onHighlight?: (index: number | null) => void;
}

export function SegmentAttention({
  result,
  agentWeights,
  title = "Where the model looked",
  compact = false,
  scrollable = false,
  selectedIndex = null,
  highlightIndex = null,
  onSelect,
  onHighlight,
}: Props) {
  const segments = getTranscriptSegments(result);
  const attn = result.analysis.advanced.segment_attention;
  const weights =
    agentWeights && agentWeights.length === segments.length
      ? agentWeights
      : attn?.mean_weights_all || [];

  if (!segments.length || !weights.length) {
    return (
      <div className="rounded-xl ui-inset px-4 py-3">
        <p className="text-[12px] text-white/45">Segment attention unavailable for this run.</p>
      </div>
    );
  }

  const maxW = Math.max(0.001, ...weights);
  const ranked = segments
    .map((text, i) => ({ text, weight: weights[i] ?? 0, i }))
    .sort((a, b) => b.weight - a.weight);
  const topIdx = ranked[0]?.i ?? 0;

  return (
    <div className="rounded-xl ui-inset overflow-hidden">
      <div className="px-4 py-3 ui-panel-header flex items-center justify-between gap-3">
        <div>
          <h3 className="text-[13px] font-semibold text-white">{title}</h3>
          {!compact && (
            <p className="text-[11px] text-white/45 mt-0.5">
              Click a segment to inspect attention weight. PathPool weights averaged across exposed
              personas{agentWeights ? " (this agent)" : ""}.
            </p>
          )}
        </div>
        <span className="text-[10px] text-white/35 tabular-nums shrink-0">
          {segments.length} segments
        </span>
      </div>

      <div
        className={cn(
          "px-4 py-3 space-y-1",
          compact && "max-h-[140px] overflow-y-auto",
          scrollable && !compact && "max-h-[280px] overflow-y-auto"
        )}
      >
        {segments.map((text, i) => {
          const w = weights[i] ?? 0;
          const pct = w / maxW;
          const isTop = i === topIdx;
          const isSelected = selectedIndex === i;
          const isHighlighted = highlightIndex === i;
          const active = isSelected || isHighlighted;

          return (
            <button
              key={i}
              type="button"
              onClick={() => onSelect?.(i)}
              onMouseEnter={() => onHighlight?.(i)}
              onMouseLeave={() => onHighlight?.(null)}
              className={cn(
                "w-full text-left rounded-lg px-2 py-2 -mx-2 transition-all duration-200",
                "hover:bg-white/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                isSelected && "bg-white/[0.06] ring-1 ring-white/25",
                isHighlighted && !isSelected && "bg-white/[0.03]"
              )}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] text-white/35 w-4 tabular-nums">{i + 1}</span>
                <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(4, pct * 100)}%` }}
                    transition={{ duration: 0.5, delay: i * 0.04 }}
                    className="h-full rounded-full"
                    style={{
                      background: active || isTop ? "#ffffff" : "rgba(255,255,255,0.35)",
                      opacity: active ? 1 : isTop ? 0.9 : 0.7,
                    }}
                  />
                </div>
                <span
                  className={cn(
                    "text-[10px] tabular-nums w-10 text-right",
                    active || isTop ? "text-white font-semibold" : "text-white/50"
                  )}
                >
                  {fmtPct(w, 0)}
                </span>
              </div>
              <p
                className={cn(
                  "text-[12px] leading-relaxed pl-6",
                  active || isTop ? "text-white/90" : "text-white/55"
                )}
              >
                {text}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
