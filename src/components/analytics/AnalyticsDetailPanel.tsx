import { motion, AnimatePresence } from "framer-motion";
import { Info, X } from "lucide-react";
import type { EmotionKey, ReactionKey, SimulationResult } from "../../api";
import {
  ACTION_HELP,
  AFFINITY_HELP,
  EMOTION_HELP,
  KPI_HELP,
} from "../../lib/analyticsHelp";
import { ACTION_META, fmtPct, getTranscriptSegments } from "../../lib/simulation";
import { cn } from "../../lib/utils";

export type AnalyticsSelection =
  | { kind: "kpi"; key: string }
  | { kind: "action"; key: string }
  | { kind: "emotion"; key: EmotionKey }
  | { kind: "wave"; key: string }
  | { kind: "archetype-like"; key: string }
  | { kind: "archetype-dislike"; key: string }
  | { kind: "flow"; index: number }
  | { kind: "segment"; index: number }
  | { kind: "affinity"; key: string }
  | { kind: "lift"; emotion: string }
  | null;

interface Props {
  selection: AnalyticsSelection;
  pinned?: boolean;
  result: SimulationResult;
  onClear: () => void;
}

function selectionKey(sel: AnalyticsSelection): string {
  if (!sel) return "default";
  if (sel.kind === "flow" || sel.kind === "segment") return `${sel.kind}-${sel.index}`;
  if (sel.kind === "lift") return `lift-${sel.emotion}`;
  return `${sel.kind}-${sel.key}`;
}

export function AnalyticsDetailPanel({ selection, pinned = false, result, onClear }: Props) {
  const { summary, advanced } = result.analysis;
  const ins = advanced.insights;
  const v = advanced.virality;

  const content = (() => {
    if (!selection) {
      return {
        title: "Explore wellbeing signals",
        body: "Hover or click any chart, KPI, or list item to see how it relates to emotional impact, harm risk, and audience mental-health framing.",
        stats: [] as { label: string; value: string }[],
      };
    }

    if (selection.kind === "kpi") {
      const help = KPI_HELP[selection.key];
      const stats: { label: string; value: string }[] = [];
      if (selection.key === "virality") {
        stats.push(
          { label: "Score", value: ins.virality_score.toFixed(2) },
          { label: "Amplification", value: `${v.amplification_factor.toFixed(1)}×` },
          { label: "Share rate", value: fmtPct(v.share_conversion_rate) }
        );
      } else if (selection.key === "amplification") {
        stats.push(
          { label: "Factor", value: `${v.amplification_factor.toFixed(2)}×` },
          { label: "Seeds exposed", value: summary.seeds_exposed.toLocaleString() },
          { label: "Total reached", value: summary.total_exposed.toLocaleString() }
        );
      } else if (selection.key === "reach") {
        stats.push(
          { label: "Total", value: summary.total_exposed.toLocaleString() },
          { label: "From shares", value: summary.viral_exposed.toLocaleString() },
          { label: "Unique personas", value: summary.unique_personas_exposed.toLocaleString() }
        );
      } else if (selection.key === "share_rate") {
        stats.push(
          { label: "Observed", value: fmtPct(v.share_conversion_rate) },
          { label: "Predicted", value: fmtPct(advanced.reactions.predicted_share_probability) },
          { label: "Share events", value: String(summary.share_events) }
        );
      } else if (selection.key === "cascade") {
        stats.push(
          { label: "Depth", value: String(v.cascade_depth) },
          { label: "Branching R", value: v.effective_branching_factor.toFixed(2) },
          { label: "Longest chain", value: `${ins.longest_share_chain_depth} hops` }
        );
      } else if (selection.key === "polarization") {
        stats.push(
          { label: "Index", value: ins.polarization_index.toFixed(2) },
          { label: "Likes", value: summary.likes.toLocaleString() },
          { label: "Dislikes", value: summary.dislikes.toLocaleString() }
        );
      }
      return {
        title: help?.title ?? selection.key,
        body: help?.body ?? "",
        formula: help?.formula,
        stats,
      };
    }

    if (selection.kind === "action") {
      const count = summary.action_counts[selection.key] ?? 0;
      const pct = summary.total_exposed > 0 ? count / summary.total_exposed : 0;
      const meta = ACTION_META[selection.key as ReactionKey];
      return {
        title: meta?.label ?? selection.key,
        body: ACTION_HELP[selection.key] ?? "",
        stats: [
          { label: "Agents", value: count.toLocaleString() },
          { label: "Share of exposed", value: fmtPct(pct, 0) },
          {
            label: "Mean model prob",
            value: fmtPct(
              advanced.reactions.mean_probs[selection.key as ReactionKey] ?? 0
            ),
          },
        ],
      };
    }

    if (selection.kind === "emotion") {
      const k = selection.key;
      const prob = advanced.emotions.mean_probs[k];
      const sharer = advanced.emotions.mean_probs_sharers[k];
      const lift = ins.emotion_share_lift[k];
      return {
        title: k,
        body: EMOTION_HELP[k],
        stats: [
          { label: "Mean intensity", value: fmtPct(prob, 0) },
          { label: "Among sharers", value: fmtPct(sharer, 0) },
          { label: "Share lift", value: `${lift >= 0 ? "+" : ""}${lift.toFixed(3)}` },
        ],
      };
    }

    if (selection.kind === "wave") {
      const w = selection.key;
      const exp = advanced.share_patterns.exposures_by_wave[w] ?? 0;
      const sh = advanced.share_patterns.shares_by_wave?.[w] ?? 0;
      const rate = ins.share_rate_by_wave[w];
      const emo = advanced.emotions.mean_probs_by_wave[w];
      const dom = emo
        ? (Object.entries(emo).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—")
        : "—";
      return {
        title: `Wave ${w}`,
        body:
          w === "0"
            ? "Seed audience — initial exposures before any reshares."
            : `Generation ${w} of spread — people exposed because wave ${Number(w) - 1} shared.`,
        stats: [
          { label: "Exposures", value: exp.toLocaleString() },
          { label: "Shares", value: sh.toLocaleString() },
          { label: "Share rate", value: rate != null ? fmtPct(rate) : "—" },
          { label: "Top emotion", value: dom },
        ],
      };
    }

    if (selection.kind === "archetype-like" || selection.kind === "archetype-dislike") {
      const positive = selection.kind === "archetype-like";
      const entry = (positive
        ? result.analysis.top_liked_archetypes
        : result.analysis.top_disliked_archetypes
      ).find((a) => a.archetype === selection.key);
      const agents = result.raw.agents.filter((a) => a.archetype === selection.key);
      const shareRate =
        agents.length > 0
          ? agents.filter((a) => a.sampled_action.includes("share")).length / agents.length
          : 0;
      return {
        title: selection.key.replace(/_/g, " "),
        body: positive
          ? "Archetype with the strongest positive reactions in this simulation."
          : "Archetype with the strongest negative reactions in this simulation.",
        stats: [
          { label: positive ? "Likes" : "Dislikes", value: String(entry?.count ?? 0) },
          { label: "Total exposed", value: agents.length.toLocaleString() },
          { label: "Share rate", value: fmtPct(shareRate, 0) },
        ],
      };
    }

    if (selection.kind === "flow") {
      const f = advanced.share_patterns.top_occupation_flows[selection.index];
      if (!f) return { title: "Route", body: "", stats: [] };
      return {
        title: "Sharing route",
        body: `Content flowed from ${f.from_occupation.replace(/_/g, " ")} to ${f.to_occupation.replace(/_/g, " ")} via a share edge.`,
        stats: [{ label: "Share events", value: String(f.count) }],
      };
    }

    if (selection.kind === "segment") {
      const segments = getTranscriptSegments(result);
      const text = segments[selection.index] ?? "";
      const w = advanced.segment_attention?.mean_weights_all?.[selection.index] ?? 0;
      return {
        title: `Segment ${selection.index + 1}`,
        body: text || "Transcript segment.",
        stats: [{ label: "Mean attention", value: fmtPct(w, 0) }],
      };
    }

    if (selection.kind === "affinity") {
      const val = advanced.affinity[selection.key as keyof typeof advanced.affinity];
      return {
        title: AFFINITY_HELP[selection.key]?.split(" — ")[0] ?? selection.key,
        body: AFFINITY_HELP[selection.key] ?? "",
        stats: [{ label: "Score", value: val != null ? val.toFixed(3) : "—" }],
      };
    }

    if (selection.kind === "lift") {
      const e = ins.top_share_lift_emotions.find((x) => x.emotion === selection.emotion);
      return {
        title: selection.emotion,
        body: "Difference in mean emotion intensity between sharers and non-sharers. Positive = more common among people who reshared.",
        stats: [{ label: "Lift", value: e ? `${e.lift >= 0 ? "+" : ""}${e.lift.toFixed(3)}` : "—" }],
      };
    }

    return { title: "", body: "", stats: [] };
  })();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={selectionKey(selection)}
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -4 }}
        transition={{ duration: 0.22 }}
        className="sim-panel rounded-2xl p-4 md:p-5"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0 flex-1">
            <div className="w-9 h-9 rounded-xl ui-inset grid place-items-center shrink-0">
              <Info className="w-4 h-4 text-white/60" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-[15px] font-semibold text-white capitalize tracking-tight">
                {content.title}
              </h3>
              <p className="text-[12px] text-white/55 leading-relaxed mt-1">{content.body}</p>
              {"formula" in content && content.formula && (
                <p className="text-[10px] text-white/35 font-mono-data mt-2">{content.formula}</p>
              )}
              {content.stats.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-3">
                  {content.stats.map((s) => (
                    <div
                      key={s.label}
                      className="px-2.5 py-1.5 rounded-lg ui-inset"
                    >
                      <div className="text-[9px] uppercase tracking-wider text-white/35">{s.label}</div>
                      <div className="text-[13px] font-semibold tabular-nums text-white">{s.value}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          {pinned && (
            <button
              type="button"
              onClick={onClear}
              className="p-1.5 rounded-lg text-white/40 hover:text-white hover:bg-white/[0.06] transition shrink-0"
              aria-label="Clear selection"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

export function isSelected(
  selection: AnalyticsSelection,
  kind: NonNullable<AnalyticsSelection>["kind"],
  match: string | number
): boolean {
  if (!selection || selection.kind !== kind) return false;
  switch (selection.kind) {
    case "flow":
    case "segment":
      return selection.index === match;
    case "lift":
      return selection.emotion === match;
    default:
      return selection.key === match;
  }
}

export function selectionRing(selected: boolean): string {
  return cn(
    "transition-all duration-200 cursor-pointer",
    selected && "ring-1 ring-white/40 bg-white/[0.04]"
  );
}
