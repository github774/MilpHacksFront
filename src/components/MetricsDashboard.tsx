import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  Zap,
  Users,
  Share2,
  GitBranch,
  Scale,
  Flame,
  Heart,
  ArrowRight,
} from "lucide-react";
import type { EmotionKey, SimulationResult } from "../api";
import { MentalHealthAnalytics } from "./MentalHealthAnalytics";
import { SegmentAttention } from "./SegmentAttention";
import {
  AnalyticsDetailPanel,
  isSelected,
  selectionRing,
  type AnalyticsSelection,
} from "./analytics/AnalyticsDetailPanel";
import { MONO, REACTION_STYLE } from "../lib/theme";
import { wellbeingSpreadLabel } from "../lib/mentalHealthCopy";
import { cn } from "../lib/utils";

interface Props {
  result: SimulationResult;
}

const EMOTION_ORDER: EmotionKey[] = [
  "empathy",
  "relation",
  "inspiration",
  "curiosity",
  "joy",
];

type EmotionView = "all" | "sharers" | "non_sharers";

const fmtPct = (x: number, d = 0) => `${(x * 100).toFixed(d)}%`;
const fmtNum = (x: number, d = 2) => x.toFixed(d);

function spreadScoreLabel(score: number): { label: string; tone: string } {
  return wellbeingSpreadLabel(score);
}

function Card({
  children,
  className = "",
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`glass-panel glass-panel-hover rounded-2xl p-4 ${className}`}
    >
      {children}
    </motion.div>
  );
}

function KpiCard({
  icon,
  label,
  value,
  sub,
  accent,
  delay,
  selected,
  dimmed,
  onSelect,
  onHover,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
  accent?: string;
  delay: number;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
  onHover: (active: boolean) => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className={cn(
        "text-left w-full rounded-2xl transition-all duration-200 focus:outline-none focus-visible:ring-1 focus-visible:ring-white/40",
        selectionRing(selected),
        dimmed && "opacity-45 hover:opacity-70"
      )}
    >
      <Card delay={delay} className="flex flex-col justify-between min-h-[112px] h-full pointer-events-none">
        <div className="flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-wider text-white/45 font-medium">
            {label}
          </span>
          <span className="text-white/40" style={{ color: accent }}>
            {icon}
          </span>
        </div>
        <div>
          <div
            className="text-3xl font-bold tracking-tight tabular-nums"
            style={{ color: accent || "#fff" }}
          >
            {value}
          </div>
          {sub && <div className="text-[11px] text-white/45 mt-0.5">{sub}</div>}
        </div>
      </Card>
    </button>
  );
}

function BarRow({
  label,
  value,
  max,
  color,
  display,
  selected,
  dimmed,
  onSelect,
  onHover,
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  display: string;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
  onHover: (active: boolean) => void;
}) {
  const pct = max > 0 ? Math.max(2, (value / max) * 100) : 0;
  return (
    <button
      type="button"
      onClick={onSelect}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className={cn(
        "w-full flex items-center gap-2.5 rounded-lg px-2 py-1.5 -mx-2 transition-all duration-200",
        "hover:bg-white/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
        selectionRing(selected),
        dimmed && "opacity-40 hover:opacity-65"
      )}
    >
      <span className="text-[11px] text-white/55 w-[120px] truncate capitalize text-left">
        {label}
      </span>
      <div className="flex-1 h-2 rounded-full bg-white/[0.06] overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{
            background: color,
            boxShadow: selected ? `0 0 16px ${color}88` : `0 0 8px ${color}44`,
          }}
        />
      </div>
      <span className="text-[11px] text-white/70 w-[52px] text-right tabular-nums">{display}</span>
    </button>
  );
}

function EmotionRadar({
  probs,
  dominant,
  selected,
  highlight,
  onSelect,
  onHighlight,
}: {
  probs: Record<EmotionKey, number>;
  dominant: EmotionKey;
  selected: EmotionKey | null;
  highlight: EmotionKey | null;
  onSelect: (k: EmotionKey) => void;
  onHighlight: (k: EmotionKey | null) => void;
}) {
  const size = 200;
  const c = size / 2;
  const rMax = size / 2 - 30;
  const n = EMOTION_ORDER.length;
  const angle = (i: number) => -Math.PI / 2 + (i / n) * Math.PI * 2;
  const point = (i: number, r: number) => ({
    x: c + Math.cos(angle(i)) * r,
    y: c + Math.sin(angle(i)) * r,
  });
  const dataPoly = EMOTION_ORDER.map((k, i) => {
    const p = point(i, rMax * Math.min(1, probs[k]));
    return `${p.x},${p.y}`;
  }).join(" ");

  const active = selected ?? highlight;

  return (
    <svg width="100%" viewBox={`0 0 ${size} ${size}`} className="overflow-visible">
      {[0.25, 0.5, 0.75, 1].map((g) => (
        <polygon
          key={g}
          points={EMOTION_ORDER.map((_, i) => {
            const p = point(i, rMax * g);
            return `${p.x},${p.y}`;
          }).join(" ")}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={1}
        />
      ))}
      {EMOTION_ORDER.map((_, i) => {
        const p = point(i, rMax);
        return (
          <line
            key={i}
            x1={c}
            y1={c}
            x2={p.x}
            y2={p.y}
            stroke={
              active === EMOTION_ORDER[i] ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.07)"
            }
            strokeWidth={active === EMOTION_ORDER[i] ? 1.5 : 1}
          />
        );
      })}
      <motion.polygon
        initial={{ opacity: 0, scale: 0.6 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        style={{ transformOrigin: "center" }}
        points={dataPoly}
        fill="rgba(255,255,255,0.16)"
        stroke="#ffffff"
        strokeWidth={1.5}
      />
      {EMOTION_ORDER.map((k, i) => {
        const p = point(i, rMax * Math.min(1, probs[k]));
        const isActive = active === k;
        const isDom = k === dominant;
        return (
          <g key={k}>
            <circle
              cx={p.x}
              cy={p.y}
              r={14}
              fill="transparent"
              className="cursor-pointer"
              onClick={() => onSelect(k)}
              onMouseEnter={() => onHighlight(k)}
              onMouseLeave={() => onHighlight(null)}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={isActive ? 4 : 2.5}
              fill={isActive ? "#ffffff" : isDom ? "#e5e5e5" : "rgba(255,255,255,0.7)"}
              className="pointer-events-none"
              style={{ filter: isActive ? "drop-shadow(0 0 6px rgba(255,255,255,0.8))" : undefined }}
            />
          </g>
        );
      })}
      {EMOTION_ORDER.map((k, i) => {
        const p = point(i, rMax + 16);
        const isActive = active === k;
        return (
          <text
            key={k}
            x={p.x}
            y={p.y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize="9"
            fill={isActive ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.55)"}
            className="capitalize pointer-events-none select-none"
            fontWeight={isActive ? 600 : 400}
          >
            {k}
          </text>
        );
      })}
    </svg>
  );
}

function ViewToggle<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex gap-1 p-0.5 rounded-lg ui-inset">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          onClick={() => onChange(o.id)}
          className={cn(
            "px-2 py-1 rounded-md text-[10px] font-medium transition",
            value === o.id ? "bg-white text-black" : "text-white/45 hover:text-white/75"
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function MetricsDashboard({ result }: Props) {
  const [selection, setSelection] = useState<AnalyticsSelection>(null);
  const [hover, setHover] = useState<AnalyticsSelection>(null);
  const [emotionView, setEmotionView] = useState<EmotionView>("all");
  const [hoverEmotion, setHoverEmotion] = useState<EmotionKey | null>(null);

  const detail = selection ?? hover;

  const { summary, advanced, top_liked_archetypes, top_disliked_archetypes } =
    result.analysis;
  const v = advanced.virality;
  const ins = advanced.insights;

  const vScore = ins.virality_score;
  const vl = spreadScoreLabel(vScore);

  const actionOrder = ["like_share", "like", "neutral", "dislike", "dislike_share"];
  const actionColor: Record<string, string> = {
    like_share: REACTION_STYLE.like_share.color,
    like: REACTION_STYLE.like.color,
    neutral: REACTION_STYLE.neutral.color,
    dislike: REACTION_STYLE.dislike.color,
    dislike_share: REACTION_STYLE.dislike_share.color,
  };
  const actionLabel: Record<string, string> = {
    like_share: "Like + share",
    like: "Like",
    neutral: "Neutral",
    dislike: "Dislike",
    dislike_share: "Dislike + share",
  };
  const maxAction = Math.max(1, ...Object.values(summary.action_counts));

  const waves = Object.keys(advanced.share_patterns.exposures_by_wave).sort(
    (a, b) => Number(a) - Number(b)
  );
  const maxWaveExp = Math.max(
    1,
    ...Object.values(advanced.share_patterns.exposures_by_wave)
  );

  const emotionProbs = useMemo(() => {
    if (emotionView === "sharers") return advanced.emotions.mean_probs_sharers;
    if (emotionView === "non_sharers") return advanced.emotions.mean_probs_non_sharers;
    return advanced.emotions.mean_probs;
  }, [emotionView, advanced.emotions]);

  const maxEmotion = Math.max(0.001, ...EMOTION_ORDER.map((k) => emotionProbs[k]));
  const liftMax = Math.max(
    0.001,
    ...ins.top_share_lift_emotions.map((e) => Math.abs(e.lift))
  );

  const focusedEmotion =
    selection?.kind === "emotion"
      ? selection.key
      : selection?.kind === "lift"
        ? (selection.emotion as EmotionKey)
        : hoverEmotion;

  const hasFocus = selection !== null || hover !== null || hoverEmotion !== null;

  const select = (next: AnalyticsSelection) => {
    setSelection((prev) => {
      if (!next) return null;
      if (
        prev &&
        prev.kind === next.kind &&
        ("key" in prev && "key" in next
          ? prev.key === next.key
          : prev.kind === "flow" && next.kind === "flow"
            ? prev.index === next.index
            : prev.kind === "segment" && next.kind === "segment"
              ? prev.index === next.index
              : prev.kind === "lift" && next.kind === "lift"
                ? prev.emotion === next.emotion
                : false)
      ) {
        return null;
      }
      return next;
    });
  };

  return (
    <div className="space-y-5">
      <MentalHealthAnalytics />

      <SegmentAttention
        result={result}
        title="Which lines carried the most emotional weight"
        scrollable
        selectedIndex={selection?.kind === "segment" ? selection.index : null}
        highlightIndex={null}
        onSelect={(i) => select({ kind: "segment", index: i })}
        onHighlight={(i) => setHover(i !== null ? { kind: "segment", index: i } : null)}
      />

      <AnalyticsDetailPanel
        selection={detail}
        pinned={selection !== null}
        result={result}
        onClear={() => {
          setSelection(null);
          setHover(null);
          setHoverEmotion(null);
        }}
      />

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <KpiCard
          delay={0}
          icon={<Flame className="w-4 h-4" />}
          label="Spread score"
          value={fmtNum(vScore)}
          sub={vl.label}
          accent={vl.tone}
          selected={isSelected(selection, "kpi", "virality")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "virality") && hover?.kind !== "kpi"}
          onSelect={() => select({ kind: "kpi", key: "virality" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "virality" } : null)}
        />
        <KpiCard
          delay={0.05}
          icon={<Zap className="w-4 h-4" />}
          label="Emotional amplification"
          value={`${fmtNum(v.amplification_factor, 1)}×`}
          sub="reach vs first viewers"
          selected={isSelected(selection, "kpi", "amplification")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "amplification")}
          onSelect={() => select({ kind: "kpi", key: "amplification" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "amplification" } : null)}
        />
        <KpiCard
          delay={0.1}
          icon={<Users className="w-4 h-4" />}
          label="Total reached"
          value={summary.total_exposed.toLocaleString()}
          sub={`${summary.viral_exposed.toLocaleString()} from shares`}
          selected={isSelected(selection, "kpi", "reach")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "reach")}
          onSelect={() => select({ kind: "kpi", key: "reach" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "reach" } : null)}
        />
        <KpiCard
          delay={0.15}
          icon={<Share2 className="w-4 h-4" />}
          label="Share rate"
          value={fmtPct(v.share_conversion_rate, 1)}
          sub={`${summary.share_events} share events`}
          selected={isSelected(selection, "kpi", "share_rate")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "share_rate")}
          onSelect={() => select({ kind: "kpi", key: "share_rate" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "share_rate" } : null)}
        />
        <KpiCard
          delay={0.2}
          icon={<GitBranch className="w-4 h-4" />}
          label="Cascade depth"
          value={`${v.cascade_depth}`}
          sub={`R ≈ ${fmtNum(v.effective_branching_factor, 2)}`}
          selected={isSelected(selection, "kpi", "cascade")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "cascade")}
          onSelect={() => select({ kind: "kpi", key: "cascade" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "cascade" } : null)}
        />
        <KpiCard
          delay={0.25}
          icon={<Scale className="w-4 h-4" />}
          label="Polarization"
          value={
            ins.polarization_index > 0
              ? `+${fmtNum(ins.polarization_index)}`
              : fmtNum(ins.polarization_index)
          }
          sub={ins.polarization_index >= 0 ? "skews negative affect" : "skews positive affect"}
          accent={Math.abs(ins.polarization_index) > 0.3 ? MONO.muted : undefined}
          selected={isSelected(selection, "kpi", "polarization")}
          dimmed={hasFocus && !isSelected(selection, "kpi", "polarization")}
          onSelect={() => select({ kind: "kpi", key: "polarization" })}
          onHover={(a) => setHover(a ? { kind: "kpi", key: "polarization" } : null)}
        />
      </div>

      {/* Affinity + model calibration */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Card delay={0} className="p-4">
          <h3 className="text-[13px] font-semibold text-white mb-3">Persona–content affinity</h3>
          <div className="grid grid-cols-2 gap-3 text-[12px]">
            {(
              [
                { key: "mean_all", label: "All exposed" },
                { key: "mean_sharers", label: "Sharers" },
                { key: "mean_non_sharers", label: "Non-sharers" },
                { key: "mean_viral", label: "Viral cohort" },
              ] as const
            ).map((row) => {
              const val = advanced.affinity?.[row.key];
              const selected = isSelected(selection, "affinity", row.key);
              return (
                <button
                  key={row.key}
                  type="button"
                  onClick={() => select({ kind: "affinity", key: row.key })}
                  onMouseEnter={() => setHover({ kind: "affinity", key: row.key })}
                  onMouseLeave={() => setHover(null)}
                  className={cn(
                    "rounded-lg px-3 py-2 text-left transition-all",
                    "hover:bg-white/[0.05] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                    selected ? "bg-white/[0.08] ring-1 ring-white/25" : "bg-white/[0.03]",
                    hasFocus && !selected && "opacity-50"
                  )}
                >
                  <div className="text-white/40 text-[10px]">{row.label}</div>
                  <div className="text-white font-semibold tabular-nums">
                    {val != null ? val.toFixed(3) : "—"}
                  </div>
                </button>
              );
            })}
          </div>
        </Card>
        <Card delay={0} className="p-4">
          <h3 className="text-[13px] font-semibold text-white mb-3">Share prediction vs reality</h3>
          <div className="space-y-3">
            <div className="flex justify-between text-[12px]">
              <span className="text-white/50">PathPool predicted share rate</span>
              <span className="text-white tabular-nums">
                {fmtPct(advanced.reactions.predicted_share_probability, 1)}
              </span>
            </div>
            <div className="flex justify-between text-[12px]">
              <span className="text-white/50">Observed share rate</span>
              <span className="text-white tabular-nums">
                {fmtPct(advanced.reactions.observed_share_rate, 1)}
              </span>
            </div>
            <button
              type="button"
              onClick={() => select({ kind: "kpi", key: "share_rate" })}
              className="w-full flex justify-between text-[12px] rounded-lg px-2 py-1.5 -mx-2 hover:bg-white/[0.04] transition"
            >
              <span className="text-white/50">Delta (sim − model)</span>
              <span
                className="tabular-nums font-medium"
                style={{
                  color: advanced.reactions.share_rate_delta >= 0 ? MONO.fg : MONO.dim,
                }}
              >
                {advanced.reactions.share_rate_delta >= 0 ? "+" : ""}
                {fmtPct(advanced.reactions.share_rate_delta, 1)}
              </span>
            </button>
            <p className="text-[11px] text-white/40 leading-relaxed pt-1 border-t border-white/[0.05]">
              Seeds: {summary.seeds_exposed.toLocaleString()} exposed of{" "}
              {summary.seeds_requested.toLocaleString()} requested · {summary.waves_completed} waves
              · {summary.share_recipients_dropped} share targets dropped
            </p>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card delay={0.1} className="lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-white/50" /> How viewers responded
            </h3>
            <span className="text-[11px] text-white/40">{summary.total_exposed} agents</span>
          </div>
          <div className="space-y-1">
            {actionOrder.map((a) => {
              const count = summary.action_counts[a] || 0;
              const pct =
                summary.total_exposed > 0
                  ? `${((count / summary.total_exposed) * 100).toFixed(0)}%`
                  : "0%";
              return (
                <BarRow
                  key={a}
                  label={actionLabel[a]}
                  value={count}
                  max={maxAction}
                  color={actionColor[a]}
                  display={`${count} · ${pct}`}
                  selected={isSelected(selection, "action", a)}
                  dimmed={hasFocus && !isSelected(selection, "action", a)}
                  onSelect={() => select({ kind: "action", key: a })}
                  onHover={(active) =>
                    setHover(active ? { kind: "action", key: a } : null)
                  }
                />
              );
            })}
          </div>
        </Card>

        <Card delay={0.15}>
          <div className="flex flex-col gap-2 mb-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Heart className="w-4 h-4 text-white/50" /> Affect profile
              </h3>
              <span
                className="text-[11px] font-medium capitalize px-2 py-0.5 rounded-full"
                style={{ background: "rgba(255,255,255,0.12)", color: MONO.fg }}
              >
                {advanced.emotions.dominant_emotion}
              </span>
            </div>
            <ViewToggle<EmotionView>
              options={[
                { id: "all", label: "All" },
                { id: "sharers", label: "Sharers" },
                { id: "non_sharers", label: "Non-sharers" },
              ]}
              value={emotionView}
              onChange={setEmotionView}
            />
          </div>
          <EmotionRadar
            probs={emotionProbs}
            dominant={advanced.emotions.dominant_emotion}
            selected={selection?.kind === "emotion" ? selection.key : null}
            highlight={hoverEmotion}
            onSelect={(k) => select({ kind: "emotion", key: k })}
            onHighlight={(k) => {
              setHoverEmotion(k);
              setHover(k ? { kind: "emotion", key: k } : null);
            }}
          />
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card delay={0.1}>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <GitBranch className="w-4 h-4 text-white/50" /> Contagion by wave
          </h3>
          <div className="flex items-end gap-2 h-[140px]">
            {waves.map((w) => {
              const exp = advanced.share_patterns.exposures_by_wave[w] || 0;
              const sh = advanced.share_patterns.shares_by_wave?.[w] || 0;
              const selected = isSelected(selection, "wave", w);
              return (
                <button
                  key={w}
                  type="button"
                  onClick={() => select({ kind: "wave", key: w })}
                  onMouseEnter={() => setHover({ kind: "wave", key: w })}
                  onMouseLeave={() => setHover(null)}
                  className={cn(
                    "flex-1 flex flex-col items-center justify-end gap-1 h-full rounded-lg transition-all",
                    "hover:bg-white/[0.03] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                    selected && "bg-white/[0.05] ring-1 ring-white/20",
                    hasFocus && !selected && "opacity-45"
                  )}
                >
                  <span className="text-[10px] text-white/50 tabular-nums">{exp}</span>
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${(exp / maxWaveExp) * 100}%` }}
                    transition={{ duration: 0.6, delay: Number(w) * 0.08 }}
                    className="w-full rounded-t-md relative min-h-[4px]"
                    style={{
                      background: selected
                        ? "linear-gradient(180deg,#ffffff,#a3a3a3)"
                        : "linear-gradient(180deg,#ffffff,#737373)",
                    }}
                  >
                    {sh > 0 && (
                      <span className="absolute -top-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)]" />
                    )}
                  </motion.div>
                  <span className="text-[10px] text-white/40">w{w}</span>
                </button>
              );
            })}
          </div>
          <p className="text-[11px] text-white/40 mt-3 leading-relaxed">
            Click a wave for exposure, share count, and dominant emotion. Longest chain:{" "}
            {ins.longest_share_chain_depth} hops · {advanced.share_patterns.unique_sharers} unique
            sharers
          </p>
        </Card>

        <Card delay={0.15}>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-white/50" /> What drives sharing
          </h3>
          <div className="space-y-1">
            {ins.top_share_lift_emotions.map((e) => (
              <BarRow
                key={e.emotion}
                label={e.emotion}
                value={Math.abs(e.lift)}
                max={liftMax}
                color={e.lift >= 0 ? MONO.fg : MONO.dim}
                display={`${e.lift >= 0 ? "+" : ""}${fmtNum(e.lift)}`}
                selected={isSelected(selection, "lift", e.emotion)}
                dimmed={hasFocus && !isSelected(selection, "lift", e.emotion) && focusedEmotion !== e.emotion}
                onSelect={() => select({ kind: "lift", emotion: e.emotion })}
                onHover={(active) =>
                  setHover(active ? { kind: "lift", emotion: e.emotion } : null)
                }
              />
            ))}
          </div>
        </Card>

        <Card delay={0.2}>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <Heart className="w-4 h-4 text-white/50" /> Mean emotion intensity
          </h3>
          <div className="space-y-1">
            {EMOTION_ORDER.map((k) => (
              <BarRow
                key={k}
                label={k}
                value={emotionProbs[k]}
                max={maxEmotion}
                color={
                  k === advanced.emotions.dominant_emotion || focusedEmotion === k
                    ? MONO.fg
                    : MONO.mid
                }
                display={fmtPct(emotionProbs[k], 0)}
                selected={isSelected(selection, "emotion", k)}
                dimmed={!!focusedEmotion && focusedEmotion !== k}
                onSelect={() => select({ kind: "emotion", key: k })}
                onHover={(active) => {
                  setHoverEmotion(active ? k : null);
                  setHover(active ? { kind: "emotion", key: k } : null);
                }}
              />
            ))}
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card delay={0.1}>
          <h3 className="text-sm font-semibold text-white mb-3">Most emotionally uplifted groups</h3>
          <div className="space-y-1">
            {top_liked_archetypes.slice(0, 5).map((a) => (
              <button
                key={a.archetype}
                type="button"
                onClick={() => select({ kind: "archetype-like", key: a.archetype })}
                onMouseEnter={() => setHover({ kind: "archetype-like", key: a.archetype })}
                onMouseLeave={() => setHover(null)}
                className={cn(
                  "w-full flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 -mx-2 transition-all",
                  "hover:bg-white/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                  isSelected(selection, "archetype-like", a.archetype) &&
                    "bg-white/[0.06] ring-1 ring-white/20",
                  hasFocus && !isSelected(selection, "archetype-like", a.archetype) && "opacity-45"
                )}
              >
                <span className="text-[12px] text-white/70 capitalize truncate">{a.archetype}</span>
                <span className="text-[11px] text-white tabular-nums font-medium">{a.count}</span>
              </button>
            ))}
            {top_liked_archetypes.length === 0 && (
              <p className="text-[11px] text-white/35">No strong positive cohort.</p>
            )}
          </div>
        </Card>

        <Card delay={0.15}>
          <h3 className="text-sm font-semibold text-white mb-3">Groups most negatively affected</h3>
          <div className="space-y-1">
            {top_disliked_archetypes.slice(0, 5).map((a) => (
              <button
                key={a.archetype}
                type="button"
                onClick={() => select({ kind: "archetype-dislike", key: a.archetype })}
                onMouseEnter={() => setHover({ kind: "archetype-dislike", key: a.archetype })}
                onMouseLeave={() => setHover(null)}
                className={cn(
                  "w-full flex items-center justify-between gap-2 rounded-lg px-2 py-1.5 -mx-2 transition-all",
                  "hover:bg-white/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                  isSelected(selection, "archetype-dislike", a.archetype) &&
                    "bg-white/[0.06] ring-1 ring-white/20",
                  hasFocus && !isSelected(selection, "archetype-dislike", a.archetype) && "opacity-45"
                )}
              >
                <span className="text-[12px] text-white/70 capitalize truncate">{a.archetype}</span>
                <span className="text-[11px] text-white/50 tabular-nums font-medium">{a.count}</span>
              </button>
            ))}
            {top_disliked_archetypes.length === 0 && (
              <p className="text-[11px] text-white/35">No strong negative cohort.</p>
            )}
          </div>
        </Card>

        <Card delay={0.2}>
          <h3 className="text-sm font-semibold text-white mb-3">Top sharing routes</h3>
          <div className="space-y-1">
            {advanced.share_patterns.top_occupation_flows.slice(0, 5).map((f, i) => (
              <button
                key={i}
                type="button"
                onClick={() => select({ kind: "flow", index: i })}
                onMouseEnter={() => setHover({ kind: "flow", index: i })}
                onMouseLeave={() => setHover(null)}
                className={cn(
                  "w-full flex items-center gap-1.5 text-[11px] rounded-lg px-2 py-1.5 -mx-2 transition-all",
                  "hover:bg-white/[0.04] focus:outline-none focus-visible:ring-1 focus-visible:ring-white/30",
                  selection?.kind === "flow" && selection.index === i &&
                    "bg-white/[0.06] ring-1 ring-white/20",
                  hasFocus && !(selection?.kind === "flow" && selection.index === i) && "opacity-45"
                )}
              >
                <span className="text-white/70 capitalize truncate flex-1 text-right">
                  {f.from_occupation.replace(/_/g, " ")}
                </span>
                <ArrowRight className="w-3 h-3 text-white/30 shrink-0" />
                <span className="text-white/70 capitalize truncate flex-1">
                  {f.to_occupation.replace(/_/g, " ")}
                </span>
                <span className="text-white/40 tabular-nums w-5 text-right">{f.count}</span>
              </button>
            ))}
            {advanced.share_patterns.top_occupation_flows.length === 0 && (
              <p className="text-[11px] text-white/35">No shares propagated.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
