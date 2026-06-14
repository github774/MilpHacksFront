import type {
  Agent,
  EmotionKey,
  ReactionKey,
  ShareEdge,
  SimulationResult,
} from "../api";
import { REACTION_STYLE } from "./theme";

export const EMOTION_ORDER: EmotionKey[] = [
  "empathy",
  "relation",
  "inspiration",
  "curiosity",
  "joy",
];

export const REACTION_ORDER: ReactionKey[] = [
  "like_share",
  "like",
  "neutral",
  "dislike",
  "dislike_share",
];

export const ACTION_META: Record<
  ReactionKey,
  { label: string; short: string; color: string }
> = {
  like: { label: "Liked", short: "Like", color: REACTION_STYLE.like.color },
  like_share: { label: "Liked + shared", short: "Like + share", color: REACTION_STYLE.like_share.color },
  neutral: { label: "Passed", short: "Neutral", color: REACTION_STYLE.neutral.color },
  dislike: { label: "Disliked", short: "Dislike", color: REACTION_STYLE.dislike.color },
  dislike_share: { label: "Disliked + shared", short: "Dislike + share", color: REACTION_STYLE.dislike_share.color },
};

export function topEmotion(probs: Record<EmotionKey, number>): {
  key: EmotionKey;
  value: number;
} {
  let best: EmotionKey = "curiosity";
  let bestV = -1;
  for (const k of EMOTION_ORDER) {
    const v = probs[k] ?? 0;
    if (v > bestV) {
      bestV = v;
      best = k;
    }
  }
  return { key: best, value: bestV };
}

export function fmtPct(x: number, d = 0) {
  return `${(x * 100).toFixed(d)}%`;
}

export function getTranscriptSegments(result: SimulationResult): string[] {
  const raw = result.raw.transcript?.segments;
  if (Array.isArray(raw) && raw.length) {
    return raw.map((s: { text?: string }) => String(s.text || "").trim()).filter(Boolean);
  }
  return [];
}

/** Keep graph responsive for large seed runs while preserving cascade structure. */
export function sampleAgentsForGraph(
  agents: Agent[],
  shareEdges: ShareEdge[],
  maxDisplay = 1400
): { agents: Agent[]; sampled: boolean; total: number } {
  if (agents.length <= maxDisplay) {
    return { agents, sampled: false, total: agents.length };
  }

  const mustKeep = new Set<number>();
  for (const e of shareEdges) {
    mustKeep.add(e.from_exposure_id);
  }
  for (const a of agents) {
    if (a.shared_to_indices.length > 0 || a.wave > 0) mustKeep.add(a.exposure_id);
  }

  const seeds = agents.filter((a) => a.wave === 0);
  const seedBudget = Math.min(
    seeds.length,
    Math.max(200, maxDisplay - mustKeep.size)
  );
  const seedStep = Math.max(1, Math.ceil(seeds.length / seedBudget));
  seeds.forEach((a, i) => {
    if (i % seedStep === 0) mustKeep.add(a.exposure_id);
  });

  const picked = agents.filter((a) => mustKeep.has(a.exposure_id));
  if (picked.length > maxDisplay) {
    return { agents: picked.slice(0, maxDisplay), sampled: true, total: agents.length };
  }
  return { agents: picked, sampled: true, total: agents.length };
}
