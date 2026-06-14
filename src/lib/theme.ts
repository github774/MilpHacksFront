/** Monochrome design tokens — white / gray only, no chromatic accents. */

export const MONO = {
  fg: "#ffffff",
  bright: "#e5e5e5",
  mid: "#a3a3a3",
  muted: "#737373",
  dim: "#525252",
  faint: "#8b8b94",
  glow: "rgba(255,255,255,0.9)",
  glowMid: "rgba(255,255,255,0.55)",
  glowSoft: "rgba(255,255,255,0.35)",
  glowFaint: "rgba(255,255,255,0.12)",
  rgba: (a: number) => `rgba(255,255,255,${a})`,
} as const;

export function viralityTone(score: number): string {
  if (score >= 1) return MONO.fg;
  if (score >= 0.4) return MONO.bright;
  if (score >= 0.1) return MONO.mid;
  return MONO.muted;
}

export const REACTION_STYLE: Record<
  string,
  { color: string; glow: string }
> = {
  like_share: { color: MONO.fg, glow: MONO.glow },
  like: { color: MONO.bright, glow: MONO.glowMid },
  neutral: { color: MONO.faint, glow: "rgba(139,139,148,0.35)" },
  dislike: { color: MONO.muted, glow: "rgba(115,115,115,0.5)" },
  dislike_share: { color: MONO.dim, glow: "rgba(82,82,82,0.65)" },
};
