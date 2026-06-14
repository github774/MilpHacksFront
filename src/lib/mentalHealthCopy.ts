import { viralityTone } from "./theme";

/** Brand & wellbeing framing (UI copy only — model unchanged). */
export const MH = {
  productName: "SHIELD",
  productTagline: "Content wellbeing lab",
  heroHeadline: "Stress-test how your message lands before it spreads.",
  heroSubcopy: "",
  composerLabel: "Content",
  composerPlaceholder: "Paste a post, caption, or script…",
  videoLabel: "Video",
  runButton: "Run wellbeing simulation",
  runButtonLoading: "Mapping emotional response…",
  seedLabel: "Diverse audience sample",
  footer:
    "Built for mental-health-aware publishing · Local Whisper · 10k-persona affect model · Empathy chat",
  disclaimer:
    "Research & education prototype — not clinical advice. Use insights to reduce harm and promote supportive messaging.",
} as const;

export const MH_PILLARS = [
  {
    title: "Emotional mapping",
    body: "See which brain-linked affects dominate — empathy, connection, inspiration, curiosity, or joy — across your audience.",
  },
  {
    title: "Harm signal detection",
    body: "Surface dislike-share patterns and polarization before outrage or toxic narratives amplify through the network.",
  },
  {
    title: "Empathy interviews",
    body: "Chat with simulated viewers in first person. Hear how your words might land on someone struggling or healing.",
  },
  {
    title: "Contagion preview",
    body: "Watch emotional spread wave-by-wave — the same dynamics that help supportive posts lift communities can also carry harm.",
  },
] as const;

export const MH_EXAMPLES = [
  {
    label: "Supportive",
    text: "If you're having a hard week, you're not broken — you're human. Reaching out isn't weakness. I waited too long to talk to someone, and it changed everything. You deserve support.",
  },
  {
    label: "Toxic hustle",
    text: "If you're not grinding 24/7 you're going to lose. Sleep is a choice and most of you are choosing to be average. Comfort is the enemy. Stop scrolling and go build something.",
  },
  {
    label: "Hope & recovery",
    text: "Three years ago I couldn't get out of bed most mornings. Today I'm sharing this because someone told me recovery was possible. If you're in the dark right now: please stay. Help exists.",
  },
  {
    label: "Anxiety trigger",
    text: "BREAKING: A new study says the algorithm is quietly rewiring an entire generation's attention span and nobody is talking about it. The numbers are genuinely terrifying. Share this before it gets buried.",
  },
] as const;

export const MH_WHY_IT_MATTERS = [
  "Social content shapes mood, self-worth, and help-seeking — especially for young audiences.",
  "One viral post can normalize burnout, stigma, or hope depending on how it's framed.",
  "Creators deserve a sandbox to preview emotional impact, not just click-through rates.",
] as const;

export function wellbeingVerdict(score: number): {
  label: string;
  tone: string;
  blurb: string;
} {
  if (score >= 1)
    return {
      label: "High contagion",
      tone: viralityTone(score),
      blurb:
        "Strong emotional spread — message may reach far beyond your seed audience. Review tone for unintended harm or overwhelm.",
    };
  if (score >= 0.4)
    return {
      label: "Moderate spread",
      tone: viralityTone(score),
      blurb:
        "Healthy organic reach with measurable share waves. Check whether dominant emotions align with your wellbeing intent.",
    };
  if (score >= 0.1)
    return {
      label: "Limited spread",
      tone: viralityTone(score),
      blurb:
        "Mostly contained to initial viewers. Emotional signal is present but not propagating widely.",
    };
  return {
    label: "Contained",
    tone: viralityTone(score),
    blurb: "Stays within the seed audience — low risk of network-wide emotional contagion.",
  };
}

export function wellbeingSpreadLabel(score: number): { label: string; tone: string } {
  const v = wellbeingVerdict(score);
  return { label: v.label, tone: v.tone };
}
