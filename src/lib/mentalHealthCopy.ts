import { viralityTone } from "./theme";

/** Brand & wellbeing framing (UI copy only — model unchanged). */
export const MH = {
  productName: "SHIELD",
  productTagline: "Content wellbeing lab",
  heroHeadline: "Stress-test how your message lands before it spreads.",
  heroSubcopy:
    "See who may benefit, who may struggle, and what could spread — before you publish.",
  composerLabel: "Content to preview",
  composerPlaceholder: "Paste a post, caption, or script you plan to share…",
  composerHint: "We model mood, empathy, and harm — not engagement alone.",
  examplesLabel: "Sample tones",
  videoLabel: "Video",
  videoHint: "Transcribed locally · same wellbeing preview as text",
  runButton: "Preview emotional impact",
  runButtonLoading: "Mapping emotional responses…",
  seedLabel: "Audience sample",
  seedHint:
    "Diverse synthetic viewers — estimate benefit and harm across backgrounds, not one average reaction.",
  simLoadingBadge: "Simulation in progress",
  simLoadingTitle: "Modeling emotional responses",
  simLoadingSub:
    "Predicting who benefits, who withdraws, and how affect may spread through shares.",
  spreadSidebarTitle: "Explore emotional spread",
  spreadSidebarBody:
    "Select a viewer for their personal impact — or chat to hear how your words might help or hurt someone healing.",
  spreadEmptyShares:
    "No reshares this run — lower contagion can mean less unintended harm, even when the message still lands strongly.",
  resultsImpactLabel: "Wellbeing impact",
  tabSpreadDesc: "Who feels what · contagion map",
  tabAnalysisDesc: "Benefit signals · harm risk · affect profile",
  tabModelDesc: "How we model viewer wellbeing",
  footer:
    "Built for mental-health-aware publishing · Local Whisper · 10k-persona affect model · Empathy chat",
  disclaimer:
    "Research & education prototype — not clinical advice. Use insights to reduce harm and promote supportive messaging.",
} as const;

export const MH_PILLARS = [
  {
    title: "Emotional mapping",
    body: "See which affects dominate — empathy, connection, inspiration, curiosity, or joy — and whether they align with the benefit you intended.",
  },
  {
    title: "Harm signal detection",
    body: "Surface dislike-shares and polarization before triggering language amplifies — protect vulnerable viewers before publish.",
  },
  {
    title: "Empathy interviews",
    body: "Chat with simulated viewers in first person. Hear what would have helped — or what felt unsafe — in their own words.",
  },
  {
    title: "Contagion preview",
    body: "Watch emotional spread wave-by-wave. Supportive posts can lift communities; the same dynamics can carry harm if tone misfires.",
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
  "One post can normalize burnout, stigma, or hope; the benefit depends on framing, not reach alone.",
  "Creators deserve a sandbox to preview emotional impact before vulnerable people see it in the wild.",
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
        "Strong emotional spread — review tone for unintended harm or overwhelm before wider benefit can land safely.",
    };
  if (score >= 0.4)
    return {
      label: "Moderate spread",
      tone: viralityTone(score),
      blurb:
        "Healthy organic reach with share waves. Check whether dominant emotions match the benefit you meant to offer.",
    };
  if (score >= 0.1)
    return {
      label: "Limited spread",
      tone: viralityTone(score),
      blurb:
        "Mostly contained to initial viewers. Emotional signal is present but not propagating widely — lower risk of network harm.",
    };
  return {
    label: "Contained",
    tone: viralityTone(score),
    blurb: "Stays within the seed audience — low contagion risk; good when protecting sensitive topics.",
  };
}

export function wellbeingSpreadLabel(score: number): { label: string; tone: string } {
  const v = wellbeingVerdict(score);
  return { label: v.label, tone: v.tone };
}
