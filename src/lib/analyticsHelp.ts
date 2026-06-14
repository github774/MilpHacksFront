import type { EmotionKey } from "../api";

export const KPI_HELP: Record<
  string,
  { title: string; body: string; formula?: string }
> = {
  virality: {
    title: "Emotional spread score",
    body: "How far content travels beyond the first viewers through shares — a proxy for emotional contagion. High scores mean the message may amplify widely; check whether that helps or harms wellbeing.",
    formula: "Based on amplification, share rate, and cascade depth",
  },
  amplification: {
    title: "Emotional amplification",
    body: "Total people reached divided by initial viewers. Higher values mean each first viewer led to more secondary exposure — stronger network ripple.",
    formula: "total_exposed ÷ seeds_exposed",
  },
  reach: {
    title: "People reached",
    body: "Every simulated viewing event, including people who saw the content because someone shared it emotionally.",
  },
  share_rate: {
    title: "Share rate",
    body: "Share of viewers who reshared (supportive or negative). High negative shares can signal harm amplification.",
    formula: "share_events ÷ total_exposed",
  },
  cascade: {
    title: "Contagion depth",
    body: "How many waves of sharing occurred after the first push. Deeper cascades mean the emotional message kept traveling.",
  },
  polarization: {
    title: "Polarization index",
    body: "Whether reactions skew negative (+) or positive (−). Strong polarization can indicate divisive or triggering content.",
  },
};

export const ACTION_HELP: Record<string, string> = {
  like_share: "Viewer felt positively and reshared — supportive emotional contagion.",
  like: "Positive reaction without sharing. Warmth without amplification.",
  neutral: "Viewer saw the content but it did not land strongly enough to react.",
  dislike: "Negative reaction, contained — did not propagate further.",
  dislike_share: "Negative reaction that still spread — potential harm or outrage amplification.",
};

export const EMOTION_HELP: Record<EmotionKey, string> = {
  empathy: "Compassion and emotional resonance — feeling with the speaker.",
  relation: "Connection, belonging, and shared identity — 'this is about people like me.'",
  inspiration: "Hope, motivation, and uplift — often protective for mental health when authentic.",
  curiosity: "Information-seeking and engagement — can be healthy or anxiety-driving depending on framing.",
  joy: "Delight, humor, and positive affect — mood-lifting when sincere.",
};

export const AFFINITY_HELP: Record<string, string> = {
  mean_all: "Average persona–content fit across everyone who saw the post.",
  mean_sharers: "Affinity among viewers who reshared — often higher when content matches sharer identity.",
  mean_non_sharers: "Affinity among viewers who did not share.",
  mean_viral: "Affinity among the contagion cohort (exposures from shares, not first viewers).",
};
