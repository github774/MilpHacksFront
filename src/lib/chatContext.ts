import type {
  Agent,
  ChatSimulationPayload,
  SimulationResult,
  TranscribeResult,
  TranscriptSegment,
} from "../api";
import { topEmotion } from "./simulation";

export interface ChatContentContext {
  text: string;
  segments?: Array<{ text: string; start?: number; end?: number; index?: number }>;
  meta?: Record<string, unknown>;
}

export type ChatSimulationContext = ChatSimulationPayload;

export function buildChatContentContext(opts: {
  analyzedText: string;
  inputMode: "text" | "video";
  videoName: string | null;
  transcribeInfo: TranscribeResult | null;
  result: SimulationResult | null;
}): ChatContentContext {
  const { analyzedText, inputMode, videoName, transcribeInfo, result } = opts;

  let text = analyzedText.trim();
  if (!text && result) {
    const rawSegs = result.raw.transcript?.segments;
    if (Array.isArray(rawSegs) && rawSegs.length) {
      text = rawSegs
        .map((s: { text?: string }) => String(s.text || "").trim())
        .filter(Boolean)
        .join(" ");
    }
  }

  const transcriptMeta = (result?.raw.transcript || {}) as Record<string, unknown>;

  let segments: ChatContentContext["segments"];
  if (transcribeInfo?.segments?.length) {
    segments = transcribeInfo.segments.map((s: TranscriptSegment) => ({
      text: s.text,
      start: s.start,
      end: s.end,
    }));
  } else {
    const whisper = transcriptMeta.whisper_segments;
    if (Array.isArray(whisper) && whisper.length) {
      segments = whisper.map((s: TranscriptSegment) => ({
        text: s.text,
        start: s.start,
        end: s.end,
      }));
    } else {
      const rawSegs = result?.raw.transcript?.segments;
      if (Array.isArray(rawSegs) && rawSegs.length) {
        segments = rawSegs.map((s: { text?: string; start?: number; end?: number; index?: number }) => ({
          text: String(s.text || ""),
          start: s.start,
          end: s.end,
          index: s.index,
        }));
      }
    }
  }

  const meta: Record<string, unknown> = {
    source: inputMode,
    input_mode: inputMode,
    text,
    segments,
    video_filename: videoName,
    video: videoName,
    ...(transcriptMeta.source ? { sim_source: transcriptMeta.source } : {}),
  };

  if (transcribeInfo) {
    meta.language = transcribeInfo.language;
    meta.language_probability = transcribeInfo.language_probability;
    meta.duration = transcribeInfo.duration;
    meta.transcribe_seconds = transcribeInfo.transcribe_seconds;
  } else {
    if (transcriptMeta.language) meta.language = transcriptMeta.language;
    if (transcriptMeta.language_probability != null) {
      meta.language_probability = transcriptMeta.language_probability;
    }
    if (transcriptMeta.duration != null) meta.duration = transcriptMeta.duration;
    if (transcriptMeta.transcribe_seconds != null) {
      meta.transcribe_seconds = transcriptMeta.transcribe_seconds;
    }
  }

  if (result) {
    const s = result.analysis.summary;
    meta.simulation_summary = {
      total_exposed: s.total_exposed,
      unique_personas_exposed: s.unique_personas_exposed,
      likes: s.likes,
      dislikes: s.dislikes,
      like_shares: s.like_shares,
      dislike_shares: s.dislike_shares,
      neutral: s.neutral,
      waves_completed: s.waves_completed,
      virality_score: result.analysis.advanced.insights.virality_score,
    };
    meta.top_liked_archetypes = result.analysis.top_liked_archetypes?.slice(0, 5);
    meta.top_disliked_archetypes = result.analysis.top_disliked_archetypes?.slice(0, 5);
    const segAtt = result.analysis.advanced.segment_attention;
    if (segAtt) {
      meta.segment_attention_mean = segAtt.mean_weights_all;
    }
  }

  return { text, segments, meta };
}

export function buildChatSimulationContext(agent: Agent): ChatSimulationContext {
  const { key: dominant_emotion } = topEmotion(agent.emotion_probs);
  return {
    sampled_action: agent.sampled_action,
    dominant_emotion,
    reaction_probs: agent.reaction_probs,
    emotion_probs: agent.emotion_probs,
    segment_weights: agent.segment_weights,
    wave: agent.wave,
    affinity: agent.affinity,
    shared_to_count: agent.shared_to_indices?.length ?? 0,
    exposed_by_share_type: agent.exposed_by_share_type,
  };
}
