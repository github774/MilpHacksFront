// API client for the SHIELD FastAPI backend.

const API_BASE =
  (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env
    ?.VITE_API_BASE || "http://localhost:8000";

// --------------------------------------------------------------------------- //
// Types mirroring backend/server.py + network_simulation.py export
// --------------------------------------------------------------------------- //
export type ReactionKey =
  | "neutral"
  | "like"
  | "dislike"
  | "like_share"
  | "dislike_share";

export type EmotionKey =
  | "empathy"
  | "relation"
  | "inspiration"
  | "curiosity"
  | "joy";

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

export interface TranscribeResult {
  text: string;
  segments: TranscriptSegment[];
  language: string | null;
  language_probability: number;
  duration: number;
  transcribe_seconds: number;
}

export interface PersonaRecord {
  uuid?: string;
  professional_persona?: string | null;
  persona?: string | null;
  cultural_background?: string | null;
  skills_and_expertise?: string | null;
  hobbies_and_interests?: string | null;
  career_goals_and_ambitions?: string | null;
  sex?: string | null;
  age?: number | string | null;
  marital_status?: string | null;
  education_level?: string | null;
  bachelors_field?: string | null;
  occupation?: string | null;
  city?: string | null;
  state?: string | null;
  country?: string | null;
  [key: string]: unknown;
}

export interface PersonaEntry {
  persona_index: number;
  uuid: string;
  archetype: string;
  occupation: string;
  record: PersonaRecord;
}

export interface Agent {
  exposure_id: number;
  persona_index: number;
  uuid: string;
  archetype: string;
  occupation: string;
  wave: number;
  exposed_by_index: number | null;
  exposed_by_uuid: string | null;
  exposed_by_share_type: string | null;
  reaction_probs: Record<ReactionKey, number>;
  emotion_probs: Record<EmotionKey, number>;
  segment_weights: number[];
  affinity: number;
  affinity_s: number;
  sampled_action: ReactionKey;
  shared_to_indices: number[];
  shared_to_uuids: string[];
}

export interface ShareEdge {
  from_exposure_id: number;
  from_index: number;
  from_uuid: string;
  from_archetype: string;
  from_occupation: string;
  to_index: number;
  to_uuid: string;
  to_archetype: string;
  to_occupation: string;
  share_type: string;
  wave: number;
}

export interface SimSummary {
  total_exposed: number;
  unique_personas_exposed: number;
  seeds_requested: number;
  seeds_exposed: number;
  viral_exposed: number;
  waves_completed: number;
  likes: number;
  dislikes: number;
  like_shares: number;
  dislike_shares: number;
  share_events: number;
  share_recipients: number;
  share_recipients_simulated: number;
  share_recipients_dropped: number;
  neutral: number;
  action_counts: Record<string, number>;
  warning?: string;
}

export interface ArchetypeCount {
  archetype: string;
  count: number;
}

export interface AdvancedAnalysis {
  emotions: {
    mean_probs: Record<EmotionKey, number>;
    dominant_emotion: EmotionKey;
    mean_probs_by_wave: Record<string, Record<EmotionKey, number>>;
    mean_probs_sharers: Record<EmotionKey, number>;
    mean_probs_non_sharers: Record<EmotionKey, number>;
    top_emotions_among_sharers: { emotion: string; count: number }[];
  };
  reactions: {
    mean_probs: Record<ReactionKey, number>;
    dominant_reaction: ReactionKey;
    predicted_share_probability: number;
    observed_share_rate: number;
    share_rate_delta: number;
  };
  virality: {
    amplification_factor: number;
    viral_reach_ratio: number;
    share_conversion_rate: number;
    effective_branching_factor: number;
    secondary_share_rate: number;
    like_share_ratio: number | null;
    max_share_wave: number | null;
    max_exposure_wave: number;
    cascade_depth: number;
    share_recipients_per_seed: number;
    simulated_recipients_per_seed: number;
  };
  share_patterns: {
    share_type_counts: Record<string, number>;
    shares_by_wave: Record<string, number>;
    exposures_by_wave: Record<string, number>;
    inbound_share_types: Record<string, number>;
    top_sharing_archetypes: ArchetypeCount[];
    top_recipient_archetypes: ArchetypeCount[];
    top_occupation_flows: {
      from_occupation: string;
      to_occupation: string;
      count: number;
    }[];
    unique_sharers: number;
    unique_recipients: number;
    reshare_events: number;
  };
  affinity: {
    mean_all: number | null;
    mean_sharers: number | null;
    mean_non_sharers: number | null;
    mean_viral: number | null;
    mean_affinity_s_all?: number | null;
    mean_affinity_s_sharers?: number | null;
  };
  segment_attention?: {
    segment_count: number;
    mean_weights_all: number[];
    mean_weights_sharers: number[];
  };
  insights: {
    emotion_share_lift: Record<EmotionKey, number>;
    top_share_lift_emotions: { emotion: string; lift: number }[];
    emotion_polarization_like_minus_dislike: Record<EmotionKey, number>;
    polarization_index: number;
    share_rate_by_wave: Record<string, number>;
    longest_share_chain_depth: number;
    virality_score: number;
  };
}

export interface SimulationResult {
  schema_version: string;
  generated_at: string;
  taxonomy: {
    reaction_keys: ReactionKey[];
    emotion_keys: EmotionKey[];
  };
  catalog: { size: number; referenced_persona_count: number };
  config: Record<string, unknown>;
  raw: {
    transcript: Record<string, unknown>;
    personas: Record<string, PersonaEntry>;
    seed_indices: number[];
    agents: Agent[];
    share_edges: ShareEdge[];
  };
  analysis: {
    summary: SimSummary;
    top_liked_archetypes: ArchetypeCount[];
    top_disliked_archetypes: ArchetypeCount[];
    advanced: AdvancedAnalysis;
  };
  timing?: { simulate_seconds: number };
}

export interface HealthStatus {
  status: string;
  simulation_ready: boolean;
  whisper_ready: boolean;
  ollama_up: boolean;
  ollama_models: string[];
  ollama_model: string;
}

export interface SimulateParams {
  text?: string;
  segments?: string[];
  n_seeds: number;
  max_waves?: number;
  sharpness?: number;
  seed?: number;
  transcript_meta?: Record<string, unknown>;
}

// --------------------------------------------------------------------------- //
// Client functions
// --------------------------------------------------------------------------- //
export async function checkHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    return (await res.json()) as HealthStatus;
  } catch {
    return null;
  }
}

export async function transcribeVideo(file: File): Promise<TranscribeResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/transcribe`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Transcription failed.");
  }
  return (await res.json()) as TranscribeResult;
}

export async function runSimulation(
  params: SimulateParams
): Promise<SimulationResult> {
  const controller = new AbortController();
  const seeds = params.n_seeds ?? 80;
  // Large seed runs can take 30–90s; avoid aborting early.
  const timeoutMs = seeds >= 2000 ? 300_000 : seeds >= 500 ? 120_000 : 60_000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_waves: 4,
        sharpness: 2.5,
        seed: 7,
        ...params,
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Simulation failed.");
    }
    return (await res.json()) as SimulationResult;
  } catch (e) {
    if ((e as Error).name === "AbortError") {
      throw new Error(
        `Simulation timed out after ${Math.round(timeoutMs / 1000)}s. Try fewer seeds or fewer waves.`
      );
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export interface ChatSimulationPayload {
  sampled_action: string;
  dominant_emotion: string;
  reaction_probs?: Record<string, number>;
  emotion_probs?: Record<string, number>;
  segment_weights?: number[];
  wave?: number;
  affinity?: number;
  shared_to_count?: number;
  exposed_by_share_type?: string | null;
}

export interface ChatPersonaParams {
  messages: { role: "user" | "persona"; content: string }[];
  persona_index?: number;
  persona_record?: PersonaRecord;
  archetype?: string;
  occupation?: string;
  reaction?: {
    sampled_action?: string;
    dominant_emotion?: string;
    reaction_probs?: Record<string, number>;
    emotion_probs?: Record<string, number>;
  };
  content_text?: string;
  content_segments?: Array<{ text: string; start?: number; end?: number; index?: number }>;
  content_meta?: Record<string, unknown>;
  simulation?: ChatSimulationPayload;
}

// Streams persona reply tokens. Calls onToken for each delta.
// Returns the full text. Throws on error payloads from the backend.
export async function streamPersonaChat(
  params: ChatPersonaParams,
  onToken: (full: string, delta: string) => void,
  signal?: AbortSignal
): Promise<string> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat failed.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      let payload: { token?: string; done?: boolean; error?: string };
      try {
        payload = JSON.parse(line);
      } catch {
        continue;
      }
      if (payload.error) throw new Error(payload.error);
      if (payload.token) {
        full += payload.token;
        onToken(full, payload.token);
      }
    }
  }
  return full;
}
