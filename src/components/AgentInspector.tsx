import { MapPin, MessageCircle, X } from "lucide-react";
import type { Agent, PersonaEntry } from "../api";
import {
  ACTION_META,
  EMOTION_ORDER,
  REACTION_ORDER,
  fmtPct,
  topEmotion,
} from "../lib/simulation";
import { SegmentAttention } from "./SegmentAttention";
import type { SimulationResult } from "../api";

interface Props {
  agent: Agent;
  persona: PersonaEntry | null;
  result: SimulationResult;
  onClose: () => void;
  onChat: () => void;
  chatOpen: boolean;
}

function field(persona: PersonaEntry | null, key: string): string {
  const v = persona?.record?.[key];
  if (v == null) return "";
  const s = String(v).trim();
  return s.toLowerCase() === "nan" ? "" : s;
}

export function AgentInspector({
  agent,
  persona,
  result,
  onClose,
  onChat,
  chatOpen,
}: Props) {
  const action = ACTION_META[agent.sampled_action];
  const { key: emotion, value: emotionVal } = topEmotion(agent.emotion_probs);
  const occupation = (persona?.occupation || agent.occupation || "person").replace(/_/g, " ");
  const locale = [field(persona, "city"), field(persona, "state")].filter(Boolean).join(", ");
  const blurb =
    field(persona, "professional_persona") || field(persona, "persona") || agent.archetype;

  return (
    <div className="flex flex-col h-full rounded-xl glass-panel overflow-hidden">
      <div className="p-4 ui-panel-header">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-[15px] font-semibold text-white capitalize truncate">{occupation}</div>
            <div className="text-[11px] text-white/45 mt-0.5 capitalize truncate">{agent.archetype}</div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/10 text-white/45 hover:text-white shrink-0"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Vote + top emotion — primary read */}
        <div className="mt-4 grid grid-cols-2 gap-2">
          <div
            className="rounded-xl px-3 py-2.5 border"
            style={{
              background: `${action.color}12`,
              borderColor: `${action.color}33`,
            }}
          >
            <div className="text-[9px] uppercase tracking-wider text-white/45 mb-1">Vote</div>
            <div className="text-[13px] font-semibold" style={{ color: action.color }}>
              {action.label}
            </div>
            <div className="text-[10px] text-white/40 mt-0.5">wave {agent.wave}</div>
          </div>
          <div className="rounded-xl px-3 py-2.5 border border-white/20 bg-white/[0.06]">
            <div className="text-[9px] uppercase tracking-wider text-white/45 mb-1">Top emotion</div>
            <div className="text-[13px] font-semibold text-white capitalize">{emotion}</div>
            <div className="text-[10px] text-white/40 mt-0.5 tabular-nums">{fmtPct(emotionVal, 0)} intensity</div>
          </div>
        </div>

        {locale && (
          <div className="flex items-center gap-1 mt-2 text-[10px] text-white/40">
            <MapPin className="w-3 h-3" /> {locale}
          </div>
        )}
        {blurb && (
          <p className="text-[11px] text-white/45 mt-2 line-clamp-2 leading-relaxed">{blurb}</p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-white/40 mb-2">Reaction probabilities</div>
          <div className="space-y-1.5">
            {REACTION_ORDER.map((k) => {
              const v = agent.reaction_probs[k] ?? 0;
              const isSampled = k === agent.sampled_action;
              return (
                <div key={k} className="flex items-center gap-2">
                  <span className="text-[10px] text-white/50 w-[72px] truncate">{ACTION_META[k].short}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(2, v * 100)}%`,
                        background: isSampled ? ACTION_META[k].color : "rgba(255,255,255,0.2)",
                      }}
                    />
                  </div>
                  <span className="text-[10px] tabular-nums text-white/55 w-8 text-right">{fmtPct(v, 0)}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <div className="text-[10px] uppercase tracking-wider text-white/40 mb-2">Emotion profile</div>
          <div className="space-y-1.5">
            {EMOTION_ORDER.map((k) => {
              const v = agent.emotion_probs[k] ?? 0;
              const isTop = k === emotion;
              return (
                <div key={k} className="flex items-center gap-2">
                  <span className="text-[10px] text-white/50 w-[72px] capitalize">{k}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.max(2, v * 100)}%`,
                        background: isTop ? "#ffffff" : "rgba(255,255,255,0.18)",
                      }}
                    />
                  </div>
                  <span className="text-[10px] tabular-nums text-white/55 w-8 text-right">{fmtPct(v, 0)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {agent.segment_weights?.length > 0 && (
          <SegmentAttention
            result={result}
            agentWeights={agent.segment_weights}
            title="What this persona focused on"
            compact
          />
        )}
      </div>

      {!chatOpen && (
        <div className="p-3 border-t border-white/[0.06]">
          <button
            onClick={onChat}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-white text-black text-[13px] font-semibold hover:bg-white/90 transition"
          >
            <MessageCircle className="w-4 h-4" /> Empathy interview
          </button>
        </div>
      )}
    </div>
  );
}
