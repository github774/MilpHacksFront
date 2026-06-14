import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { SendIcon, XIcon, MapPin, Sparkles } from "lucide-react";
import {
  streamPersonaChat,
  type Agent,
  type PersonaEntry,
} from "../api";
import {
  buildChatSimulationContext,
  type ChatContentContext,
} from "../lib/chatContext";
import { topEmotion } from "../lib/simulation";
import { REACTION_STYLE } from "../lib/theme";

interface PersonaChatProps {
  agent: Agent;
  persona: PersonaEntry | null;
  content: ChatContentContext;
  ollamaUp: boolean;
  onClose: () => void;
}

interface Msg {
  role: "user" | "persona";
  content: string;
  hidden?: boolean;
}

const ACTION_BADGE: Record<string, { label: string; color: string }> = {
  like: { label: "Liked it", color: REACTION_STYLE.like.color },
  like_share: { label: "Liked + reshared", color: REACTION_STYLE.like_share.color },
  neutral: { label: "Scrolled past", color: REACTION_STYLE.neutral.color },
  dislike: { label: "Disliked it", color: REACTION_STYLE.dislike.color },
  dislike_share: { label: "Disliked + reshared", color: REACTION_STYLE.dislike_share.color },
};
function field(persona: PersonaEntry | null, key: string): string {
  const v = persona?.record?.[key];
  if (v == null) return "";
  const s = String(v).trim();
  return s.toLowerCase() === "nan" ? "" : s;
}

export function PersonaChat({ agent, persona, content, ollamaUp, onClose }: PersonaChatProps) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const action = ACTION_BADGE[agent.sampled_action] || ACTION_BADGE.neutral;
  const { key: emotion, value: emotionVal } = topEmotion(agent.emotion_probs);

  const occupation = (persona?.occupation || agent.occupation || "person").replace(/_/g, " ");
  const age = field(persona, "age");
  const sex = field(persona, "sex");
  const locale = [field(persona, "city"), field(persona, "state")].filter(Boolean).join(", ");
  const blurb =
    field(persona, "professional_persona") || field(persona, "persona") || agent.archetype;

  const reaction = {
    sampled_action: agent.sampled_action,
    dominant_emotion: emotion,
    reaction_probs: agent.reaction_probs,
    emotion_probs: agent.emotion_probs,
  };
  const simulation = buildChatSimulationContext(agent);

  const runStream = async (convo: Msg[]) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStreaming(true);
    setError(null);
    setMessages([...convo, { role: "persona", content: "" }]);
    try {
      await streamPersonaChat(
        {
          messages: convo.map((m) => ({ role: m.role, content: m.content })),
          persona_index: agent.persona_index,
          persona_record: persona?.record,
          archetype: agent.archetype,
          occupation: agent.occupation,
          reaction,
          content_text: content.text,
          content_segments: content.segments,
          content_meta: content.meta,
          simulation,
        },
        (full) => {
          setMessages((prev) => {
            const next = [...prev];
            next[next.length - 1] = { role: "persona", content: full };
            return next;
          });
        },
        ctrl.signal
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message || "Chat failed.");
        setMessages((prev) => prev.filter((m) => m.content !== ""));
      }
    } finally {
      setStreaming(false);
    }
  };

  // Auto-open with the persona's first reaction when the selected agent changes.
  useEffect(() => {
    setInput("");
    setError(null);
    if (!ollamaUp) {
      setMessages([]);
      return;
    }
    const opening: Msg[] = [
      {
        role: "user",
        hidden: true,
        content:
          "You just scrolled past this content in your feed. In one or two sentences, react honestly as yourself — how did it affect your mood? Cite something specific from the transcript.",
      },
    ];
    runStream(opening);
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent.exposure_id, ollamaUp]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = () => {
    if (!input.trim() || streaming || !ollamaUp) return;
    const visible = messages.filter((m) => m.content !== "" || m.role === "user");
    const convo: Msg[] = [...visible, { role: "user", content: input.trim() }];
    setInput("");
    runStream(convo);
  };

  const display = messages.filter((m) => !m.hidden);

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      transition={{ duration: 0.3 }}
      className="flex flex-col h-full glass-panel rounded-2xl overflow-hidden"
    >
      {/* Header */}
      <div className="p-4 border-b border-white/10 ui-panel-header">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="w-10 h-10 rounded-full grid place-items-center text-base font-bold shrink-0 border"
              style={{
                background: `${action.color}1a`,
                borderColor: `${action.color}55`,
                color: action.color,
              }}
            >
              {occupation.charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold text-white capitalize truncate">
                {occupation}
              </div>
              <div className="text-[11px] text-white/45 truncate">
                {[age && `${age}`, sex && sex.toLowerCase(), locale].filter(Boolean).join(" · ")}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/10 text-white/50 hover:text-white transition shrink-0"
            title="Back to agent profile"
          >
            <XIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="flex items-center gap-2 mt-3 flex-wrap">
          <span
            className="text-[10.5px] font-medium px-2 py-0.5 rounded-full"
            style={{ background: `${action.color}1a`, color: action.color }}
          >
            {action.label}
          </span>
          <span className="text-[10.5px] font-medium px-2 py-0.5 rounded-full bg-white/[0.06] text-white/60 capitalize">
            top emotion: {emotion} ({Math.round(emotionVal * 100)}%)
          </span>
          {locale && (
            <span className="text-[10.5px] text-white/40 flex items-center gap-1">
              <MapPin className="w-3 h-3" /> {locale}
            </span>
          )}
        </div>
        {blurb && (
          <p className="text-[11px] text-white/45 mt-2.5 line-clamp-2 leading-relaxed">{blurb}</p>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 min-h-[180px]">
        {!ollamaUp && (
          <div className="text-center py-8">
            <Sparkles className="w-6 h-6 text-white/30 mx-auto mb-2" />
            <p className="text-[12px] text-white/50 max-w-[240px] mx-auto leading-relaxed">
              Start Ollama to run empathy interviews with simulated viewers.
              <br />
              <code className="text-[11px] text-white/70 bg-white/10 px-1.5 py-0.5 rounded mt-1 inline-block">
                ollama serve
              </code>
            </p>
          </div>
        )}
        {display.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] px-3 py-2 rounded-2xl text-[12.5px] leading-relaxed ${
                m.role === "user"
                  ? "bg-white text-black rounded-br-md"
                  : "bg-white/[0.07] text-white/90 rounded-bl-md border border-white/8"
              }`}
            >
              {m.content || (
                <span className="inline-flex gap-1 items-center py-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-white/50 animate-pulse" />
                  <span className="w-1.5 h-1.5 rounded-full bg-white/50 animate-pulse [animation-delay:0.15s]" />
                  <span className="w-1.5 h-1.5 rounded-full bg-white/50 animate-pulse [animation-delay:0.3s]" />
                </span>
              )}
            </div>
          </div>
        ))}
        {error && (
          <div className="text-[11px] text-white/70 bg-white/[0.06] border border-white/15 rounded-lg px-3 py-2">
            {error}
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Input */}
      <div className="p-3 border-t border-white/8">
        <div className="flex items-center gap-2 bg-white/[0.05] border border-white/10 rounded-xl px-3 py-2 focus-within:border-white/25 transition">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={!ollamaUp || streaming}
            placeholder={ollamaUp ? "Ask how this landed — or what would have helped…" : "Ollama offline"}
            className="flex-1 bg-transparent text-[13px] text-white placeholder:text-white/30 outline-none disabled:opacity-50"
          />
          <button
            onClick={send}
            disabled={!input.trim() || streaming || !ollamaUp}
            className="p-1.5 rounded-lg bg-white text-black disabled:opacity-25 disabled:cursor-not-allowed hover:bg-white/90 transition"
          >
            <SendIcon className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
