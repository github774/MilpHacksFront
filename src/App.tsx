import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Brain,
  Cpu,
  Film,
  Flame,
  Loader2,
  Network,
  Play,
  RotateCcw,
  Settings2,
  Share2,
  Shield,
  Sparkles,
  Type,
  Upload,
  Users,
  X,
  Zap,
} from "lucide-react";
import { cn } from "./lib/utils";
import {
  checkHealth,
  runSimulation,
  transcribeVideo,
  type Agent,
  type HealthStatus,
  type PersonaEntry,
  type SimulationResult,
  type TranscribeResult,
} from "./api";
import { NetworkGraph } from "./components/NetworkGraph";
import { MetricsDashboard } from "./components/MetricsDashboard";
import { PersonaChat } from "./components/PersonaChat";
import { AgentInspector } from "./components/AgentInspector";
import { ModelArchitecture } from "./components/ModelArchitecture";
import { BrainSimulation, EMOTION_REGIONS } from "./components/BrainSimulation";
import { TextScramble } from "./components/TextScramble";
import { UploadAnimation } from "./components/UploadAnimation";
import { MentalHealthAnalytics } from "./components/MentalHealthAnalytics";
import { HeroSplineBackground } from "./components/HeroSplineBackground";
import { SilkBackground } from "./components/SilkBackground";
import { sampleAgentsForGraph, topEmotion } from "./lib/simulation";
import { buildChatContentContext } from "./lib/chatContext";
import { MH, MH_EXAMPLES, wellbeingVerdict } from "./lib/mentalHealthCopy";
import { MONO } from "./lib/theme";

type Phase = "input" | "results";
type InputMode = "text" | "video";

const EXAMPLES = MH_EXAMPLES;

const SEED_MIN = 1;
const SEED_MAX = 10000;
const clampSeeds = (v: number) =>
  Number.isNaN(v) ? SEED_MIN : Math.max(SEED_MIN, Math.min(SEED_MAX, Math.floor(v)));

function StatusDot({ up }: { up: boolean }) {
  return (
    <span
      className={cn("w-1.5 h-1.5 rounded-full", up ? "bg-white" : "bg-white/25")}
      style={up ? { boxShadow: "0 0 8px rgba(255,255,255,0.8)" } : undefined}
    />
  );
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("input");
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [segments, setSegments] = useState<string[] | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [transcribeInfo, setTranscribeInfo] = useState<TranscribeResult | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);

  const [nSeeds, setNSeeds] = useState(80);
  const [maxWaves, setMaxWaves] = useState(4);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [simulating, setSimulating] = useState(false);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [runKey, setRunKey] = useState(0);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [analyzedText, setAnalyzedText] = useState("");

  const fileRef = useRef<HTMLInputElement>(null);
  const videoObjectUrlRef = useRef<string | null>(null);

  const clearVideoPreview = useCallback(() => {
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
      videoObjectUrlRef.current = null;
    }
    setVideoPreviewUrl(null);
    setVideoName(null);
    setTranscribeInfo(null);
  }, []);

  useEffect(() => {
    return () => {
      if (videoObjectUrlRef.current) URL.revokeObjectURL(videoObjectUrlRef.current);
    };
  }, []);

  // Poll backend health. Keep last-known-good through transient blips so a
  // single timed-out poll (e.g. during a heavy run) doesn't disable chat.
  const healthFailsRef = useRef(0);
  useEffect(() => {
    let alive = true;
    const ping = async () => {
      const h = await checkHealth();
      if (!alive) return;
      if (h) {
        healthFailsRef.current = 0;
        setHealth(h);
      } else {
        healthFailsRef.current += 1;
        if (healthFailsRef.current >= 2) setHealth(null);
      }
    };
    ping();
    const id = setInterval(ping, 6000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const backendUp = !!health;
  const ollamaUp = !!health?.ollama_up;

  const handleVideo = useCallback(async (file: File) => {
    if (videoObjectUrlRef.current) URL.revokeObjectURL(videoObjectUrlRef.current);
    const previewUrl = URL.createObjectURL(file);
    videoObjectUrlRef.current = previewUrl;
    setVideoPreviewUrl(previewUrl);
    setVideoName(file.name);
    setTranscribing(true);
    setError(null);
    setTranscribeInfo(null);
    try {
      const r = await transcribeVideo(file);
      setText(r.text);
      setSegments(r.segments.map((s) => s.text).filter(Boolean));
      setTranscribeInfo(r);
    } catch (e) {
      setError((e as Error).message || "Transcription failed.");
    } finally {
      setTranscribing(false);
    }
  }, []);

  const onTextChange = (val: string) => {
    setText(val);
    setSegments(null);
    if (inputMode === "text") setTranscribeInfo(null);
  };

  const run = async () => {
    const content = text.trim();
    if (!content || simulating) return;
    setSimulating(true);
    setError(null);
    setSelectedAgent(null);
    setResult(null);
    setPhase("results");
    setAnalyzedText(content);
    try {
      const r = await runSimulation({
        ...(segments && segments.length ? { segments } : { text: content }),
        n_seeds: clampSeeds(nSeeds),
        max_waves: maxWaves,
        transcript_meta: {
          source: inputMode,
          video: videoName,
          ...(transcribeInfo && {
            language: transcribeInfo.language,
            language_probability: transcribeInfo.language_probability,
            duration: transcribeInfo.duration,
            transcribe_seconds: transcribeInfo.transcribe_seconds,
            whisper_segments: transcribeInfo.segments,
          }),
        },
      });
      setResult(r);
      setRunKey((k) => k + 1);
    } catch (e) {
      setError((e as Error).message || "Simulation failed.");
    } finally {
      setSimulating(false);
    }
  };

  const reset = () => {
    setPhase("input");
    setResult(null);
    setSelectedAgent(null);
    setError(null);
    clearVideoPreview();
    setText("");
    setSegments(null);
  };

  const selectedPersona =
    result && selectedAgent
      ? result.raw.personas[String(selectedAgent.persona_index)] || null
      : null;

  // Notable spreaders for the "interview" suggestions.
  const topSpreaders =
    result?.raw.agents
      .filter((a) => a.shared_to_indices.length > 0)
      .sort((a, b) => b.shared_to_indices.length - a.shared_to_indices.length)
      .slice(0, 6) || [];

  return (
    <div className="min-h-screen bg-[#030303] text-white relative">
      {phase === "input" ? <HeroSplineBackground /> : <SilkBackground />}

      {/* Top bar */}
      <header className="sticky top-0 z-40 liquid-glass-header">
        <div className="max-w-[1400px] mx-auto px-5 h-14 flex items-center justify-between">
          <button onClick={reset} className="flex items-center group" aria-label="Home">
            <div className="w-8 h-8 rounded-xl liquid-glass-nested grid place-items-center">
              <Shield className="w-4 h-4 text-white/90" strokeWidth={2} />
            </div>
          </button>

          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-3 px-3 py-1.5 rounded-full liquid-glass-nested text-[11px] text-white/70">
              <span className="flex items-center gap-1.5">
                <StatusDot up={backendUp} /> affect model
              </span>
              <span className="w-px h-3 bg-white/10" />
              <span className="flex items-center gap-1.5">
                <StatusDot up={ollamaUp} /> empathy chat
              </span>
            </div>
            {phase === "results" && (
              <button
                onClick={reset}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.06] hover:bg-white/10 border border-white/[0.08] text-[12px] font-medium transition"
              >
                <RotateCcw className="w-3.5 h-3.5" /> New
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-5 relative z-10">
        <AnimatePresence mode="wait">
          {phase === "input" ? (
            <motion.div
              key="input"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, y: -10 }}
              className="py-16 md:py-24"
            >
              {/* Hero — minimal */}
              <div className="text-center max-w-xl mx-auto mb-8 flex flex-col items-center hero-readable">
                <motion.div
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mb-5"
                >
                  <TextScramble text="SHIELD" autoPlay />
                </motion.div>
                <motion.p
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.08 }}
                  className="text-white/70 text-[15px] leading-relaxed"
                >
                  {MH.heroHeadline}
                </motion.p>
              </div>

              {/* Composer */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="max-w-2xl mx-auto liquid-glass-prompt rounded-[28px] p-3 md:p-5"
              >
                {/* Mode tabs */}
                <div className="flex items-center gap-1 p-1 liquid-glass-segment rounded-2xl mb-4">
                  {[
                    { id: "text" as const, label: "Paste text", icon: Type },
                    { id: "video" as const, label: "Upload video", icon: Film },
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setInputMode(m.id)}
                      className={cn(
                        "flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-[13px] font-semibold transition-all duration-200",
                        inputMode === m.id
                          ? "liquid-glass-segment-active"
                          : "text-white/65 hover:text-white hover:bg-white/[0.06]"
                      )}
                    >
                      <m.icon className="w-4 h-4" /> {m.label}
                    </button>
                  ))}
                </div>

                <div className="px-1 pb-1">
                  {inputMode === "text" ? (
                    <>
                      <label className="block text-[11px] uppercase tracking-[0.14em] text-white/55 font-mono-data mb-2 px-1">
                        {MH.composerLabel}
                      </label>
                      <div className="liquid-glass-input rounded-2xl p-3 transition-all">
                        <textarea
                          value={text}
                          onChange={(e) => onTextChange(e.target.value)}
                          placeholder={MH.composerPlaceholder}
                          rows={5}
                          className="w-full bg-transparent text-[15px] text-white placeholder:text-white/45 outline-none resize-none leading-relaxed min-h-[120px]"
                        />
                      </div>
                      <div className="flex flex-wrap gap-1.5 mt-3 px-1">
                        <span className="text-[10px] text-white/40 w-full mb-0.5">Examples</span>
                        {EXAMPLES.map((ex) => (
                          <button
                            key={ex.label}
                            onClick={() => onTextChange(ex.text)}
                            className="text-[11px] px-2.5 py-1 rounded-full liquid-glass-chip text-white/85 hover:text-white"
                          >
                            {ex.label}
                          </button>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="space-y-3">
                      <label className="block text-[11px] uppercase tracking-[0.14em] text-white/60 font-mono-data mb-1 px-1">
                        {MH.videoLabel}
                      </label>
                      <input
                        ref={fileRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) handleVideo(f);
                          e.target.value = "";
                        }}
                      />
                      {videoPreviewUrl ? (
                        <>
                          <div className="relative rounded-2xl overflow-hidden liquid-glass-nested">
                            <video
                              src={videoPreviewUrl}
                              controls={!transcribing}
                              playsInline
                              preload="metadata"
                              className="w-full aspect-video max-h-[min(340px,52vh)] object-contain bg-[#0a0a0a]"
                            />
                            {transcribing && (
                              <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px] flex items-center justify-center p-4">
                                <UploadAnimation
                                  fileName={videoName ?? "video"}
                                  className="min-h-0 w-full max-w-md"
                                />
                              </div>
                            )}
                          </div>
                          <div className="flex items-center justify-between gap-2 px-0.5">
                            <span className="text-[11px] text-white/65 truncate">{videoName}</span>
                            {!transcribing && (
                              <button
                                type="button"
                                onClick={() => fileRef.current?.click()}
                                className="text-[11px] text-white/50 hover:text-white shrink-0 transition"
                              >
                                Replace video
                              </button>
                            )}
                          </div>
                        </>
                      ) : transcribing ? (
                        <UploadAnimation fileName={videoName ?? "video"} />
                      ) : (
                        <button
                          onClick={() => fileRef.current?.click()}
                          className="w-full border border-dashed border-white/20 hover:border-white/35 rounded-2xl py-8 flex flex-col items-center gap-2 transition-all liquid-glass-nested"
                        >
                          <Upload className="w-6 h-6 text-white/75" />
                          <span className="text-[13px] text-white/90 font-semibold">
                            Click to upload a video
                          </span>
                          <span className="text-[11px] text-white/55">
                            mp4, mov, webm — transcribed locally with Whisper
                          </span>
                        </button>
                      )}
                      {transcribeInfo && (
                        <div className="p-3 rounded-2xl liquid-glass-nested">
                          <div className="flex items-center gap-3 text-[11px] text-white/60 mb-1.5">
                            <span>{transcribeInfo.language?.toUpperCase() || "—"}</span>
                            <span>{transcribeInfo.duration.toFixed(1)}s clip</span>
                            <span>{transcribeInfo.segments.length} segments</span>
                            <span className="ml-auto text-white/60">transcribed ✓</span>
                          </div>
                          <p className="text-[13px] text-white/90 leading-relaxed line-clamp-4">
                            {transcribeInfo.text}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div className="mt-4 pt-4 px-1 border-t border-white/10">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-[12px] text-white/75 font-medium flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5" /> {MH.seedLabel}
                    </label>
                    <div className="flex items-center gap-1.5">
                      <input
                        type="number"
                        min={SEED_MIN}
                        max={SEED_MAX}
                        value={nSeeds}
                        onChange={(e) => setNSeeds(clampSeeds(Number(e.target.value)))}
                        onFocus={(e) => e.target.select()}
                        className="w-[74px] liquid-glass-nested rounded-lg px-2 py-0.5 text-[13px] font-semibold tabular-nums text-white text-right outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                      />
                      <span className="text-[12px] text-white/60">personas</span>
                    </div>
                  </div>
                  <input
                    type="range"
                    min={SEED_MIN}
                    max={SEED_MAX}
                    step={1}
                    value={nSeeds}
                    onChange={(e) => setNSeeds(Number(e.target.value))}
                    className="w-full accent-white cursor-pointer h-2"
                  />
                  <p className="text-[10px] text-white/50 mt-1.5">
                    Diverse synthetic personas from the Nemotron catalog — up to{" "}
                    {SEED_MAX.toLocaleString()}
                  </p>
                  <div className="flex items-center justify-between mt-2">
                    <button
                      onClick={() => setShowAdvanced((s) => !s)}
                      className="text-[11px] text-white/55 hover:text-white flex items-center gap-1 transition"
                    >
                      <Settings2 className="w-3 h-3" /> Advanced
                    </button>
                    <div className="flex gap-1">
                      {[80, 500, 2000, 10000].map((n) => (
                        <button
                          key={n}
                          onClick={() => setNSeeds(n)}
                          className={cn(
                            "text-[11px] px-2 py-0.5 rounded-md transition",
                            nSeeds === n
                              ? "liquid-glass-segment-active px-2 py-0.5 rounded-lg"
                              : "text-white/55 hover:text-white hover:bg-white/[0.06] px-2 py-0.5 rounded-lg"
                          )}
                        >
                          {n}
                        </button>
                      ))}
                    </div>
                  </div>
                  <AnimatePresence>
                    {showAdvanced && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="flex items-center justify-between pt-3 mt-1">
                          <label className="text-[12px] text-white/75">
                            Emotional contagion waves
                          </label>
                          <span className="text-[13px] font-semibold tabular-nums">
                            {maxWaves}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={1}
                          max={6}
                          step={1}
                          value={maxWaves}
                          onChange={(e) => setMaxWaves(Number(e.target.value))}
                          className="w-full accent-white cursor-pointer"
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Run */}
                <div className="pt-3 px-1">
                  <button
                    onClick={run}
                    disabled={!text.trim() || transcribing || simulating}
                    className="w-full flex items-center justify-center gap-2 py-4 rounded-2xl liquid-glass-button text-black font-semibold text-[15px] disabled:opacity-35 disabled:cursor-not-allowed"
                  >
                    {simulating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" /> {MH.runButtonLoading}
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" fill="currentColor" /> {MH.runButton}
                      </>
                    )}
                  </button>
                </div>
              </motion.div>

              {!backendUp && (
                <p className="text-center text-[12px] text-white/75 mt-5 hero-readable px-4">
                  Engine offline — start the backend:{" "}
                  <code className="liquid-glass-nested px-1.5 py-0.5 rounded-lg text-white/90">
                    uvicorn server:app --port 8000
                  </code>{" "}
                  in <code className="liquid-glass-nested px-1.5 py-0.5 rounded-lg text-white/90">backend/</code>
                </p>
              )}
              {error && (
                <p className="text-center text-[12px] text-red-200 mt-4 hero-readable px-4 liquid-glass-nested py-2.5 rounded-2xl border border-red-300/20 max-w-2xl mx-auto">
                  {error}
                </p>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="results"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="py-6"
            >
              {simulating && !result && (
                <div className="max-w-3xl mx-auto py-8">
                  <div className="sim-panel rounded-2xl overflow-hidden relative">
                    <div className="sim-scan-line rounded-2xl" />
                    <div className="h-[min(380px,50vh)] min-h-[300px]">
                      <BrainSimulation isActive emotions={{}} />
                    </div>
                    <div className="border-t border-white/10 px-6 py-5 text-center ui-panel-header">
                      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-[11px] text-white font-medium mb-3">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        Simulation in progress
                      </div>
                      <h2 className="text-xl font-semibold tracking-tight">
                        Modeling {nSeeds.toLocaleString()} emotional responses
                      </h2>
                      <p className="text-white/45 text-[13px] mt-1.5 max-w-md mx-auto">
                        Embedding your message · predicting affect & reactions · tracing emotional contagion
                      </p>
                      <div className="flex justify-center gap-6 mt-5 text-[10px] uppercase tracking-widest text-white/30 font-mono-data">
                        <span className="text-white/60">Understand</span>
                        <span>→</span>
                        <span>Feel</span>
                        <span>→</span>
                        <span>Spread</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {error && !result && (
                <div className="max-w-md mx-auto mt-24 text-center glass-panel rounded-2xl p-8">
                  <X className="w-8 h-8 text-white/50 mx-auto mb-3" />
                  <h2 className="text-lg font-semibold mb-1">Simulation failed</h2>
                  <p className="text-white/50 text-[13px] mb-5">{error}</p>
                  <div className="flex gap-2 justify-center">
                    <button
                      onClick={reset}
                      className="px-4 py-2 rounded-lg bg-white/[0.06] border border-white/10 text-[13px] hover:bg-white/10 transition"
                    >
                      Back
                    </button>
                    <button
                      onClick={run}
                      className="px-4 py-2 rounded-lg bg-white text-black text-[13px] font-medium hover:bg-white/90 transition"
                    >
                      Retry
                    </button>
                  </div>
                </div>
              )}

              {result && (
                <ResultsView
                  result={result}
                  runKey={runKey}
                  analyzedText={analyzedText}
                  inputMode={inputMode}
                  videoName={videoName}
                  transcribeInfo={transcribeInfo}
                  selectedAgent={selectedAgent}
                  selectedPersona={selectedPersona}
                  onSelect={setSelectedAgent}
                  topSpreaders={topSpreaders}
                  ollamaUp={ollamaUp}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

function ResultsView({
  result,
  runKey,
  analyzedText,
  inputMode,
  videoName,
  transcribeInfo,
  selectedAgent,
  selectedPersona,
  onSelect,
  topSpreaders,
  ollamaUp,
}: {
  result: SimulationResult;
  runKey: number;
  analyzedText: string;
  inputMode: InputMode;
  videoName: string | null;
  transcribeInfo: TranscribeResult | null;
  selectedAgent: Agent | null;
  selectedPersona: PersonaEntry | null;
  onSelect: (a: Agent | null) => void;
  topSpreaders: Agent[];
  ollamaUp: boolean;
}) {
  const [tab, setTab] = useState<"spread" | "analysis" | "model">("spread");
  const [chatOpen, setChatOpen] = useState(false);

  const v = wellbeingVerdict(result.analysis.advanced.insights.virality_score);
  const s = result.analysis.summary;
  const warning = s.warning;
  const vir = result.analysis.advanced.virality;
  const emo = result.analysis.advanced.emotions;
  const score = result.analysis.advanced.insights.virality_score;

  const chatContent = useMemo(
    () =>
      buildChatContentContext({
        analyzedText,
        inputMode,
        videoName,
        transcribeInfo,
        result,
      }),
    [analyzedText, inputMode, videoName, transcribeInfo, result]
  );

  const graphData = useMemo(
    () => sampleAgentsForGraph(result.raw.agents, result.raw.share_edges),
    [result]
  );

  const brainEmotions = useMemo(() => {
    if (selectedAgent?.emotion_probs) return selectedAgent.emotion_probs;
    return emo.mean_probs;
  }, [selectedAgent, emo.mean_probs]);

  const brainDominant = useMemo(() => {
    if (selectedAgent?.emotion_probs) return topEmotion(selectedAgent.emotion_probs).key;
    return emo.dominant_emotion;
  }, [selectedAgent, emo.dominant_emotion]);

  const brainDominantRegion = useMemo(
    () => EMOTION_REGIONS.find((r) => r.key === brainDominant)?.region ?? brainDominant,
    [brainDominant]
  );

  useEffect(() => {
    setChatOpen(false);
  }, [selectedAgent?.exposure_id]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [tab]);

  const tabs = [
    { id: "spread" as const, label: "Emotional spread", icon: Network, desc: "Contagion map & brain affect" },
    { id: "analysis" as const, label: "Wellbeing analytics", icon: BarChart3, desc: "Affect profile & harm signals" },
    { id: "model" as const, label: "How it works", icon: Cpu, desc: "Simulation pipeline" },
  ];

  const kpis = [
    {
      label: "People reached",
      value: s.total_exposed.toLocaleString(),
      sub: `${s.unique_personas_exposed.toLocaleString()} unique viewers simulated`,
      icon: Users,
      accent: MONO.bright,
    },
    {
      label: "Emotional amplification",
      value: `${vir.amplification_factor.toFixed(1)}×`,
      sub: `${s.viral_exposed.toLocaleString()} reached via shares`,
      icon: Zap,
      accent: MONO.bright,
    },
    {
      label: "Share events",
      value: String(s.share_events),
      sub: `${s.like_shares} supportive · ${s.dislike_shares} negative shares`,
      icon: Share2,
      accent: MONO.fg,
    },
    {
      label: "Dominant affect",
      value: emo.dominant_emotion,
      sub: `${Math.round((emo.mean_probs[emo.dominant_emotion] ?? 0) * 100)}% mean intensity across audience`,
      icon: Sparkles,
      accent: MONO.mid,
      capitalize: true,
    },
  ];

  return (
    <div className="space-y-6 pb-12 relative">
      {/* Hero command strip */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
        className="sim-panel rounded-2xl overflow-hidden"
      >
        <div className="p-5 md:p-6 lg:p-7">
          <div className="flex flex-col lg:flex-row lg:items-start gap-6 lg:gap-8">
            {/* Verdict + score */}
            <div className="flex items-start gap-4 shrink-0">
              <div
                className="relative w-[72px] h-[72px] rounded-2xl grid place-items-center sim-pulse"
                style={{
                  background: `linear-gradient(135deg, ${v.tone}18, transparent)`,
                  border: `1px solid ${v.tone}44`,
                  boxShadow: `0 0 40px ${v.tone}22`,
                }}
              >
                <Flame className="w-7 h-7" style={{ color: v.tone }} />
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-[0.2em] text-white/35 font-mono-data mb-1">
                  Wellbeing impact
                </div>
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-2xl md:text-3xl font-bold tracking-tight" style={{ color: v.tone }}>
                    {v.label}
                  </span>
                  <span className="text-[13px] font-mono-data tabular-nums text-white/40">
                    {score.toFixed(2)}
                  </span>
                </div>
                <p className="text-[13px] text-white/45 mt-1 max-w-xs leading-relaxed">{v.blurb}</p>
              </div>
            </div>

            {/* Content excerpt */}
            <div className="flex-1 min-w-0 lg:border-l lg:border-white/[0.06] lg:pl-8">
              <div className="text-[10px] uppercase tracking-[0.18em] text-white/35 font-mono-data mb-2">
                Content analyzed
              </div>
              <blockquote className="text-[14px] md:text-[15px] text-white/75 leading-relaxed line-clamp-3 border-l-2 border-white/30 pl-4">
                {analyzedText}
              </blockquote>
              <div className="flex flex-wrap gap-2 mt-3 text-[10px] text-white/35 font-mono-data">
                <span className="px-2 py-0.5 rounded-md ui-inset">
                  {result.raw.agents.length} agents
                </span>
                <span className="px-2 py-0.5 rounded-md ui-inset">
                  {s.waves_completed} waves
                </span>
                {result.timing?.simulate_seconds != null && (
                  <span className="px-2 py-0.5 rounded-md ui-inset">
                    {result.timing.simulate_seconds.toFixed(1)}s compute
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* KPI row */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-6 pt-6 border-t border-white/[0.06]">
            {kpis.map((k) => (
              <div key={k.label} className="sim-kpi rounded-xl p-3.5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase tracking-wider text-white/40 font-medium">
                    {k.label}
                  </span>
                  <k.icon className="w-3.5 h-3.5 opacity-40" style={{ color: k.accent }} />
                </div>
                <div
                  className={`text-xl font-bold tabular-nums tracking-tight ${k.capitalize ? "capitalize" : ""}`}
                  style={{ color: k.accent }}
                >
                  {k.value}
                </div>
                <div className="text-[10px] text-white/35 mt-0.5 truncate">{k.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {warning && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-xl ui-inset px-4 py-3 text-[12px] text-white/80 flex items-start gap-2"
        >
          <span className="text-white/50 shrink-0">⚠</span>
          {warning}
        </motion.div>
      )}

      {/* Tab navigation */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div className="flex gap-1 p-1 rounded-xl ui-inset w-fit">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={cn(
                "relative flex items-center gap-2 px-4 py-2.5 rounded-lg text-[13px] font-medium transition-all duration-200",
                tab === t.id
                  ? "bg-white text-black shadow-md"
                  : "text-white/60 hover:text-white hover:bg-white/10"
              )}
            >
              <t.icon className="w-4 h-4" />
              {t.label}
            </button>
          ))}
        </div>
        <p className="text-[12px] text-white/50 hidden sm:block">
          {tabs.find((t) => t.id === tab)?.desc}
        </p>
      </div>

      <AnimatePresence initial={false}>
        {tab === "spread" && (
          <motion.div
            key="spread"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
            className="space-y-4"
          >
            {/* ── BRAIN CENTERPIECE ── */}
            <div className="sim-panel rounded-2xl overflow-hidden relative">
              <div className="sim-scan-line rounded-2xl opacity-40" />
              <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 ui-panel-header">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-white/15 to-transparent border border-white/20 grid place-items-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-[15px] font-semibold text-white tracking-tight">
                      Neural emotional response
                    </div>
                    <div className="text-[11px] text-white/40 mt-0.5">
                      {selectedAgent ? (
                        <>
                          Individual affect ·{" "}
                          <span className="text-white/55 capitalize">
                            {(selectedPersona?.occupation || selectedAgent.archetype).replace(/_/g, " ")}
                          </span>
                        </>
                      ) : (
                        "Population brain map · empathy, connection, inspiration, curiosity, joy"
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1">
                  <span className="text-[10px] uppercase tracking-widest text-white/30 font-mono-data">
                    Peak region
                  </span>
                  <span className="px-3 py-1 rounded-full bg-white/10 border border-white/25 text-[11px] font-semibold text-white text-right max-w-[200px] leading-snug">
                    {brainDominantRegion}
                  </span>
                  <span className="text-[10px] text-white/40 capitalize">{brainDominant}</span>
                </div>
              </div>
              <div className="h-[min(560px,58vh)] min-h-[400px]">
                <BrainSimulation hero emotions={brainEmotions} dominantEmotion={brainDominant} />
              </div>
            </div>

            {/* ── CASCADE + AGENT SIDEBAR ── */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 min-h-[540px]">
              <div className="xl:col-span-8 sim-panel rounded-2xl overflow-hidden flex flex-col min-h-[480px]">
                <div className="flex items-center justify-between px-4 py-3 ui-panel-header shrink-0">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-white/10 to-transparent border border-white/15 grid place-items-center">
                      <Network className="w-4 h-4 text-white/80" />
                    </div>
                    <div>
                      <div className="text-[13px] font-semibold text-white">Emotional contagion map</div>
                      <div className="text-[10px] text-white/40 font-mono-data">
                        How affect travels person-to-person ·{" "}
                        {graphData.sampled
                          ? `${graphData.agents.length} nodes · ${graphData.total} total`
                          : `${graphData.total} exposure events`}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/[0.04] border border-white/[0.06] text-[9px] text-white/40 font-mono-data uppercase tracking-wider">
                      Scroll zoom · Drag pan
                    </div>
                    <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-white/[0.06] border border-white/15">
                      <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                      <span className="text-[10px] text-white/70 font-medium uppercase tracking-wider">
                        Live
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex-1 min-h-[420px] p-0.5">
                  <NetworkGraph
                    agents={graphData.agents}
                    shareEdges={result.raw.share_edges}
                    personas={result.raw.personas}
                    runKey={runKey}
                    onSelect={onSelect}
                    selectedExposureId={selectedAgent?.exposure_id ?? null}
                    displayTotal={graphData.total}
                    sampled={graphData.sampled}
                  />
                </div>
              </div>

              {/* Agent detail / spreaders */}
              <div className="xl:col-span-4 sim-panel rounded-2xl overflow-hidden flex flex-col min-h-[480px]">
                {selectedAgent ? (
                  <>
                    {!chatOpen && (
                      <AgentInspector
                        agent={selectedAgent}
                        persona={selectedPersona}
                        result={result}
                        onClose={() => onSelect(null)}
                        onChat={() => setChatOpen(true)}
                        chatOpen={chatOpen}
                      />
                    )}
                    {chatOpen && (
                      <PersonaChat
                        key={selectedAgent.exposure_id}
                        agent={selectedAgent}
                        persona={selectedPersona}
                        content={chatContent}
                        ollamaUp={ollamaUp}
                        onClose={() => setChatOpen(false)}
                      />
                    )}
                  </>
                ) : (
                  <div className="flex flex-col flex-1 p-5 min-h-0">
                    <div className="flex items-center gap-2 mb-1">
                      <Sparkles className="w-4 h-4 text-white/40" />
                      <h3 className="text-[14px] font-semibold text-white">Explore emotional spread</h3>
                    </div>
                    <p className="text-[12px] text-white/45 leading-relaxed">
                      Select a viewer to see how the content landed on them — their vote, emotional profile, and attention — or chat empathetically in first person.
                    </p>
                    <button
                      type="button"
                      onClick={() => setTab("analysis")}
                      className="mt-3 self-start text-[11px] text-white/60 hover:text-white flex items-center gap-1 transition"
                    >
                      View wellbeing analytics <ArrowRight className="w-3 h-3" />
                    </button>

                    <div className="mt-5 pt-4 border-t border-white/[0.06] flex-1 flex flex-col min-h-0">
                      <div className="text-[10px] uppercase tracking-[0.15em] text-white/35 font-mono-data mb-2">
                        Top emotional amplifiers
                      </div>
                      <div className="space-y-1 overflow-y-auto flex-1 -mx-1 px-1">
                        {topSpreaders.length === 0 && (
                          <p className="text-[12px] text-white/35 py-4 text-center">
                            No reshares in this run — emotional signal stayed mostly private to initial viewers.
                          </p>
                        )}
                        {topSpreaders.map((a, i) => {
                          const occ = (a.occupation || a.archetype).replace(/_/g, " ");
                          const { key: em } = topEmotion(a.emotion_probs);
                          return (
                            <button
                              key={a.exposure_id}
                              onClick={() => onSelect(a)}
                              className="w-full flex items-center gap-3 p-2.5 rounded-xl hover:bg-white/[0.04] border border-transparent hover:border-white/[0.06] text-left transition group"
                            >
                              <span className="w-8 h-8 rounded-lg bg-gradient-to-br from-white/10 to-white/[0.02] border border-white/[0.08] grid place-items-center text-[11px] font-bold text-white/70 font-mono-data shrink-0">
                                {i + 1}
                              </span>
                              <div className="min-w-0 flex-1">
                                <div className="text-[12.5px] text-white/85 capitalize truncate font-medium">
                                  {occ}
                                </div>
                                <div className="text-[10px] text-white/40 mt-0.5">
                                  → {a.shared_to_indices.length} recipients · wave {a.wave}
                                </div>
                              </div>
                              <span className="text-[10px] text-white/50 capitalize shrink-0 hidden sm:block">
                                {em}
                              </span>
                              <ArrowRight className="w-3.5 h-3.5 text-white/20 group-hover:text-white/60 transition shrink-0" />
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        {tab === "analysis" && (
          <motion.div
            key="analysis"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            <MetricsDashboard result={result} />
          </motion.div>
        )}

        {tab === "model" && (
          <motion.div
            key="model"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25 }}
          >
            <ModelArchitecture result={result} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
