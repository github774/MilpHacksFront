import { useEffect, useRef, useCallback, useTransition, useState } from "react";
import { cn } from "./lib/utils";
import {
  FileText,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Heart,
  Eye,
  RefreshCw,
  Sparkles,
  ArrowUpIcon,
  Paperclip,
  SendIcon,
  XIcon,
  LoaderIcon,
  Command,
  HelpCircle,
  RotateCcw,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import * as React from "react";
import { BrainSimulation } from "./components/BrainSimulation";

interface UseAutoResizeTextareaProps {
  minHeight: number;
  maxHeight?: number;
}

function useAutoResizeTextarea({
  minHeight,
  maxHeight,
}: UseAutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      if (reset) {
        textarea.style.height = `${minHeight}px`;
        return;
      }

      textarea.style.height = `${minHeight}px`;
      const newHeight = Math.max(
        minHeight,
        Math.min(
          textarea.scrollHeight,
          maxHeight ?? Number.POSITIVE_INFINITY
        )
      );

      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = `${minHeight}px`;
    }
  }, [minHeight]);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [adjustHeight]);

  return { textareaRef, adjustHeight };
}

interface CommandSuggestion {
  icon: React.ReactNode;
  label: string;
  description: string;
  prefix: string;
  text: string;
  groups: string[];
}

interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  containerClassName?: string;
  showRing?: boolean;
}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, containerClassName, showRing = true, ...props }, ref) => {
    const [isFocused, setIsFocused] = React.useState(false);

    return (
      <div className={cn("relative", containerClassName)}>
        <textarea
          className={cn(
            "flex min-h-[80px] w-full rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm",
            "transition-all duration-200 ease-in-out",
            "placeholder:text-white/20 text-white/90",
            "disabled:cursor-not-allowed disabled:opacity-50",
            showRing ? "focus-visible:outline-none focus-visible:ring-0 focus-visible:ring-offset-0" : "",
            className
          )}
          ref={ref}
          onFocus={(e) => {
            setIsFocused(true);
            props.onFocus?.(e);
          }}
          onBlur={(e) => {
            setIsFocused(false);
            props.onBlur?.(e);
          }}
          {...props}
        />

        {showRing && isFocused && (
          <motion.span
            className="absolute inset-0 rounded-md pointer-events-none ring-2 ring-offset-0 ring-white/30"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          />
        )}
      </div>
    );
  }
);
Textarea.displayName = "Textarea";

interface MetricItem {
  name: string;
  score: number;
  description: string;
  icon: React.ReactNode;
  color: string;
}

interface SimulationResults {
  riskScore: number;
  empathyScore: number;
  attentionScore: number;
  sentimentScore: number;
  harmScore: number;
  supportScore: number;
  saferRewrite: string;
  affectedGroups: { name: string; impact: string; severity: "high" | "medium" | "low" }[];
  predictedEmotions: { name: string; percentage: number; color: string }[];
}

export default function App() {
  const [value, setValue] = useState("");
  const [selectedGroups, setSelectedGroups] = useState<string[]>(["teens", "anxious"]);
  const [isTyping, setIsTyping] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [activeSuggestion, setActiveSuggestion] = useState<number>(-1);
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [recentCommand, setRecentCommand] = useState<string | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 60,
    maxHeight: 200,
  });
  const [inputFocused, setInputFocused] = useState(false);
  const commandPaletteRef = useRef<HTMLDivElement>(null);

  // App Layout States
  const [showDashboard, setShowDashboard] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [results, setResults] = useState<SimulationResults | null>(null);

  const commandSuggestions: CommandSuggestion[] = [
    {
      icon: <Sparkles className="w-4 h-4 text-neutral-400" />,
      label: "TikTok Preset",
      description: "Self-deprecating video caption",
      prefix: "/tiktok",
      text: "I look so ugly in this video, I should literally delete myself from existence. Nobody would notice anyway.",
      groups: ["teens", "anxious"],
    },
    {
      icon: <TrendingUp className="w-4 h-4 text-neutral-400" />,
      label: "Grindset Tweet",
      description: "Toxic productivity hustle content",
      prefix: "/tweet",
      text: "If you aren't grinding 24/7 you are a failure and deserve to fail. Stop whining, sleep when you are dead.",
      groups: ["teens", "anxious", "general"],
    },
    {
      icon: <Heart className="w-4 h-4 text-neutral-400" />,
      label: "Caregiver Ad",
      description: "Sensationalized mental health notice",
      prefix: "/caregiver",
      text: "Warning: Suicide rates are skyrocketing. Caregivers are completely failing to notice the signs until it's too late.",
      groups: ["caregivers", "general"],
    },
    {
      icon: <FileText className="w-4 h-4 text-neutral-400" />,
      label: "Sensational Headline",
      description: "Sensational news headline",
      prefix: "/headline",
      text: "New study reveals social media is permanently destroying teens' brains. There is no hope left for this generation.",
      groups: ["teens", "caregivers", "general"],
    },
  ];

  useEffect(() => {
    if (value.startsWith("/") && !value.includes(" ")) {
      setShowCommandPalette(true);

      const matchingSuggestionIndex = commandSuggestions.findIndex((cmd) =>
        cmd.prefix.startsWith(value)
      );

      if (matchingSuggestionIndex >= 0) {
        setActiveSuggestion(matchingSuggestionIndex);
      } else {
        setActiveSuggestion(-1);
      }
    } else {
      setShowCommandPalette(false);
    }
  }, [value]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      const commandButton = document.querySelector("[data-command-button]");

      if (
        commandPaletteRef.current &&
        !commandPaletteRef.current.contains(target) &&
        !commandButton?.contains(target)
      ) {
        setShowCommandPalette(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showCommandPalette) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActiveSuggestion((prev) =>
          prev < commandSuggestions.length - 1 ? prev + 1 : 0
        );
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActiveSuggestion((prev) =>
          prev > 0 ? prev - 1 : commandSuggestions.length - 1
        );
      } else if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        if (activeSuggestion >= 0) {
          applyPreset(commandSuggestions[activeSuggestion]);
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        setShowCommandPalette(false);
      }
    } else if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (value.trim()) {
        runSimulation();
      }
    }
  };

  const applyPreset = (preset: CommandSuggestion) => {
    setValue(preset.text);
    setSelectedGroups(preset.groups);
    setShowCommandPalette(false);
    setRecentCommand(preset.label);
    setTimeout(() => {
      setRecentCommand(null);
      adjustHeight();
    }, 2000);
  };

  // Run Simulation analysis calculations
  const runSimulation = () => {
    if (!value.trim()) return;

    setIsSimulating(true);
    setShowDashboard(true);

    // Dynamic generation based on keyword checks
    const textLower = value.toLowerCase();
    
    // Heuristics for calculations
    const isToxicProductivity = textLower.includes("grind") || textLower.includes("fail") || textLower.includes("deserve");
    const isSelfHarm = textLower.includes("ugly") || textLower.includes("delete myself") || textLower.includes("notice") || textLower.includes("die") || textLower.includes("suicide");
    const isSensational = textLower.includes("destroying") || textLower.includes("no hope") || textLower.includes("skyrocketing") || textLower.includes("failing");
    
    let calculatedRisk = 30;
    if (isSelfHarm) calculatedRisk = 92;
    else if (isToxicProductivity) calculatedRisk = 78;
    else if (isSensational) calculatedRisk = 65;

    // Adjust risk slightly based on audience count
    calculatedRisk += selectedGroups.length * 2;
    calculatedRisk = Math.max(10, Math.min(calculatedRisk, 98));

    // Wait 3.5s for particle simulation to run
    setTimeout(() => {
      let empathy = 100 - calculatedRisk + 10;
      let attention = isToxicProductivity ? 82 : (isSelfHarm ? 94 : (isSensational ? 76 : 50));
      let spread = calculatedRisk + 5;
      let harm = calculatedRisk;
      let support = 100 - calculatedRisk;

      empathy = Math.max(5, Math.min(empathy, 95));
      attention = Math.max(15, Math.min(attention, 98));
      spread = Math.max(10, Math.min(spread, 99));
      harm = Math.max(5, Math.min(harm, 98));
      support = Math.max(5, Math.min(support, 95));

      // Rewrite options
      let rewrite = "Let's share this in a way that respects mental health boundaries.";
      if (isSelfHarm) {
        rewrite = "Feeling really overwhelmed today and struggling with self-image. Taking a break from social media to ground myself. Sending love to anyone else having a hard day.";
      } else if (isToxicProductivity) {
        rewrite = "Consistent hard work can yield great results, but sustainable success requires rest and self-care. Take care of your mental health first.";
      } else if (isSensational) {
        rewrite = "Recent studies open up important discussions about social media's role in adolescent brain development, highlighting areas where caregivers can offer support.";
      } else {
        rewrite = "Sharing some perspectives on personal wellness and digital usage. Let's build supportive discussions.";
      }

      // Personality group impacts
      const groups = [
        { name: "Teens", impact: isSelfHarm ? "High emotional distress risk" : (isToxicProductivity ? "Triggers validation anxiety" : "Mild engagement"), severity: ((isSelfHarm || isToxicProductivity) ? "high" : "medium") as "high" | "medium" | "low" },
        { name: "Anxious Users", impact: isSelfHarm ? "Triggers rumination loop" : (isSensational ? "Amplifies general hopelessness" : "Moderate stress increase"), severity: ((isSelfHarm || isSensational) ? "high" : "medium") as "high" | "medium" | "low" },
        { name: "Caregivers", impact: isSensational ? "Promotes parental burnout/anxiety" : "Induces protective concerns", severity: (isSensational ? "high" : "low") as "high" | "medium" | "low" },
        { name: "General Public", impact: "Passive content saturation and spread", severity: "low" as "high" | "medium" | "low" }
      ].filter(g => selectedGroups.includes(g.name.toLowerCase().replace(" ", "")));

      // Emotions chart
      const emotions = [
        { name: "Anxiety", percentage: Math.round(calculatedRisk * 0.45), color: "#a3a3a3" },
        { name: "Hopelessness", percentage: Math.round(calculatedRisk * 0.35), color: "#525252" },
        { name: "Validation-Seeking", percentage: Math.round(attention * 0.4), color: "#d4d4d4" },
        { name: "Empathy / Care", percentage: Math.round(support * 0.8), color: "#e5e5e5" }
      ].sort((a, b) => b.percentage - a.percentage);

      setResults({
        riskScore: calculatedRisk,
        empathyScore: empathy,
        attentionScore: attention,
        sentimentScore: spread,
        harmScore: harm,
        supportScore: support,
        saferRewrite: rewrite,
        affectedGroups: groups,
        predictedEmotions: emotions
      });
      setIsSimulating(false);
    }, 3200);
  };

  const handleSendMessage = () => {
    runSimulation();
  };

  const handleGroupToggle = (group: string) => {
    setSelectedGroups((prev) =>
      prev.includes(group) ? prev.filter((g) => g !== group) : [...prev, group]
    );
  };

  const adoptRewrite = () => {
    if (!results) return;
    setValue(results.saferRewrite);
    // Auto-simulates rewrite
    setTimeout(() => {
      // Re-trigger simulation with lower scores
      setIsSimulating(true);
      setTimeout(() => {
        setResults((prev) => {
          if (!prev) return null;
          return {
            ...prev,
            riskScore: 18,
            empathyScore: 88,
            attentionScore: 54,
            sentimentScore: 28,
            harmScore: 12,
            supportScore: 85,
            affectedGroups: prev.affectedGroups.map(g => ({ ...g, impact: "Supportive/Positive framework adopted", severity: "low" })),
            predictedEmotions: [
              { name: "Empathy / Care", percentage: 80, color: "#e5e5e5" },
              { name: "Validation-Seeking", percentage: 30, color: "#d4d4d4" },
              { name: "Anxiety", percentage: 10, color: "#a3a3a3" },
              { name: "Hopelessness", percentage: 5, color: "#525252" }
            ]
          };
        });
        setIsSimulating(false);
      }, 2000);
    }, 150);
  };

  const resetAll = () => {
    setValue("");
    setSelectedGroups(["teens", "anxious"]);
    setResults(null);
    setShowDashboard(false);
    setIsSimulating(false);
  };

  return (
    <div className="min-h-screen bg-[#030304] text-white flex flex-col relative overflow-hidden font-sans">
      {/* Dynamic atmospheric grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none opacity-40" />

      {/* Floating neon mesh background */}
      <div className="absolute inset-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute top-[-20%] left-[10%] w-[45rem] h-[45rem] bg-white/[0.02] rounded-full mix-blend-screen filter blur-[150px] animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[10%] w-[40rem] h-[40rem] bg-neutral-800/[0.03] rounded-full mix-blend-screen filter blur-[150px] animate-pulse-slow delay-1000" />
      </div>

      {/* Main Header */}
      <header className="relative z-30 px-6 py-4 border-b border-white/[0.05] backdrop-blur-md flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="relative">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-white to-neutral-400 flex items-center justify-center shadow-lg shadow-white/5">
              <span className="font-extrabold text-sm tracking-tighter text-black">SM</span>
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white rounded-full border border-black animate-ping" />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white rounded-full border border-black" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              SwarmMind
            </h1>
            <p className="text-[10px] text-white/40 tracking-wider uppercase font-semibold">
              Mental Health Spread Engine
            </p>
          </div>
        </div>

        {showDashboard && (
          <div className="flex items-center gap-2">
            <button
              onClick={resetAll}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] border border-white/10 text-xs font-medium text-white/70 hover:text-white transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Engine</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Area */}
      <main className="flex-1 w-full max-w-[1400px] mx-auto p-4 md:p-6 relative z-10 flex flex-col justify-center">
        <AnimatePresence mode="wait">
          {!showDashboard ? (
            /* ==================== LANDING / INPUT VIEW ==================== */
            <motion.div
              key="landing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="w-full max-w-2xl mx-auto space-y-10 py-10"
            >
              <div className="text-center space-y-3">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1, duration: 0.5 }}
                  className="inline-block"
                >
                  <h2 className="text-4xl md:text-5xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/40 pb-2">
                    SwarmMind
                  </h2>
                  <p className="text-base text-neutral-300 font-medium">
                    Real-time mental health impact simulation for digital content.
                  </p>
                  <motion.div
                    className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent mt-3"
                    initial={{ width: 0, opacity: 0 }}
                    animate={{ width: "100%", opacity: 1 }}
                    transition={{ delay: 0.4, duration: 0.8 }}
                  />
                </motion.div>
                <p className="text-xs text-white/40 max-w-md mx-auto">
                  Paste content, select vulnerable audiences, and simulate cognitive spread across brain pathways before you publish.
                </p>
              </div>

              {/* Chat Input Container */}
              <div className="relative backdrop-blur-2xl bg-white/[0.02] rounded-2xl border border-white/[0.06] shadow-2xl shadow-black/80">
                {/* Command palette */}
                <AnimatePresence>
                  {showCommandPalette && (
                    <motion.div
                      ref={commandPaletteRef}
                      className="absolute left-4 right-4 bottom-full mb-2 backdrop-blur-xl bg-black/95 rounded-lg z-50 shadow-2xl border border-white/10 overflow-hidden"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 5 }}
                      transition={{ duration: 0.15 }}
                    >
                      <div className="py-1.5 bg-black/95">
                        <div className="px-3 py-1.5 text-[10px] font-bold text-white/30 uppercase tracking-wider border-b border-white/5 mb-1">
                          Presets Command Palette
                        </div>
                        {commandSuggestions.map((suggestion, index) => (
                          <motion.div
                            key={suggestion.prefix}
                            className={cn(
                              "flex items-center justify-between px-3 py-2 text-xs transition-colors cursor-pointer",
                              activeSuggestion === index
                                ? "bg-white/10 text-white"
                                : "text-white/70 hover:bg-white/5"
                            )}
                            onClick={() => applyPreset(suggestion)}
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: index * 0.03 }}
                          >
                            <div className="flex items-center gap-2.5">
                              <div className="w-5 h-5 flex items-center justify-center">
                                {suggestion.icon}
                              </div>
                              <div>
                                <span className="font-semibold">{suggestion.label}</span>
                                <span className="text-white/40 text-[10px] ml-2 font-light">
                                  {suggestion.description}
                                </span>
                              </div>
                            </div>
                             <div className="text-white/60 font-mono text-[10px] bg-white/5 border border-white/10 px-1.5 py-0.5 rounded">
                              {suggestion.prefix}
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Input Text Area */}
                <div className="p-4">
                  <Textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(e) => {
                      setValue(e.target.value);
                      adjustHeight();
                    }}
                    onKeyDown={handleKeyDown}
                    onFocus={() => setInputFocused(true)}
                    onBlur={() => setInputFocused(false)}
                    placeholder="Paste digital content, or type / to see presets..."
                    containerClassName="w-full"
                    className="w-full px-4 py-3 resize-none bg-transparent border-none text-white/90 text-sm focus:outline-none placeholder:text-white/20 min-h-[70px] overflow-hidden"
                    showRing={false}
                  />
                </div>

                {/* Target Archetype Checkboxes inside input box */}
                 <div className="px-4 pb-3 border-b border-white/[0.04] flex flex-col gap-2">
                  <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                    Simulated Audience Archetypes
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { key: "teens", label: "Teens" },
                      { key: "anxious", label: "Anxious Users" },
                      { key: "caregivers", label: "Caregivers" },
                      { key: "general", label: "General Public" },
                    ].map((g) => {
                      const isActive = selectedGroups.includes(g.key);
                      return (
                        <button
                          key={g.key}
                          onClick={() => handleGroupToggle(g.key)}
                          className={cn(
                            "px-3 py-1 rounded-full text-xs font-medium border transition-all flex items-center gap-1.5",
                            isActive ? "bg-white border-white text-black" : "bg-white/[0.01] border-white/10 text-white/50 hover:text-white/80"
                          )}
                        >
                          <span className={cn("w-1.5 h-1.5 rounded-full", isActive ? "bg-black" : "bg-white/20")} />
                          {g.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Footer buttons */}
                <div className="p-4 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <button
                      type="button"
                      data-command-button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowCommandPalette((prev) => !prev);
                      }}
                      className={cn(
                        "p-2 text-white/40 hover:text-white/95 rounded-lg transition-colors bg-white/[0.02] border border-white/5 hover:border-white/10 flex items-center gap-1.5 text-xs",
                        showCommandPalette && "bg-white/10 text-white/95 border-white/20"
                      )}
                    >
                      <Command className="w-3.5 h-3.5" />
                      <span>Presets</span>
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={runSimulation}
                    disabled={!value.trim()}
                    className={cn(
                      "px-5 py-2.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 tracking-wide border",
                      value.trim()
                        ? "bg-white text-black border-transparent hover:bg-neutral-100 active:scale-98"
                        : "bg-white/[0.02] border-white/5 text-white/30 cursor-not-allowed"
                    )}
                  >
                    <span>Run Simulation</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Suggestions row */}
              <div className="space-y-2">
                <div className="text-center text-[10px] font-bold text-white/30 uppercase tracking-widest">
                  Quick Preset Simulations
                </div>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  {commandSuggestions.map((suggestion, index) => (
                    <button
                      key={suggestion.prefix}
                      onClick={() => applyPreset(suggestion)}
                      className="flex items-center gap-2 px-3.5 py-2 bg-white/[0.02] hover:bg-white/[0.06] border border-white/[0.04] hover:border-white/10 rounded-xl text-xs text-white/60 hover:text-white transition-all"
                    >
                      {suggestion.icon}
                      <span>{suggestion.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            /* ==================== 3-PANEL DASHBOARD VIEW ==================== */
            <motion.div
              key="dashboard"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-6 w-full items-stretch"
            >
              {/* LEFT PANEL: Live Controls */}
              <div className="lg:col-span-3 flex flex-col gap-5">
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <div className="flex items-center justify-between border-b border-white/5 pb-3">
                     <span className="text-xs font-bold text-white uppercase tracking-wider">
                      Simulation Inputs
                    </span>
                    {isSimulating && (
                      <LoaderIcon className="w-4 h-4 text-white animate-spin" />
                    )}
                  </div>

                  {/* Editable Input */}
                  <div className="space-y-2">
                    <label className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                      Content Text
                    </label>
                    <textarea
                      value={value}
                      onChange={(e) => setValue(e.target.value)}
                      placeholder="Paste text to analyze..."
                      className="w-full min-h-[120px] bg-black/40 border border-white/5 hover:border-white/10 focus:border-violet-500/50 rounded-xl p-3 text-xs text-white/80 placeholder:text-white/20 focus:outline-none transition-all resize-none"
                    />
                  </div>

                  {/* Archetype checkboxes */}
                  <div className="space-y-2.5">
                    <label className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                      Audience Toggles
                    </label>
                    <div className="flex flex-col gap-2">
                      {[
                        { key: "teens", label: "Teens", activeBg: "bg-white/15 border-white/40 text-white" },
                        { key: "anxious", label: "Anxious Users", activeBg: "bg-neutral-200/15 border-neutral-300/40 text-neutral-200" },
                        { key: "caregivers", label: "Caregivers", activeBg: "bg-neutral-400/15 border-neutral-400/40 text-neutral-300" },
                        { key: "general", label: "General Public", activeBg: "bg-neutral-600/15 border-neutral-500/40 text-neutral-400" }
                      ].map((g) => {
                        const isActive = selectedGroups.includes(g.key);
                        return (
                          <button
                            key={g.key}
                            onClick={() => handleGroupToggle(g.key)}
                            className={cn(
                              "w-full text-left px-3 py-2 rounded-xl text-xs font-medium border transition-all flex items-center justify-between",
                              isActive ? g.activeBg : "bg-white/[0.01] border-white/5 text-white/40 hover:text-white/70"
                            )}
                          >
                            <div className="flex items-center gap-2">
                              <span className={cn("w-1.5 h-1.5 rounded-full", isActive ? "bg-current animate-pulse" : "bg-white/25")} />
                              <span>{g.label}</span>
                            </div>
                            {isActive && <span className="text-[8px] bg-white/10 px-1.5 py-0.5 rounded font-bold uppercase tracking-wide">Active</span>}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <button
                    onClick={runSimulation}
                    disabled={isSimulating || !value.trim()}
                    className={cn(
                      "w-full py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 border tracking-wide",
                      value.trim()
                        ? "bg-white text-black border-transparent hover:bg-white/90 active:scale-98"
                        : "bg-white/[0.02] border-white/5 text-white/30 cursor-not-allowed"
                    )}
                  >
                    <RefreshCw className={cn("w-3.5 h-3.5", isSimulating && "animate-spin")} />
                    <span>{isSimulating ? "Analyzing..." : "Re-Simulate"}</span>
                  </button>
                </div>

                {/* Info Tip Card */}
                <div className="glass-panel p-4 rounded-2xl text-[11px] text-white/40 flex gap-2.5 items-start">
                  <HelpCircle className="w-4 h-4 text-white/60 shrink-0 mt-0.5" />
                  <p>
                    The SwarmMind model runs local NLP checks on the text and estimates impact metrics, displaying node network flows mapping path transmission.
                  </p>
                </div>
              </div>

              {/* MIDDLE PANEL: Network Graph */}
              <div className="lg:col-span-5 flex flex-col gap-4 min-h-[400px] lg:min-h-0">
                <div className="flex-1 relative flex flex-col">
                  {/* Top graph bar header */}
                  <div className="absolute top-4 left-4 right-4 z-20 flex justify-between items-center pointer-events-none">
                    <div className="bg-black/70 backdrop-blur-md px-3 py-1.5 rounded-lg border border-white/10 flex items-center gap-2 text-xs">
                      <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                      <span className="font-semibold text-white/90">Brain Pathway Live Feed</span>
                    </div>
                    {isSimulating && (
                      <div className="bg-neutral-900/80 border border-white/10 text-white/90 backdrop-blur-md px-3 py-1.5 rounded-lg text-xs font-semibold animate-pulse">
                        Transmitting Swarm Signal...
                      </div>
                    )}
                  </div>

                  {/* The interactive Canvas container */}
                  <BrainSimulation
                    isActive={isSimulating}
                    selectedGroups={selectedGroups}
                    riskScore={results?.riskScore || 50}
                  />
                </div>
              </div>

              {/* RIGHT PANEL: Metrics & Suggested Rewrite */}
              <div className="lg:col-span-4 flex flex-col gap-5">
                {isSimulating ? (
                  /* Loading placeholders for metrics */
                  <div className="glass-panel p-6 rounded-2xl flex-1 flex flex-col items-center justify-center space-y-4 min-h-[300px]">
                    <LoaderIcon className="w-8 h-8 text-white/85 animate-spin" />
                    <div className="text-center space-y-1">
                      <div className="text-xs font-bold text-white/80 uppercase tracking-widest">
                        Running Swarm Analysis
                      </div>
                      <div className="text-[10px] text-white/40">
                        Scanning words, mapping triggers, computing spread path...
                      </div>
                    </div>
                  </div>
                ) : results ? (
                  /* Real Metrics Display */
                  <div className="space-y-5 flex-1 flex flex-col">
                    {/* Overall Risk Score Indicator */}
                    <div className="glass-panel p-5 rounded-2xl flex items-center justify-between relative overflow-hidden shimmer-effect">
                      <div className="space-y-1 relative z-10">
                        <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                          Overall Mental Health Risk
                        </span>
                        <h3 className="text-xl font-bold tracking-tight uppercase">
                          {results.riskScore >= 75 ? "High Severity" : (results.riskScore >= 45 ? "Moderate Severity" : "Safe Exposure")}
                        </h3>
                        <p className="text-xs text-white/50 max-w-[170px]">
                          {results.riskScore >= 75 ? "High risk of inducing emotional spiral or self-harm triggers." : (results.riskScore >= 45 ? "Contains potential triggers; moderate warning suggested." : "Low harm probability. Supportive frameworks active.")}
                        </p>
                      </div>

                      {/* Circular Gauge */}
                      <div className="relative w-24 h-24 flex items-center justify-center shrink-0">
                        <svg className="w-full h-full">
                          <circle
                            cx="48"
                            cy="48"
                            r="38"
                            className="stroke-white/5 fill-transparent"
                            strokeWidth="6"
                          />
                          <circle
                            cx="48"
                            cy="48"
                            r="38"
                            className="progress-ring__circle fill-transparent"
                            strokeWidth="7"
                            strokeDasharray={`${2 * Math.PI * 38}`}
                            strokeDashoffset={`${2 * Math.PI * 38 * (1 - results.riskScore / 100)}`}
                            strokeLinecap="round"
                            stroke={results.riskScore >= 75 ? "#ffffff" : (results.riskScore >= 45 ? "#a3a3a3" : "#525252")}
                          />
                        </svg>
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                          <span className="text-xl font-bold tracking-tighter">
                            {results.riskScore}%
                          </span>
                          <span className="text-[8px] text-white/40 font-bold uppercase tracking-wider">
                            Risk Index
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Interactive Metrics Cards Scrollable */}
                    <div className="space-y-3 max-h-[360px] overflow-y-auto pr-1">
                      {[
                        {
                          name: "Empathy Impact",
                          score: results.empathyScore,
                          description: results.riskScore >= 75 ? "Critical drop. Promotes disconnection." : "Positive. Encourages supportive empathy.",
                          icon: <Heart className="w-4 h-4" />,
                          color: "text-white bg-white/10",
                          barColor: "bg-white"
                        },
                        {
                          name: "Attention Retention",
                          score: results.attentionScore,
                          description: results.riskScore >= 75 ? "Sensational attention lock." : "Healthy moderate interest levels.",
                          icon: <Eye className="w-4 h-4" />,
                          color: "text-neutral-200 bg-white/5",
                          barColor: "bg-neutral-200"
                        },
                        {
                          name: "Sentiment Spread",
                          score: results.sentimentScore,
                          description: results.riskScore >= 75 ? "Rapid negative replication potential." : "Slow, controlled distribution.",
                          icon: <TrendingUp className="w-4 h-4" />,
                          color: "text-neutral-300 bg-white/5",
                          barColor: "bg-neutral-300"
                        },
                        {
                          name: "Harm Amplification",
                          score: results.harmScore,
                          description: results.riskScore >= 75 ? "High risk of harmful echo effects." : "Low amplification likelihood.",
                          icon: <AlertTriangle className="w-4 h-4" />,
                          color: "text-neutral-400 bg-white/5",
                          barColor: "bg-neutral-400"
                        },
                        {
                          name: "Supportive Probability",
                          score: results.supportScore,
                          description: results.riskScore >= 75 ? "Extremely low comfort probability." : "High possibility of constructive care.",
                          icon: <Sparkles className="w-4 h-4" />,
                          color: "text-neutral-100 bg-white/10",
                          barColor: "bg-neutral-100"
                        }
                      ].map((card) => (
                        <div key={card.name} className="glass-panel p-3.5 rounded-xl space-y-2.5">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <div className={cn("p-1.5 rounded-lg", card.color)}>
                                {card.icon}
                              </div>
                              <span className="text-xs font-semibold text-white/90">{card.name}</span>
                            </div>
                            <span className="text-xs font-bold">{card.score}%</span>
                          </div>
                          
                          {/* Dwell bar */}
                          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                            <div className={cn("h-full rounded-full", card.barColor)} style={{ width: `${card.score}%` }} />
                          </div>
                          
                          <p className="text-[10px] text-white/50">{card.description}</p>
                        </div>
                      ))}
                    </div>

                    {/* SUGGESTED SAFER REWRITE CARD */}
                    <div className="glass-panel p-5 rounded-2xl border border-white/10 bg-white/[0.02] flex-1 flex flex-col justify-between space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between border-b border-white/5 pb-2">
                          <div className="flex items-center gap-2">
                            <Sparkles className="w-4 h-4 text-white" />
                            <span className="text-xs font-bold text-neutral-300 uppercase tracking-wider">
                              Suggested Safer Rewrite
                            </span>
                          </div>
                          <span className="text-[9px] bg-white/15 text-white px-2 py-0.5 rounded-full font-bold">
                            Safety Optimized
                          </span>
                        </div>
                        <p className="text-xs italic text-white/85 leading-relaxed">
                          "{results.saferRewrite}"
                        </p>
                      </div>

                      <button
                        onClick={adoptRewrite}
                        className="w-full py-2 rounded-xl text-xs font-bold bg-white hover:bg-neutral-200 text-black shadow-lg shadow-white/5 transition-all flex items-center justify-center gap-2 active:scale-98"
                      >
                        <span>Adopt Safer Rewrite</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Initial landing panel right side (empty/inactive) */
                  <div className="glass-panel p-6 rounded-2xl flex-1 flex flex-col items-center justify-center text-center text-white/30 border border-dashed border-white/10 min-h-[300px]">
                    <FileText className="w-8 h-8 mb-2 text-white/20" />
                    <p className="text-xs font-semibold uppercase tracking-wider mb-1">Results Inactive</p>
                    <p className="text-[10px] max-w-[180px]">Run a content simulation on the left to review metrics.</p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer credit */}
      <footer className="py-4 text-center text-[10px] text-white/20 relative z-30 border-t border-white/[0.03] backdrop-blur-md">
        © 2026 SwarmMind Mental Health Simulation Lab. All rights reserved.
      </footer>
    </div>
  );
}
