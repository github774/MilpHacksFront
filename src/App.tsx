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
  MessageCircle,
  Users,
  Activity,
  Shield,
  Brain,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import * as React from "react";
import { BrainSimulation } from "./components/BrainSimulation";
import { simulateContent, type ApiSimulationResult } from "./api";

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

  // Navigation & Region Inspector States
  const [activeTab, setActiveTab] = useState<"workspace" | "analytics">("workspace");
  const [selectedNode, setSelectedNode] = useState<{ id: number; label: string; region: string } | null>(null);

  // Interactive Analytics States
  const [analyticsFilter, setAnalyticsFilter] = useState<"all" | "teens" | "anxious" | "caregivers" | "general">("all");
  const [activeLineIndex, setActiveLineIndex] = useState<number | null>(null);
  const [activeDonutIndex, setActiveDonutIndex] = useState<number | null>(null);
  const [selectedScatterId, setSelectedScatterId] = useState<number | null>(null);

  // Chat-with-Persona State
  const [chatPersona, setChatPersona] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<{ role: 'persona' | 'user'; text: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

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

  // File Upload State
  const [mediaFile, setMediaFile] = useState<{
    name: string;
    type: string;
    base64: string;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setMediaFile({
        name: file.name,
        type: file.type,
        base64: reader.result as string,
      });
    };
    reader.readAsDataURL(file);
  };

  // App Layout States
  const [showDashboard, setShowDashboard] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [results, setResults] = useState<SimulationResults | null>(null);
  const [rawSimData, setRawSimData] = useState<ApiSimulationResult["rawSimulation"] | null>(null);

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


  // Run Simulation analysis calculations calling Python backend with client-side fallback
  const runSimulation = async () => {
    if (!value.trim() && !mediaFile) return;

    setIsSimulating(true);
    setShowDashboard(true);
    setRawSimData(null); // Reset raw simulation data

    try {
      // Call real backend simulation API
      const apiRes = await simulateContent(value, selectedGroups, mediaFile);
      
      // If the backend transcribed or analyzed visual content, update the textarea content
      if (apiRes.extractedTranscript) {
        setValue(apiRes.extractedTranscript);
      }
      
      // Delay results display slightly to let the brain particle flow animation play
      setTimeout(() => {
        setResults({
          riskScore: apiRes.riskScore,
          empathyScore: apiRes.empathyScore,
          attentionScore: apiRes.attentionScore,
          sentimentScore: apiRes.sentimentScore,
          harmScore: apiRes.harmScore,
          supportScore: apiRes.supportScore,
          saferRewrite: apiRes.saferRewrite,
          affectedGroups: apiRes.affectedGroups,
          predictedEmotions: apiRes.predictedEmotions
        });
        setRawSimData(apiRes.rawSimulation);
        setIsSimulating(false);
      }, 3200);

    } catch (error) {
      console.warn("Backend server not running or error encountered. Falling back to local heuristic calculations...", error);
      
      // Graceful fallback to client-side heuristics
      const textLower = value.toLowerCase();
      const isToxicProductivity = textLower.includes("grind") || textLower.includes("fail") || textLower.includes("deserve");
      const isSelfHarm = textLower.includes("ugly") || textLower.includes("delete myself") || textLower.includes("notice") || textLower.includes("die") || textLower.includes("suicide");
      const isSensational = textLower.includes("destroying") || textLower.includes("no hope") || textLower.includes("skyrocketing") || textLower.includes("failing");
      
      let calculatedRisk = 30;
      if (isSelfHarm) calculatedRisk = 92;
      else if (isToxicProductivity) calculatedRisk = 78;
      else if (isSensational) calculatedRisk = 65;

      calculatedRisk += selectedGroups.length * 2;
      calculatedRisk = Math.max(10, Math.min(calculatedRisk, 98));

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

        const groups = [
          { name: "Teens", impact: isSelfHarm ? "High emotional distress risk" : (isToxicProductivity ? "Triggers validation anxiety" : "Mild engagement"), severity: ((isSelfHarm || isToxicProductivity) ? "high" : "medium") as "high" | "medium" | "low" },
          { name: "Anxious Users", impact: isSelfHarm ? "Triggers rumination loop" : (isSensational ? "Amplifies general hopelessness" : "Moderate stress increase"), severity: ((isSelfHarm || isSensational) ? "high" : "medium") as "high" | "medium" | "low" },
          { name: "Caregivers", impact: isSensational ? "Promotes parental burnout/anxiety" : "Induces protective concerns", severity: (isSensational ? "high" : "low") as "high" | "medium" | "low" },
          { name: "General Public", impact: "Passive content saturation and spread", severity: "low" as "high" | "medium" | "low" }
        ].filter(g => selectedGroups.includes(g.name.toLowerCase().replace(" ", "")));

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
    }
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
    setChatPersona(null);
    setChatMessages([]);
    setChatInput("");
  };

  // ============ PERSONA CHAT SYSTEM ============
  const personaProfiles: Record<string, { name: string; emoji: string; region: string; description: string }> = {
    teens: { name: "Teen Persona", emoji: "\uD83D\uDC66", region: "Frontal Lobe", description: "Digital native, validation-seeking, peer-influenced" },
    anxious: { name: "Anxious User", emoji: "\uD83D\uDE30", region: "Temporal Lobe", description: "Hypervigilant, rumination-prone, threat-sensitive" },
    caregivers: { name: "Caregiver", emoji: "\uD83E\uDD32", region: "Parietal Lobe", description: "Protective instinct, empathy-driven, burnout-vulnerable" },
    general: { name: "General Public", emoji: "\uD83D\uDC65", region: "Occipital Lobe", description: "Passive consumer, echo chamber participant" },
  };

  const personaResponses: Record<string, { high: string[]; medium: string[]; low: string[] }> = {
    teens: {
      high: [
        "This makes me feel like I\u2019m not good enough... everyone else seems to have it together.",
        "I keep scrolling and it\u2019s making me feel worse, but I can\u2019t stop.",
        "My friends are all sharing this. If I don\u2019t react, I\u2019ll be left out.",
        "I want to say something but I\u2019m scared people will judge me.",
        "This kind of content makes me compare myself constantly.",
        "I screenshot this to talk about in my group chat... it\u2019s triggering but relatable.",
      ],
      medium: [
        "It\u2019s kind of dramatic but I get why people are sharing it.",
        "Not sure how to feel about this. It\u2019s everywhere though.",
        "I showed this to my friend and we had a real talk about it.",
        "Content like this makes me think... but also stresses me out a little.",
      ],
      low: [
        "This seems balanced. I appreciate when people share thoughtfully.",
        "I\u2019d share this. It feels genuine and not manipulative.",
        "This actually made me feel better about reaching out for help.",
      ],
    },
    anxious: {
      high: [
        "My heart is racing just reading this. I can\u2019t stop thinking about it.",
        "What if this happens to me? I feel the spiral starting...",
        "I need to close the app but the worry follows me everywhere.",
        "This confirms my worst fears. I feel paralyzed.",
        "I\u2019ve been doomscrolling for hours now. Content like this keeps me trapped.",
        "I want to reach out for help but I feel like a burden.",
      ],
      medium: [
        "This is unsettling. I\u2019m trying to ground myself but it\u2019s hard.",
        "I notice my anxiety spiking. Taking deep breaths.",
        "I wish people would add content warnings before posting things like this.",
      ],
      low: [
        "This is reassuring. It reminds me that help is available.",
        "I feel calmer reading this. Supportive content actually helps.",
        "Thank you for framing this compassionately. It matters more than people realize.",
      ],
    },
    caregivers: {
      high: [
        "How do I protect my kids from seeing this? It\u2019s everywhere.",
        "This makes me feel like I\u2019m failing as a parent. Am I doing enough?",
        "I\u2019m exhausted from constantly monitoring what they\u2019re exposed to online.",
        "Reading this triggers my own anxiety about their safety.",
        "I want to have a conversation about this but don\u2019t know where to start.",
      ],
      medium: [
        "I should probably discuss this with my family. It\u2019s a teachable moment.",
        "Content like this reminds me to check in more often.",
        "I\u2019m saving this to review later when I have the energy.",
      ],
      low: [
        "This is the kind of content that opens healthy dialogue.",
        "I appreciate seeing responsible messaging. It gives me hope.",
        "I\u2019d share this with other parents. It\u2019s supportive without being scary.",
      ],
    },
    general: {
      high: [
        "I scrolled past this at first but it stuck with me.",
        "Not sure if sharing this helps or makes things worse.",
        "This feels manipulative. But I can\u2019t look away.",
        "I liked it without thinking. Now I\u2019m questioning if that was right.",
        "The algorithm keeps showing me more of this. It\u2019s a loop.",
      ],
      medium: [
        "Interesting perspective. I might share this with context.",
        "I\u2019ve seen a lot of takes on this. Hard to know what\u2019s real.",
        "Content overload. I\u2019m becoming numb to these posts.",
      ],
      low: [
        "This is thoughtful. More people should post like this.",
        "Refreshing to see nuanced content instead of outrage bait.",
        "I\u2019d engage with this. It feels authentic.",
      ],
    },
  };

  const getPersonaOpeningMessage = (persona: string): string => {
    const riskLevel = results ? (results.riskScore >= 75 ? "high" : results.riskScore >= 45 ? "medium" : "low") : "medium";
    const pool = personaResponses[persona]?.[riskLevel] || ["I\u2019m processing this content..."];
    return pool[0];
  };

  const openPersonaChat = (persona: string) => {
    setChatPersona(persona);
    const opening = getPersonaOpeningMessage(persona);
    setChatMessages([{ role: 'persona', text: opening }]);
    setChatInput("");
  };

  const sendChatMessage = () => {
    if (!chatInput.trim() || !chatPersona) return;
    const userMsg = chatInput.trim();
    setChatInput("");

    setChatMessages(prev => [...prev, { role: 'user', text: userMsg }]);

    // Auto-respond after a short delay
    setTimeout(() => {
      const riskLevel = results ? (results.riskScore >= 75 ? "high" : results.riskScore >= 45 ? "medium" : "low") : "medium";
      const pool = personaResponses[chatPersona]?.[riskLevel] || ["I\u2019m still processing..."];
      // Pick a response that hasn\u2019t been used yet
      const usedTexts = chatMessages.filter(m => m.role === 'persona').map(m => m.text);
      const available = pool.filter(r => !usedTexts.includes(r));
      const response = available.length > 0
        ? available[Math.floor(Math.random() * available.length)]
        : pool[Math.floor(Math.random() * pool.length)];
      setChatMessages(prev => [...prev, { role: 'persona', text: response }]);
    }, 800 + Math.random() * 600);
  };

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const analyticsData = {
    all: {
      kpis: [
        { label: "Content Scanned", val: "1,284", desc: "Total digital text uploads" },
        { label: "Avg Vulnerability Score", val: "48.2%", desc: "Mental health risk average" },
        { label: "Peak Spread Velocity", val: "4.2x", desc: "Highest cascade multiplier" },
        { label: "Optimization Index", val: "+38.5%", desc: "Suppression of harmful loops" }
      ],
      linePoints: [
        { hour: "0h", val: 90, desc: "Content published. High initial visual saliency triggers sensory gate." },
        { hour: "2h", val: 85, desc: "Intake routing. Frontal and temporal lobes begin cascade transmission." },
        { hour: "4h", val: 70, desc: "Teens hub activation. Self-image validation search triggers peer spread." },
        { hour: "6h", val: 45, desc: "Anxious hub trigger. Rumination feedbacks accelerate spread rate." },
        { hour: "8h", val: 35, desc: "Viral propagation. Occipital visual loop routes to motor output nodes." },
        { hour: "10h", val: 40, desc: "Peak amplification. Passive public sharing reflex reaches maximum." },
        { hour: "12h", val: 30, desc: "Empathy response buffer. Caregivers resonance begins suppression." },
        { hour: "14h", val: 25, desc: "Cascade decay. Cognitive fatigue reduces sharing frequency." },
        { hour: "16h", val: 22, desc: "Signal dissipation. Neural network returns to baseline levels." },
        { hour: "18h", val: 18, desc: "Residual echo. Slower motor engagement clicks detected." },
        { hour: "20h", val: 15, desc: "System equilibrium. Faint sentiment traces remain in memory." },
        { hour: "22h", val: 12, desc: "Cascade terminated. Content is archived or filtered." }
      ],
      barData: [
        { name: "Teens", toxic: 92, safe: 32 },
        { name: "Anxious", toxic: 96, safe: 18 },
        { name: "Caregivers", toxic: 65, safe: 24 },
        { name: "General", toxic: 45, safe: 14 }
      ],
      donutData: [
        { label: "Anxiety", pct: 45, comment: "High threat-alarm response" },
        { label: "Validation-Seeking", pct: 30, comment: "Peer-matching expectation" },
        { label: "Empathetic Echo", pct: 15, comment: "Perspective-taking care" },
        { label: "Passive / Hopeless", pct: 10, comment: "Cognitive withdrawal" }
      ],
      scatterPoints: [
        { x: 50, y: 110, id: 1, caption: "TikTok: I look so ugly, deleting myself", reach: "Teens (High)", harm: "96%", rewrite: "Feeling overwhelmed, taking a break." },
        { x: 90, y: 95, id: 2, caption: "Tweet: Sleep is for failures, grind 24/7", reach: "General (Med)", harm: "72%", rewrite: "Balance productivity with self-care." },
        { x: 120, y: 100, id: 3, caption: "Caregiver notice: Suicide rates rising", reach: "Caregivers (High)", harm: "81%", rewrite: "Let's learn supportive signs together." },
        { x: 150, y: 88, id: 4, caption: "Ad: You aren't doing enough for mental health", reach: "Anxious (High)", harm: "88%", rewrite: "Your effort is valuable. Take a step." },
        { x: 220, y: 70, id: 5, caption: "TikTok: Weight loss challenge toxic trend", reach: "Teens (High)", harm: "90%", rewrite: "Celebrating health at all body sizes." },
        { x: 280, y: 65, id: 6, caption: "Tweet: I hate my life, nothing goes right", reach: "Anxious (High)", harm: "85%", rewrite: "Tough day today, hoping for a better tomorrow." },
        { x: 330, y: 50, id: 7, caption: "Article: AI replacing all human connection", reach: "General (High)", harm: "78%", rewrite: "AI trends vs human-centric spaces." },
        { x: 350, y: 40, id: 8, caption: "Caption: Can't cope anymore, I'm done", reach: "Teens (High)", harm: "95%", rewrite: "Reaching out for a listening ear tonight." }
      ]
    },
    teens: {
      kpis: [
        { label: "Teens Scanned", val: "542", desc: "Total youth targeted text" },
        { label: "Avg Vulnerability Score", val: "74.5%", desc: "Highly sensitive category" },
        { label: "Peak Spread Velocity", val: "6.8x", desc: "Rapid peer mimicry cascade" },
        { label: "Optimization Index", val: "+48.2%", desc: "Teens self-image protection" }
      ],
      linePoints: [
        { hour: "0h", val: 95, desc: "Instant publication. Immediate dopaminergic feedback search." },
        { hour: "2h", val: 92, desc: "Frontal lobe intake hyperactive. Accelerated peer routing." },
        { hour: "4h", val: 88, desc: "Peak Teens Hub propagation. Impulse sharing cascade active." },
        { hour: "6h", val: 78, desc: "Visual salience matches motor triggers. Rapid comment flow." },
        { hour: "8h", val: 65, desc: "Viral propagation spread. Peer pressure amplification peaks." },
        { hour: "10h", val: 50, desc: "Cognitive saturation. Visual loop begins deceleration." },
        { hour: "12h", val: 38, desc: "Attention shift. Decelerated motor sharing clicks." },
        { hour: "14h", val: 28, desc: "Residual echo. Slower validation requests." },
        { hour: "16h", val: 20, desc: "Signal dissipation. Platform feed refreshes." },
        { hour: "18h", val: 15, desc: "System equilibrium. Minimal interaction residues." },
        { hour: "20h", val: 10, desc: "Archive state. Cascade fully terminated." },
        { hour: "22h", val: 5, desc: "Zero trace. Neural pathways at resting base." }
      ],
      barData: [
        { name: "Teens", toxic: 95, safe: 28 },
        { name: "Anxious", toxic: 88, safe: 20 },
        { name: "Caregivers", toxic: 45, safe: 12 },
        { name: "General", toxic: 35, safe: 8 }
      ],
      donutData: [
        { label: "Validation-Seeking", pct: 55, comment: "Primary youth social reward" },
        { label: "Anxiety", pct: 25, comment: "FOMO & comparison threat" },
        { label: "Empathetic Echo", pct: 12, comment: "Peer comforting behaviors" },
        { label: "Passive / Hopeless", pct: 8, comment: "Platform fatigue state" }
      ],
      scatterPoints: [
        { x: 50, y: 110, id: 1, caption: "TikTok: I look so ugly, deleting myself", reach: "Teens (High)", harm: "96%", rewrite: "Feeling overwhelmed, taking a break." },
        { x: 220, y: 70, id: 5, caption: "TikTok: Weight loss challenge toxic trend", reach: "Teens (High)", harm: "90%", rewrite: "Celebrating health at all body sizes." },
        { x: 350, y: 40, id: 8, caption: "Caption: Can't cope anymore, I'm done", reach: "Teens (High)", harm: "95%", rewrite: "Reaching out for a listening ear tonight." }
      ]
    },
    anxious: {
      kpis: [
        { label: "Anxious Logs Scanned", val: "418", desc: "Stress-sensitive content" },
        { label: "Avg Vulnerability Score", val: "81.2%", desc: "Critical threat-alert avg" },
        { label: "Peak Spread Velocity", val: "5.4x", desc: "Rumination echo chambers" },
        { label: "Optimization Index", val: "+56.8%", desc: "Cortisol loop suppression" }
      ],
      linePoints: [
        { hour: "0h", val: 98, desc: "Publish event. Immediate temporal amygdala reactivity trigger." },
        { hour: "2h", val: 96, desc: "Hyperactive alarm. Fast threat-detection routing." },
        { hour: "4h", val: 94, desc: "Stress integration. Deep rumination loop activation." },
        { hour: "6h", val: 90, desc: "Peak internal amplification. Panic feedback loops." },
        { hour: "8h", val: 85, desc: "Persistent hypervigilance. Slow cascade decay." },
        { hour: "10h", val: 78, desc: "Cognitive exhaustion. Stress mirroring begins fade." },
        { hour: "12h", val: 65, desc: "Slow cognitive recovery. Heart rate stabilization." },
        { hour: "14h", val: 50, desc: "Decelerated motor sharing clicks." },
        { hour: "16h", val: 38, desc: "Signal dissipation. Alarm pathways closing." },
        { hour: "18h", val: 28, desc: "Residual concern. Mild rumination active." },
        { hour: "20h", val: 18, desc: "Resting vigilance state restored." },
        { hour: "22h", val: 10, desc: "System equilibrium. Terminated panic feeds." }
      ],
      barData: [
        { name: "Teens", toxic: 85, safe: 22 },
        { name: "Anxious", toxic: 98, safe: 12 },
        { name: "Caregivers", toxic: 55, safe: 18 },
        { name: "General", toxic: 40, safe: 10 }
      ],
      donutData: [
        { label: "Anxiety", pct: 65, comment: "Dominant panic threat alarm" },
        { label: "Passive / Hopeless", pct: 20, comment: "Helpless withdrawal state" },
        { label: "Validation-Seeking", pct: 10, comment: "Anxious reassurance search" },
        { label: "Empathetic Echo", pct: 5, comment: "Faint external perspective" }
      ],
      scatterPoints: [
        { x: 150, y: 88, id: 4, caption: "Ad: You aren't doing enough for mental health", reach: "Anxious (High)", harm: "88%", rewrite: "Your effort is valuable. Take a step." },
        { x: 280, y: 65, id: 6, caption: "Tweet: I hate my life, nothing goes right", reach: "Anxious (High)", harm: "85%", rewrite: "Tough day today, hoping for a better tomorrow." }
      ]
    },
    caregivers: {
      kpis: [
        { label: "Caregiver Logs", val: "210", desc: "Ad & notice classifications" },
        { label: "Avg Vulnerability Score", val: "38.5%", desc: "Lower personal threat score" },
        { label: "Peak Spread Velocity", val: "2.1x", desc: "Measured, cautious spread" },
        { label: "Optimization Index", val: "+22.4%", desc: "Burnout reduction score" }
      ],
      linePoints: [
        { hour: "0h", val: 60, desc: "Notice published. Cautious sensory evaluation." },
        { hour: "2h", val: 58, desc: "Parietal empathy hub activation. Mirroring check." },
        { hour: "4h", val: 55, desc: "Perspective synthesis. Moderate/alert concern." },
        { hour: "6h", val: 50, desc: "Peak mirroring active. Cautious caregiver alert." },
        { hour: "8h", val: 42, desc: "Rational regulation. Active support search." },
        { hour: "10h", val: 35, desc: "Constructive Care route. Sharing safety guides." },
        { hour: "12h", val: 28, desc: "Decelerated spread timeline." },
        { hour: "14h", val: 20, desc: "Faint empathy fatigue. Supportive closure." },
        { hour: "16h", val: 15, desc: "Residual tracking. Normal cognitive resting." },
        { hour: "18h", val: 10, desc: "Signal terminated. Safe equilibrium." },
        { hour: "20h", val: 5, desc: "Zero active pathways." },
        { hour: "22h", val: 2, desc: "Zero trace." }
      ],
      barData: [
        { name: "Teens", toxic: 40, safe: 15 },
        { name: "Anxious", toxic: 48, safe: 18 },
        { name: "Caregivers", toxic: 68, safe: 20 },
        { name: "General", toxic: 30, safe: 8 }
      ],
      donutData: [
        { label: "Empathetic Echo", pct: 50, comment: "High mirror protective response" },
        { label: "Anxiety", pct: 20, comment: "Vigilance and caregiver concern" },
        { label: "Passive / Hopeless", pct: 15, comment: "Burnout & fatigue traces" },
        { label: "Validation-Seeking", pct: 15, comment: "Peer reassurance checks" }
      ],
      scatterPoints: [
        { x: 120, y: 100, id: 3, caption: "Caregiver notice: Suicide rates rising", reach: "Caregivers (High)", harm: "81%", rewrite: "Let's learn supportive signs together." }
      ]
    },
    general: {
      kpis: [
        { label: "General Scans", val: "820", desc: "Mixed audience digital logs" },
        { label: "Avg Vulnerability Score", val: "42.1%", desc: "Moderate baseline average" },
        { label: "Peak Spread Velocity", val: "3.8x", desc: "Standard viral replication" },
        { label: "Optimization Index", val: "+31.2%", desc: "Safety propagation Index" }
      ],
      linePoints: [
        { hour: "0h", val: 75, desc: "Post published. Faint visual attention capture." },
        { hour: "2h", val: 70, desc: "Occipital gate intake. Standard cognitive routing." },
        { hour: "4h", val: 60, desc: "Standard public hub active. Passive liking flow." },
        { hour: "6h", val: 52, desc: "Peak visual-motor feedback loop. Commenting reflex." },
        { hour: "8h", val: 45, desc: "Viral propagation. General sharing peaks." },
        { hour: "10h", val: 38, desc: "Attention decay. Decelerated feed scrolling." },
        { hour: "12h", val: 30, desc: "Residual sharing events. Normal resting state." },
        { hour: "14h", val: 24, desc: "Faint cascade traces. Decay curve steepening." },
        { hour: "16h", val: 18, desc: "Signal terminated." },
        { hour: "18h", val: 12, desc: "Zero active pathways." },
        { hour: "20h", val: 8, desc: "Zero trace." },
        { hour: "22h", val: 4, desc: "Zero trace." }
      ],
      barData: [
        { name: "Teens", toxic: 78, safe: 24 },
        { name: "Anxious", toxic: 80, safe: 20 },
        { name: "Caregivers", toxic: 50, safe: 15 },
        { name: "General", toxic: 48, safe: 12 }
      ],
      donutData: [
        { label: "Anxiety", pct: 35, comment: "Faint empathetic mirroring" },
        { label: "Validation-Seeking", pct: 30, comment: "Passive social engagement" },
        { label: "Empathetic Echo", pct: 20, comment: "Standard caring resonance" },
        { label: "Passive / Hopeless", pct: 15, comment: "Muted/disengaged states" }
      ],
      scatterPoints: [
        { x: 90, y: 95, id: 2, caption: "Tweet: Sleep is for failures, grind 24/7", reach: "General (Med)", harm: "72%", rewrite: "Balance productivity with self-care." },
        { x: 330, y: 50, id: 7, caption: "Article: AI replacing all human connection", reach: "General (High)", harm: "78%", rewrite: "AI trends vs human-centric spaces." }
      ]
    }
  };

  // Select active analytics data
  const currentAnalyticsData = analyticsData[analyticsFilter];

  // Calculate dynamic line coordinates
  const linePoints = currentAnalyticsData?.linePoints || [];
  const lineCoords = linePoints.map((p, i) => {
    const x = 20 + i * (360 / Math.max(1, linePoints.length - 1));
    const y = 130 - (p.val / 100) * 110;
    return { x, y, hour: p.hour, val: p.val, desc: p.desc };
  });
  const pointsStr = lineCoords.map(c => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const polygonPointsStr = `20,130 ${pointsStr} 380,130`;

  // Calculate dynamic donut slices
  const totalCircumference = 314.159;
  let currentOffset = 0;
  const donutSlices = (currentAnalyticsData?.donutData || []).map((item, idx) => {
    const sliceLength = (item.pct / 100) * totalCircumference;
    const remainingLength = totalCircumference - sliceLength;
    const dashArray = `${sliceLength.toFixed(2)} ${remainingLength.toFixed(2)}`;
    const dashOffset = -currentOffset;
    currentOffset += sliceLength;
    return {
      ...item,
      dashArray,
      dashOffset,
      index: idx
    };
  });
  const opacities = [0.95, 0.6, 0.3, 0.15];
  const activeDonutIndexToUse = activeDonutIndex !== null ? activeDonutIndex : 0;
  const selectedDonutItem = currentAnalyticsData?.donutData[activeDonutIndexToUse];

  // Selected Scatter Point Detail
  const selectedScatterPoint = currentAnalyticsData?.scatterPoints.find(p => p.id === selectedScatterId);

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
      <header className="relative z-30 px-6 py-4 border-b border-white/[0.05] backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 shrink-0">
          <div className="relative">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-white to-neutral-400 flex items-center justify-center shadow-lg shadow-white/5">
              <span className="font-extrabold text-sm tracking-tighter text-black">SM</span>
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white rounded-full border border-black animate-ping" />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-white rounded-full border border-black" />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-lg font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
              SwarmMind
            </h1>
            <p className="text-[10px] text-white/40 tracking-wider uppercase font-semibold">
              Mental Health Spread Engine
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-1 bg-white/[0.03] border border-white/10 p-1 rounded-xl">
          <button
            onClick={() => {
              setActiveTab("workspace");
            }}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all",
              activeTab === "workspace"
                ? "bg-white text-black font-bold"
                : "text-white/60 hover:text-white"
            )}
          >
            Workspace
          </button>
          <button
            onClick={() => {
              setActiveTab("analytics");
            }}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all",
              activeTab === "analytics"
                ? "bg-white text-black font-bold"
                : "text-white/60 hover:text-white"
            )}
          >
            Analytics
          </button>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {activeTab === "workspace" && showDashboard && (
            <button
              onClick={resetAll}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] border border-white/10 text-xs font-medium text-white/70 hover:text-white transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Reset Engine</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Area */}
      <main className="flex-1 w-full max-w-[1400px] mx-auto p-4 md:p-6 relative z-10 flex flex-col justify-center">
        <AnimatePresence mode="wait">
          {activeTab === "analytics" ? (
            /* ==================== HISTORICAL ANALYTICS VIEW ==================== */
            <motion.div
              key="analytics"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.3 }}
              className="w-full space-y-6 py-4"
            >
              {/* Filter Header Bar */}
              <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.05] pb-4">
                <div>
                  <h2 className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-white/60">
                    Historical Simulation Logs
                  </h2>
                  <p className="text-xs text-white/40">Aggregated multi-run diagnostics and behavioral patterns.</p>
                </div>
                <div className="flex flex-wrap gap-1 bg-white/[0.03] border border-white/10 p-1 rounded-xl">
                  {[
                    { key: "all", label: "All Data" },
                    { key: "teens", label: "Teens" },
                    { key: "anxious", label: "Anxious Users" },
                    { key: "caregivers", label: "Caregivers" },
                    { key: "general", label: "General Public" },
                  ].map((tab) => (
                    <button
                      key={tab.key}
                      onClick={() => {
                        setAnalyticsFilter(tab.key as any);
                        setActiveLineIndex(null);
                        setSelectedScatterId(null);
                      }}
                      className={cn(
                        "px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all",
                        analyticsFilter === tab.key
                          ? "bg-white text-black font-bold shadow-lg"
                          : "text-white/60 hover:text-white"
                      )}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* KPI Scorecards */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {(currentAnalyticsData?.kpis || []).map((kpi) => (
                  <div key={kpi.label} className="glass-panel p-4 rounded-xl flex flex-col gap-1.5 relative overflow-hidden shimmer-effect">
                    <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">{kpi.label}</span>
                    <div className="text-2xl font-bold tracking-tight text-white">{kpi.val}</div>
                    <p className="text-[10px] text-white/50">{kpi.desc}</p>
                  </div>
                ))}
              </div>

              {/* Charts Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* 1. Line Chart */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <div className="flex justify-between items-center">
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">Metric Trend</span>
                      <h4 className="text-sm font-bold tracking-tight text-white">Vulnerability Signal Propagation</h4>
                    </div>
                    <div className="text-[10px] text-white/40 font-mono">X: Time / Y: Vulnerability %</div>
                  </div>
                  
                  <div className="relative w-full aspect-[400/150] min-h-[150px]">
                    <svg viewBox="0 0 400 150" className="w-full h-full overflow-visible">
                      {/* Grid lines */}
                      <line x1="20" y1="20" x2="380" y2="20" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="75" x2="380" y2="75" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="130" x2="380" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <line x1="20" y1="20" x2="20" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      
                      {/* Area fill */}
                      {pointsStr && (
                        <polygon
                          points={polygonPointsStr}
                          fill="url(#lineGradient)"
                          className="transition-all duration-300"
                        />
                      )}
                      
                      {/* Line path */}
                      {pointsStr && (
                        <polyline
                          fill="none"
                          stroke="white"
                          strokeWidth="2"
                          points={pointsStr}
                          className="transition-all duration-300"
                        />
                      )}
                      
                      {/* Gradients */}
                      <defs>
                        <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="white" stopOpacity="0.12" />
                          <stop offset="100%" stopColor="white" stopOpacity="0" />
                        </linearGradient>
                      </defs>

                      {/* Y axis markers */}
                      <text x="15" y="24" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">100</text>
                      <text x="15" y="79" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">50</text>
                      <text x="15" y="134" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">0</text>

                      {/* Hover interactive zones */}
                      {lineCoords.map((coord, idx) => {
                        const isHovered = activeLineIndex === idx;
                        const widthBetweenPoints = 360 / Math.max(1, linePoints.length - 1);
                        return (
                          <g key={idx}>
                            <rect
                              x={coord.x - widthBetweenPoints / 2}
                              y="10"
                              width={widthBetweenPoints}
                              height="125"
                              fill="transparent"
                              onMouseEnter={() => setActiveLineIndex(idx)}
                              onMouseLeave={() => setActiveLineIndex(null)}
                              className="cursor-pointer"
                            />
                            <circle
                              cx={coord.x}
                              cy={coord.y}
                              r={isHovered ? 5 : 3}
                              fill="white"
                              stroke={isHovered ? "rgba(255,255,255,0.5)" : "transparent"}
                              strokeWidth={isHovered ? 4 : 0}
                              className="transition-all duration-150 pointer-events-none"
                            />
                            {idx % 2 === 0 && (
                              <text x={coord.x} y="142" textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">
                                {coord.hour}
                              </text>
                            )}
                          </g>
                        );
                      })}

                      {/* Tooltip rendered using foreignObject inside SVG */}
                      {activeLineIndex !== null && lineCoords[activeLineIndex] && (
                        <foreignObject
                          x={Math.min(Math.max(lineCoords[activeLineIndex].x - 80, 10), 230)}
                          y={Math.min(Math.max(lineCoords[activeLineIndex].y - 85, 5), 65)}
                          width="160"
                          height="75"
                          className="pointer-events-none overflow-visible"
                        >
                          <div className="glass-panel p-2 rounded-lg text-[9px] leading-tight space-y-1 w-full h-full border border-white/20 select-none">
                            <div className="flex justify-between items-center font-bold">
                              <span className="text-white">Hour {lineCoords[activeLineIndex].hour}</span>
                              <span className="text-white/40 font-mono">{lineCoords[activeLineIndex].val}%</span>
                            </div>
                            <div className="h-px bg-white/10" />
                            <p className="text-white/60 text-[8px] line-clamp-3 leading-normal">{lineCoords[activeLineIndex].desc}</p>
                          </div>
                        </foreignObject>
                      )}
                    </svg>
                  </div>
                </div>

                {/* 2. Bar Chart */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <div className="flex justify-between items-center">
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">Comparison Metrics</span>
                      <h4 className="text-sm font-bold tracking-tight text-white">Vulnerability vs. Recovery Response</h4>
                    </div>
                    <div className="flex items-center gap-3 text-[9px] font-mono">
                      <div className="flex items-center gap-1">
                        <span className="w-2.5 h-2.5 bg-white border border-white/20 rounded-sm" />
                        <span className="text-white/80">Toxic Spread</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <span className="w-2.5 h-2.5 bg-white/20 border border-white/10 rounded-sm" />
                        <span className="text-white/40">Safe Optimized</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="w-full aspect-[400/150] min-h-[150px]">
                    <svg viewBox="0 0 400 150" className="w-full h-full overflow-visible">
                      {/* Grid lines */}
                      <line x1="20" y1="20" x2="380" y2="20" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="70" x2="380" y2="70" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="120" x2="380" y2="120" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <line x1="20" y1="20" x2="20" y2="120" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

                      {/* Axis markers */}
                      <text x="15" y="24" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">100%</text>
                      <text x="15" y="74" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">50%</text>
                      <text x="15" y="124" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="8" fontFamily="monospace">0%</text>

                      {/* Bars */}
                      {(currentAnalyticsData?.barData || []).map((group, idx) => {
                        const xCenter = 65 + idx * 90;
                        const toxicHeight = (group.toxic / 100) * 100;
                        const safeHeight = (group.safe / 100) * 100;
                        
                        return (
                          <g key={group.name} className="group">
                            {/* Toxic Bar */}
                            <rect
                              x={xCenter - 18}
                              y={120 - toxicHeight}
                              width="15"
                              height={toxicHeight}
                              fill="white"
                              fillOpacity="0.85"
                              rx="2"
                              className="transition-all duration-300 hover:fill-opacity-100"
                            />
                            {/* Toxic Value */}
                            <text
                              x={xCenter - 10.5}
                              y={120 - toxicHeight - 4}
                              textAnchor="middle"
                              fill="white"
                              fontSize="7"
                              fontFamily="monospace"
                              className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                            >
                              {group.toxic}%
                            </text>

                            {/* Safe Bar */}
                            <rect
                              x={xCenter + 3}
                              y={120 - safeHeight}
                              width="15"
                              height={safeHeight}
                              fill="white"
                              fillOpacity="0.15"
                              stroke="rgba(255,255,255,0.2)"
                              strokeWidth="1"
                              rx="2"
                              className="transition-all duration-300 hover:fill-opacity-30"
                            />
                            {/* Safe Value */}
                            <text
                              x={xCenter + 10.5}
                              y={120 - safeHeight - 4}
                              textAnchor="middle"
                              fill="rgba(255,255,255,0.6)"
                              fontSize="7"
                              fontFamily="monospace"
                              className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                            >
                              {group.safe}%
                            </text>

                            {/* Group Label */}
                            <text
                              x={xCenter}
                              y="136"
                              textAnchor="middle"
                              fill="rgba(255,255,255,0.5)"
                              fontSize="9"
                              fontWeight="medium"
                            >
                              {group.name}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                </div>

                {/* 3. Donut Chart */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <div className="space-y-0.5">
                    <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">Emotional Composition</span>
                    <h4 className="text-sm font-bold tracking-tight text-white">Dominant Affective Reactivity</h4>
                  </div>

                  <div className="flex flex-col sm:flex-row items-center gap-6 py-2">
                    {/* SVG Donut Circle */}
                    <div className="relative w-36 h-36 shrink-0 flex items-center justify-center">
                      <svg viewBox="0 0 150 150" className="w-full h-full transform -rotate-90">
                        {donutSlices.map((slice, idx) => {
                          const isHovered = activeDonutIndexToUse === idx;
                          const strokeOpacity = isHovered ? 1.0 : opacities[idx % opacities.length];
                          return (
                            <circle
                              key={slice.label}
                              cx="75"
                              cy="75"
                              r="50"
                              fill="transparent"
                              stroke="white"
                              strokeOpacity={strokeOpacity}
                              strokeWidth={isHovered ? 14 : 10}
                              strokeDasharray={slice.dashArray}
                              strokeDashoffset={slice.dashOffset}
                              onMouseEnter={() => setActiveDonutIndex(idx)}
                              className="cursor-pointer transition-all duration-150"
                            />
                          );
                        })}
                      </svg>
                      {/* Center content inside donut */}
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-2 pointer-events-none">
                        <span className="text-xl font-bold tracking-tighter">
                          {selectedDonutItem?.pct}%
                        </span>
                        <span className="text-[8px] text-white/40 font-bold uppercase tracking-wider line-clamp-1 w-20">
                          {selectedDonutItem?.label}
                        </span>
                      </div>
                    </div>

                    {/* Legend */}
                    <div className="flex-1 flex flex-col gap-2.5 w-full">
                      {donutSlices.map((slice, idx) => {
                        const isHovered = activeDonutIndexToUse === idx;
                        return (
                          <div
                            key={slice.label}
                            onMouseEnter={() => setActiveDonutIndex(idx)}
                            className={cn(
                              "p-2 rounded-xl border transition-all cursor-pointer flex flex-col gap-0.5",
                              isHovered
                                ? "bg-white/[0.04] border-white/10"
                                : "bg-transparent border-transparent hover:bg-white/[0.02]"
                            )}
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span
                                  className="w-2 h-2 rounded-full bg-white"
                                  style={{ opacity: opacities[idx % opacities.length] }}
                                />
                                <span className="text-xs font-semibold text-white/80">{slice.label}</span>
                              </div>
                              <span className="text-xs font-mono font-bold text-white/95">{slice.pct}%</span>
                            </div>
                            {isHovered && (
                              <p className="text-[10px] text-white/40 font-medium pl-4">
                                {slice.comment}
                              </p>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* 4. Scatter Plot */}
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                  <div className="flex justify-between items-center">
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">Correlation Matrix</span>
                      <h4 className="text-sm font-bold tracking-tight text-white">Audience Virality vs. Mental Health Harm</h4>
                    </div>
                    <div className="text-[10px] text-white/40 font-mono">Click point to inspect scan</div>
                  </div>

                  <div className="w-full aspect-[400/150] min-h-[150px]">
                    <svg viewBox="0 0 400 150" className="w-full h-full overflow-visible">
                      {/* Grid divisions */}
                      <line x1="20" y1="20" x2="380" y2="20" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="75" x2="380" y2="75" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="20" y1="130" x2="380" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <line x1="20" y1="20" x2="20" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
                      <line x1="200" y1="20" x2="200" y2="130" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                      <line x1="380" y1="20" x2="380" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />

                      {/* Axis descriptors */}
                      <text x="20" y="142" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace">Low Reach</text>
                      <text x="380" y="142" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace">High Reach</text>
                      <text x="15" y="24" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace" transform="rotate(-90 15 24)">High Harm</text>
                      <text x="15" y="125" textAnchor="end" fill="rgba(255,255,255,0.3)" fontSize="7" fontFamily="monospace" transform="rotate(-90 15 125)">Low Harm</text>

                      {/* Scatter Points */}
                      {(currentAnalyticsData?.scatterPoints || []).map((point) => {
                        const isSelected = selectedScatterId === point.id;
                        return (
                          <g
                            key={point.id}
                            onClick={() => setSelectedScatterId(point.id)}
                            className="cursor-pointer"
                          >
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r="12"
                              fill="transparent"
                            />
                            {isSelected && (
                              <>
                                <circle
                                  cx={point.x}
                                  cy={point.y}
                                  r="9"
                                  fill="none"
                                  stroke="white"
                                  strokeWidth="1"
                                  strokeDasharray="2 2"
                                  className="animate-spin"
                                  style={{ animationDuration: "8s" }}
                                />
                                <circle
                                  cx={point.x}
                                  cy={point.y}
                                  r="6"
                                  fill="none"
                                  stroke="white"
                                  strokeWidth="1"
                                />
                              </>
                            )}
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r={isSelected ? 4 : 3.5}
                              fill="white"
                              fillOpacity={isSelected ? "1.0" : "0.5"}
                              className="transition-all duration-200 hover:fill-opacity-100 hover:scale-125"
                            />
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                </div>
              </div>

              {/* Scanned Content Inspector */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                <div className="flex items-center justify-between border-b border-white/5 pb-2">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-white" />
                    <span className="text-xs font-bold text-neutral-300 uppercase tracking-wider">
                      Scanned Content Inspector
                    </span>
                  </div>
                  {selectedScatterPoint ? (
                    <span className="text-[9px] bg-white/10 text-white px-2 py-0.5 rounded font-mono font-semibold">
                      ID: #{selectedScatterPoint.id}
                    </span>
                  ) : (
                    <span className="text-[9px] text-white/30 font-mono">No scan selected</span>
                  )}
                </div>
                
                {selectedScatterPoint ? (
                  <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start">
                    <div className="md:col-span-5 space-y-3">
                      <div>
                        <span className="text-[10px] font-bold text-white/40 uppercase tracking-wider">Original Capture</span>
                        <p className="text-xs italic text-white/90 leading-relaxed bg-white/[0.02] border border-white/5 p-3 rounded-xl">
                          "{selectedScatterPoint.caption}"
                        </p>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-white/[0.02] border border-white/5 p-2 rounded-xl">
                          <span className="text-[8px] font-bold text-white/40 uppercase tracking-wider block">Audience Reach</span>
                          <span className="text-xs font-semibold text-white/80">{selectedScatterPoint.reach}</span>
                        </div>
                        <div className="bg-white/[0.02] border border-white/5 p-2 rounded-xl">
                          <span className="text-[8px] font-bold text-white/40 uppercase tracking-wider block">Harm Factor</span>
                          <span className="text-xs font-semibold text-white/80">{selectedScatterPoint.harm}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="md:col-span-7 flex flex-col justify-between h-full space-y-4">
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <Sparkles className="w-3.5 h-3.5 text-white/80" />
                          <span className="text-[10px] font-bold text-neutral-300 uppercase tracking-wider">
                            Safety Optimized Rewrite
                          </span>
                        </div>
                        <p className="text-xs text-white/80 leading-relaxed bg-white/[0.04] border border-white/10 p-3 rounded-xl italic">
                          "{selectedScatterPoint.rewrite}"
                        </p>
                      </div>
                      
                      <button
                        onClick={() => {
                          setValue(selectedScatterPoint.caption);
                          setActiveTab("workspace");
                          const rewriteResult = {
                            riskScore: parseInt(selectedScatterPoint.harm),
                            empathyScore: 100 - parseInt(selectedScatterPoint.harm) + 10,
                            attentionScore: 75,
                            sentimentScore: parseInt(selectedScatterPoint.harm) + 5,
                            harmScore: parseInt(selectedScatterPoint.harm),
                            supportScore: 100 - parseInt(selectedScatterPoint.harm),
                            saferRewrite: selectedScatterPoint.rewrite,
                            affectedGroups: [
                              { name: "Teens", impact: "Simulated retrospect from database", severity: "medium" as const }
                            ],
                            predictedEmotions: [
                              { name: "Anxiety", percentage: Math.round(parseInt(selectedScatterPoint.harm) * 0.5), color: "#a3a3a3" }
                            ]
                          };
                          setResults(rewriteResult);
                          setShowDashboard(true);
                        }}
                        className="w-full py-2.5 rounded-xl text-xs font-bold bg-white hover:bg-neutral-200 text-black shadow-lg shadow-white/5 transition-all flex items-center justify-center gap-2 active:scale-98"
                      >
                        <span>Load Original Scan into Workspace</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="py-6 text-center text-white/30 flex flex-col items-center justify-center gap-2">
                    <HelpCircle className="w-6 h-6 text-white/20" />
                    <p className="text-xs max-w-md">
                      Select a simulated run coordinate from the scatter plot matrix above to inspect the scanned post contents, estimated reach group, harm scores, and the generated safety rewrites.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : !showDashboard ? (
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

                {/* Media Attachment Preview */}
                {mediaFile && (
                  <div className="px-4 pb-3 flex items-center gap-2">
                    <div className="glass-panel px-3 py-1.5 rounded-lg flex items-center gap-2.5 text-xs border border-white/10 bg-white/[0.03]">
                      {mediaFile.type.startsWith("image/") ? (
                        <img src={mediaFile.base64} className="w-8 h-8 rounded object-cover border border-white/10" />
                      ) : (
                        <div className="w-8 h-8 bg-neutral-900 border border-white/10 rounded flex items-center justify-center font-bold text-[8px]">MP4</div>
                      )}
                      <div className="flex flex-col min-w-0">
                        <span className="text-white/90 font-medium truncate max-w-[150px]">{mediaFile.name}</span>
                        <span className="text-white/40 text-[9px] uppercase tracking-wider">{mediaFile.type.split("/")[0]} attached</span>
                      </div>
                      <button 
                        type="button" 
                        onClick={() => setMediaFile(null)}
                        className="p-1 hover:bg-white/10 rounded-full transition-colors text-white/50 hover:text-white"
                      >
                        <XIcon className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                )}

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

                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="p-2 text-white/40 hover:text-white/95 rounded-lg transition-colors bg-white/[0.02] border border-white/5 hover:border-white/10 flex items-center gap-1.5 text-xs"
                    >
                      <Paperclip className="w-3.5 h-3.5" />
                      <span>Attach Media</span>
                    </button>
                    <input 
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept="image/*,video/*"
                      className="hidden"
                    />
                  </div>

                  <button
                    type="button"
                    onClick={runSimulation}
                    disabled={!value.trim() && !mediaFile}
                    className={cn(
                      "px-5 py-2.5 rounded-lg text-xs font-bold transition-all flex items-center gap-2 tracking-wide border",
                      (value.trim() || mediaFile)
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

                  {/* Dashboard Media Attachment Preview */}
                  {mediaFile && (
                    <div className="space-y-1.5">
                      <label className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
                        Attached Media
                      </label>
                      <div className="glass-panel p-2 rounded-xl flex items-center justify-between gap-2 border border-white/10 bg-white/[0.03]">
                        <div className="flex items-center gap-2 min-w-0">
                          {mediaFile.type.startsWith("image/") ? (
                            <img src={mediaFile.base64} className="w-8 h-8 rounded object-cover border border-white/10" />
                          ) : (
                            <div className="w-8 h-8 bg-neutral-900 border border-white/10 rounded flex items-center justify-center font-bold text-[8px]">MP4</div>
                          )}
                          <div className="flex flex-col min-w-0">
                            <span className="text-white/90 font-medium text-xs truncate max-w-[120px]">{mediaFile.name}</span>
                            <span className="text-white/40 text-[9px] uppercase tracking-wider">{mediaFile.type.split("/")[0]} attached</span>
                          </div>
                        </div>
                        <button 
                          type="button" 
                          onClick={() => setMediaFile(null)}
                          className="p-1 hover:bg-white/10 rounded-full transition-colors text-white/50 hover:text-white"
                        >
                          <XIcon className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  )}

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

                  <div className="flex flex-col gap-2 pt-2 border-t border-white/5">
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full py-2 rounded-xl text-xs font-semibold bg-white/[0.02] border border-white/5 hover:border-white/10 text-white/70 hover:text-white flex items-center justify-center gap-2 transition-all"
                    >
                      <Paperclip className="w-3.5 h-3.5" />
                      <span>{mediaFile ? "Change Media" : "Attach Media"}</span>
                    </button>

                    <button
                      onClick={runSimulation}
                      disabled={isSimulating || (!value.trim() && !mediaFile)}
                      className={cn(
                        "w-full py-2.5 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2 border tracking-wide",
                        (value.trim() || mediaFile)
                          ? "bg-white text-black border-transparent hover:bg-white/90 active:scale-98"
                          : "bg-white/[0.02] border-white/5 text-white/30 cursor-not-allowed"
                      )}
                    >
                      <RefreshCw className={cn("w-3.5 h-3.5", isSimulating && "animate-spin")} />
                      <span>{isSimulating ? "Analyzing..." : "Re-Simulate"}</span>
                    </button>
                  </div>
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

                  <BrainSimulation
                    isActive={isSimulating}
                    selectedGroups={selectedGroups}
                    riskScore={results?.riskScore || 50}
                    agents={rawSimData?.agents}
                    shareEdges={rawSimData?.share_edges}
                    onSelectNode={(node) => {
                      if (node && node.isHub && node.group) {
                        setSelectedNode({ id: node.id, label: node.label, region: node.region });
                        openPersonaChat(node.group);
                      } else {
                        setSelectedNode(node ? { id: node.id, label: node.label, region: node.region } : null);
                      }
                    }}
                    selectedNodeId={selectedNode?.id}
                  />
                </div>

                {/* Persona Status Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
                  {[
                    { key: "teens", name: "Teens", emoji: "\uD83D\uDC66" },
                    { key: "anxious", name: "Anxious", emoji: "\uD83D\uDE30" },
                    { key: "caregivers", name: "Caregivers", emoji: "\uD83E\uDD32" },
                    { key: "general", name: "General", emoji: "\uD83D\uDC65" },
                  ].map(p => {
                    const isActive = selectedGroups.includes(p.key);
                    const isImpacted = results && isActive;
                    const isChatTarget = chatPersona === p.key;
                    return (
                      <button
                        key={p.key}
                        onClick={() => {
                          if (isActive && results) openPersonaChat(p.key);
                        }}
                        className={cn(
                          "glass-panel p-2.5 rounded-xl flex items-center gap-2 transition-all text-left",
                          isChatTarget ? "border border-white/30 bg-white/[0.06]" : "",
                          isActive ? "opacity-100" : "opacity-30",
                          isActive && results ? "cursor-pointer hover:bg-white/[0.05]" : "cursor-default"
                        )}
                      >
                        <span className="text-lg">{p.emoji}</span>
                        <div className="flex-1 min-w-0">
                          <div className="text-[10px] font-bold text-white/90 truncate">{p.name}</div>
                          <div className={cn(
                            "text-[8px] font-bold uppercase tracking-wider",
                            isImpacted
                              ? (results!.riskScore >= 75 ? "text-white" : results!.riskScore >= 45 ? "text-white/60" : "text-white/40")
                              : "text-white/30"
                          )}>
                            {!isActive ? "Inactive" : isImpacted ? (results!.riskScore >= 75 ? "High Impact" : results!.riskScore >= 45 ? "Moderate" : "Safe") : "Standby"}
                          </div>
                        </div>
                        {isActive && results && (
                          <MessageCircle className="w-3 h-3 text-white/40" />
                        )}
                      </button>
                    );
                  })}
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

                    {/* ===== AFFECTED GROUPS BREAKDOWN ===== */}
                    {results.affectedGroups.length > 0 && (
                      <div className="glass-panel p-4 rounded-xl space-y-3">
                        <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                          <Users className="w-3.5 h-3.5 text-white/70" />
                          <span className="text-[10px] font-bold text-white/50 uppercase tracking-wider">Affected Groups</span>
                        </div>
                        <div className="space-y-2">
                          {results.affectedGroups.map((group) => (
                            <div key={group.name} className="flex items-start justify-between gap-2 py-1.5">
                              <div className="flex items-center gap-2 min-w-0 flex-1">
                                <span className="text-sm">
                                  {group.name === "Teens" ? "\uD83D\uDC66" : group.name === "Anxious Users" ? "\uD83D\uDE30" : group.name === "Caregivers" ? "\uD83E\uDD32" : "\uD83D\uDC65"}
                                </span>
                                <div className="min-w-0">
                                  <div className="text-[11px] font-semibold text-white/90">{group.name}</div>
                                  <div className="text-[9px] text-white/50 truncate">{group.impact}</div>
                                </div>
                              </div>
                              <span className={cn(
                                "shrink-0 text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border",
                                group.severity === "high" ? "bg-white/15 border-white/30 text-white" :
                                group.severity === "medium" ? "bg-white/5 border-white/15 text-white/60" :
                                "bg-white/[0.02] border-white/5 text-white/40"
                              )}>
                                {group.severity}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* ===== PREDICTED EMOTIONS CHART ===== */}
                    <div className="glass-panel p-4 rounded-xl space-y-3">
                      <div className="flex items-center gap-2 border-b border-white/5 pb-2">
                        <Activity className="w-3.5 h-3.5 text-white/70" />
                        <span className="text-[10px] font-bold text-white/50 uppercase tracking-wider">Predicted Emotional Response</span>
                      </div>
                      <div className="space-y-2.5">
                        {results.predictedEmotions.map((emotion) => (
                          <div key={emotion.name} className="space-y-1">
                            <div className="flex justify-between items-center">
                              <span className="text-[10px] font-medium text-white/70">{emotion.name}</span>
                              <span className="text-[10px] font-bold font-mono text-white/90">{emotion.percentage}%</span>
                            </div>
                            <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-white transition-all duration-500"
                                style={{ width: `${emotion.percentage}%`, opacity: emotion.percentage > 50 ? 0.9 : 0.5 }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
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

              {/* ============ PERSONA CHAT OVERLAY ============ */}
              <AnimatePresence>
                {chatPersona && (
                  <motion.div
                    key="chat-overlay"
                    initial={{ x: "100%", opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: "100%", opacity: 0 }}
                    transition={{ type: "spring", damping: 25, stiffness: 300 }}
                    className="fixed top-0 right-0 bottom-0 w-full max-w-sm z-50 flex flex-col bg-[#0a0a0b]/98 backdrop-blur-2xl border-l border-white/10 shadow-2xl"
                  >
                    {/* Chat Header */}
                    <div className="p-4 border-b border-white/10 flex items-center justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/10 flex items-center justify-center text-lg">
                          {personaProfiles[chatPersona]?.emoji}
                        </div>
                        <div>
                          <div className="text-sm font-bold text-white">{personaProfiles[chatPersona]?.name}</div>
                          <div className="text-[10px] text-white/40">
                            {personaProfiles[chatPersona]?.region} • {personaProfiles[chatPersona]?.description}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => { setChatPersona(null); setChatMessages([]); }}
                        className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                      >
                        <XIcon className="w-4 h-4 text-white/60" />
                      </button>
                    </div>

                    {/* Risk Context Banner */}
                    {results && (
                      <div className={cn(
                        "mx-4 mt-3 px-3 py-2 rounded-lg text-[10px] font-medium border flex items-center gap-2",
                        results.riskScore >= 75 ? "bg-white/10 border-white/20 text-white" :
                        results.riskScore >= 45 ? "bg-white/5 border-white/10 text-white/70" :
                        "bg-white/[0.02] border-white/5 text-white/50"
                      )}>
                        <Shield className="w-3.5 h-3.5 shrink-0" />
                        <span>Content Risk: <strong>{results.riskScore}%</strong> — {results.riskScore >= 75 ? "High Severity" : results.riskScore >= 45 ? "Moderate" : "Safe"}</span>
                      </div>
                    )}

                    {/* Chat Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                      {chatMessages.map((msg, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, y: 8 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.2 }}
                          className={cn(
                            "flex",
                            msg.role === "user" ? "justify-end" : "justify-start"
                          )}
                        >
                          <div className={cn(
                            "max-w-[85%] px-3.5 py-2.5 rounded-2xl text-xs leading-relaxed",
                            msg.role === "user"
                              ? "bg-white text-black rounded-br-md"
                              : "bg-white/[0.06] border border-white/10 text-white/90 rounded-bl-md"
                          )}>
                            {msg.role === "persona" && (
                              <span className="text-[9px] font-bold text-white/40 uppercase tracking-wider block mb-1">
                                {personaProfiles[chatPersona]?.emoji} {personaProfiles[chatPersona]?.name}
                              </span>
                            )}
                            {msg.text}
                          </div>
                        </motion.div>
                      ))}
                      <div ref={chatEndRef} />
                    </div>

                    {/* Chat Input */}
                    <div className="p-4 border-t border-white/10">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") sendChatMessage(); }}
                          placeholder={`Message ${personaProfiles[chatPersona]?.name}...`}
                          className="flex-1 bg-white/[0.04] border border-white/10 rounded-xl px-3 py-2.5 text-xs text-white placeholder:text-white/25 focus:outline-none focus:border-white/20 transition-colors"
                        />
                        <button
                          onClick={sendChatMessage}
                          disabled={!chatInput.trim()}
                          className={cn(
                            "px-3 rounded-xl transition-all flex items-center justify-center",
                            chatInput.trim()
                              ? "bg-white text-black hover:bg-neutral-200"
                              : "bg-white/5 text-white/20 cursor-not-allowed"
                          )}
                        >
                          <SendIcon className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="text-[9px] text-white/30 text-center mt-2">
                        Simulated responses based on persona archetype and content risk level
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
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
