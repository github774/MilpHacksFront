import { ArrowRight, Brain, GitBranch, Heart, Layers, Sparkles, Users } from "lucide-react";
import type { SimulationResult } from "../api";
import { MH } from "../lib/mentalHealthCopy";

interface Props {
  result: SimulationResult;
}

const STEPS = [
  {
    icon: Layers,
    title: "Listen to your words",
    detail:
      "Each sentence is embedded so the model understands semantic tone — support, pressure, fear, hope — not just keywords.",
  },
  {
    icon: Users,
    title: "Meet diverse viewers",
    detail:
      "10,000 Nemotron personas with real demographics and life stories. Cosine affinity picks who encounters your content first.",
  },
  {
    icon: Brain,
    title: "Predict emotional response",
    detail:
      "PathPool maps transcript × persona to reactions and five affect dimensions (empathy, relation, inspiration, curiosity, joy).",
  },
  {
    icon: GitBranch,
    title: "Trace emotional contagion",
    detail:
      "Viewers who share pass the message to similar peers — modeling how supportive or harmful narratives spread.",
  },
];

export function ModelArchitecture({ result }: Props) {
  const cfg = (result.config ?? {}) as Record<string, unknown>;
  const catalog = result.catalog;
  const s = result.analysis.summary;
  const catalogSize = catalog?.size ?? s.seeds_requested;

  return (
    <div className="space-y-4">
      <div className="rounded-xl glass-panel p-5">
        <div className="flex items-start gap-3 mb-5">
          <div className="w-9 h-9 rounded-lg bg-white/[0.06] border border-white/12 grid place-items-center shrink-0">
            <Heart className="w-4 h-4 text-white/70" />
          </div>
          <div>
            <h3 className="text-[15px] font-semibold text-white">
              How {MH.productName} models content wellbeing
            </h3>
            <p className="text-[12.5px] text-white/50 mt-1 leading-relaxed max-w-2xl">
              Not a generic chatbot guess. Every simulated viewer has a biography, job, and values.
              The engine predicts how your exact transcript would land emotionally — then shows how
              sharing could amplify that impact across a social network.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {STEPS.map((step, i) => (
            <div key={step.title} className="relative rounded-xl ui-inset p-4">
              {i < STEPS.length - 1 && (
                <ArrowRight className="hidden lg:block absolute -right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-white/20 z-10" />
              )}
              <step.icon className="w-4 h-4 text-white/45 mb-2" />
              <div className="text-[12px] font-semibold text-white mb-1">{step.title}</div>
              <p className="text-[11px] text-white/45 leading-relaxed">{step.detail}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px]">
        {[
          { label: "Persona diversity", value: catalogSize.toLocaleString() },
          { label: "Viewers this run", value: `${s.seeds_exposed.toLocaleString()} / ${s.seeds_requested.toLocaleString()}` },
          { label: "Embed model", value: String(cfg.embed_model_name || "BGE-small-en-v1.5").split("/").pop() },
          { label: "Affect sharpness", value: String(cfg.sharpness ?? "2.5") },
          { label: "Contagion waves", value: String(cfg.max_waves ?? "—") },
          { label: "Batch infer", value: String(cfg.batch_size ?? "128") },
          { label: "Reaction classes", value: "5 (incl. share)" },
          { label: "Affect dimensions", value: "5 wellbeing-linked" },
        ].map((row) => (
          <div key={row.label} className="rounded-lg ui-inset px-3 py-2.5">
            <div className="text-white/40 uppercase tracking-wider text-[9px] mb-0.5">{row.label}</div>
            <div className="text-white/85 font-medium tabular-nums truncate">{row.value}</div>
          </div>
        ))}
      </div>

      <p className="text-[11px] text-white/40 leading-relaxed px-1">{MH.disclaimer}</p>
    </div>
  );
}
