import { Brain, Heart, HeartHandshake, ShieldAlert } from "lucide-react";
import { MH, MH_PILLARS, MH_WHY_IT_MATTERS } from "../lib/mentalHealthCopy";

const ICONS = [Brain, ShieldAlert, HeartHandshake, Heart] as const;

/** Full mental-health framing — shown on the analytics tab only. */
export function MentalHealthAnalytics() {
  return (
    <div className="space-y-4">
      <div className="glass-panel rounded-2xl p-5 md:p-6">
        <div className="flex flex-wrap items-center gap-2 mb-4">
          <span className="text-[10px] uppercase tracking-[0.16em] text-white/45 font-mono-data px-2 py-0.5 rounded-md ui-inset">
            {MH.productName}
          </span>
          <span className="text-[10px] text-white/35 font-mono-data">{MH.productTagline}</span>
        </div>
        <h2 className="text-[15px] font-semibold text-white mb-2">Why this matters for mental health</h2>
        <p className="text-[13px] text-white/60 leading-relaxed max-w-3xl mb-4">
          Social content shapes mood, self-worth, and help-seeking — especially for young audiences.
          SHIELD simulates diverse viewers reacting to your exact transcript, maps five affect
          dimensions, and previews how emotional messages propagate through a network.
        </p>
        <ul className="space-y-2">
          {MH_WHY_IT_MATTERS.map((line) => (
            <li key={line} className="flex gap-2.5 text-[12px] text-white/55 leading-relaxed">
              <span className="text-white/30 shrink-0 mt-0.5">—</span>
              {line}
            </li>
          ))}
        </ul>
        <p className="text-[11px] text-white/35 mt-4 pt-4 border-t border-white/10 leading-relaxed">
          {MH.disclaimer}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {MH_PILLARS.map((pillar, i) => {
          const Icon = ICONS[i] ?? Heart;
          return (
            <div key={pillar.title} className="ui-inset rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-8 h-8 rounded-lg bg-white/[0.06] border border-white/12 grid place-items-center">
                  <Icon className="w-4 h-4 text-white/70" />
                </div>
                <h3 className="text-[12px] font-semibold text-white">{pillar.title}</h3>
              </div>
              <p className="text-[11px] text-white/45 leading-relaxed">{pillar.body}</p>
            </div>
          );
        })}
      </div>

      <div className="ui-inset rounded-xl px-4 py-3">
        <p className="text-[12px] text-white/65 leading-relaxed">
          <span className="text-white/85 font-medium">Reading these charts:</span> Use spread score
          and share rate to gauge contagion risk; polarization and dislike-shares flag potential harm
          amplification; affect bars show whether empathy, joy, or stress-like responses dominate.
        </p>
      </div>
    </div>
  );
}
