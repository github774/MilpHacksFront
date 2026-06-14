import { AnimatePresence, motion } from "framer-motion";
import { Film } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { cn } from "../lib/utils";

const STAGES = ["Reading file", "Extracting audio", "Transcribing speech"];

interface UploadAnimationProps {
  fileName: string;
  className?: string;
}

export function UploadAnimation({ fileName, className }: UploadAnimationProps) {
  const [stage, setStage] = useState(0);
  const bars = useMemo(() => Array.from({ length: 14 }, (_, i) => i), []);

  useEffect(() => {
    const id = window.setInterval(() => {
      setStage((s) => (s + 1) % STAGES.length);
    }, 2600);
    return () => window.clearInterval(id);
  }, []);

  const shortName =
    fileName.length > 36 ? `${fileName.slice(0, 18)}…${fileName.slice(-14)}` : fileName;

  return (
    <div
      className={cn(
        "relative w-full min-h-[196px] rounded-2xl overflow-hidden glass-panel",
        className
      )}
      role="status"
      aria-live="polite"
      aria-label={`Processing ${fileName}`}
    >
      <div className="absolute inset-0 upload-grid-bg opacity-60 pointer-events-none" />
      <div className="absolute inset-0 upload-scan-sweep pointer-events-none" />

      {/* Film perforations */}
      <div className="absolute left-3 top-0 bottom-0 flex flex-col justify-evenly py-4 pointer-events-none opacity-20">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="w-1.5 h-2.5 rounded-[1px] bg-white/80" />
        ))}
      </div>
      <div className="absolute right-3 top-0 bottom-0 flex flex-col justify-evenly py-4 pointer-events-none opacity-20">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="w-1.5 h-2.5 rounded-[1px] bg-white/80" />
        ))}
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center gap-5 py-9 px-8">
        {/* Icon + pulse rings */}
        <div className="relative w-[72px] h-[72px] grid place-items-center">
          <motion.div
            className="absolute inset-0 rounded-full border border-white/25"
            animate={{ scale: [1, 1.45], opacity: [0.45, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeOut" }}
          />
          <motion.div
            className="absolute inset-3 rounded-full border border-white/15"
            animate={{ scale: [1, 1.3], opacity: [0.35, 0] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeOut", delay: 0.55 }}
          />
          <motion.div
            className="w-12 h-12 rounded-xl bg-white/[0.07] border border-white/20 grid place-items-center shadow-[0_0_24px_rgba(255,255,255,0.08)]"
            animate={{ boxShadow: ["0 0 20px rgba(255,255,255,0.06)", "0 0 32px rgba(255,255,255,0.14)", "0 0 20px rgba(255,255,255,0.06)"] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <Film className="w-5 h-5 text-white/85" strokeWidth={1.75} />
          </motion.div>
        </div>

        {/* Waveform */}
        <div className="flex items-end justify-center gap-[3px] h-11 w-full max-w-[220px]">
          {bars.map((i) => (
            <motion.div
              key={i}
              className="w-[3px] rounded-full bg-white origin-bottom"
              style={{ height: "100%" }}
              animate={{
                scaleY: [0.18, 0.45 + (i % 7) * 0.07, 0.22, 0.72 + (i % 5) * 0.05, 0.18],
                opacity: [0.25, 0.55, 0.3, 0.95, 0.25],
              }}
              transition={{
                duration: 1.05 + (i % 4) * 0.15,
                repeat: Infinity,
                ease: "easeInOut",
                delay: i * 0.045,
              }}
            />
          ))}
        </div>

        {/* Labels */}
        <div className="text-center space-y-1.5 min-h-[42px]">
          <p className="text-[13px] text-white/85 font-medium tracking-tight">{shortName}</p>
          <div className="h-4 overflow-hidden">
            <AnimatePresence mode="wait">
              <motion.p
                key={stage}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.28 }}
                className="text-[10px] text-white/40 font-mono-data uppercase tracking-[0.2em]"
              >
                {STAGES[stage]}
              </motion.p>
            </AnimatePresence>
          </div>
        </div>

        {/* Indeterminate track */}
        <div className="w-full max-w-[260px] h-[2px] rounded-full bg-white/[0.07] overflow-hidden">
          <motion.div
            className="h-full w-[38%] rounded-full bg-gradient-to-r from-transparent via-white/80 to-transparent"
            animate={{ x: ["-120%", "320%"] }}
            transition={{ duration: 1.65, repeat: Infinity, ease: "easeInOut" }}
          />
        </div>
      </div>
    </div>
  );
}
