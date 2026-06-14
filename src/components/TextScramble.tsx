import { useState, useCallback, useRef, useEffect } from "react";
import { cn } from "../lib/utils";

const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*";

interface TextScrambleProps {
  text: string;
  className?: string;
  /** Run scramble once when mounted (hero entrance). */
  autoPlay?: boolean;
}

export function TextScramble({ text, className = "", autoPlay = false }: TextScrambleProps) {
  const [displayText, setDisplayText] = useState(text);
  const [isHovering, setIsHovering] = useState(false);
  const [isScrambling, setIsScrambling] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frameRef = useRef(0);
  const hasAutoPlayed = useRef(false);

  const scramble = useCallback(() => {
    setIsScrambling(true);
    frameRef.current = 0;
    const duration = text.length * 3;

    if (intervalRef.current) clearInterval(intervalRef.current);

    intervalRef.current = setInterval(() => {
      frameRef.current++;

      const progress = frameRef.current / duration;
      const revealedLength = Math.floor(progress * text.length);

      const newText = text
        .split("")
        .map((char, i) => {
          if (char === " ") return " ";
          if (i < revealedLength) return text[i];
          return CHARS[Math.floor(Math.random() * CHARS.length)];
        })
        .join("");

      setDisplayText(newText);

      if (frameRef.current >= duration) {
        if (intervalRef.current) clearInterval(intervalRef.current);
        setDisplayText(text);
        setIsScrambling(false);
      }
    }, 30);
  }, [text]);

  const handleMouseEnter = () => {
    setIsHovering(true);
    scramble();
  };

  const handleMouseLeave = () => {
    setIsHovering(false);
  };

  useEffect(() => {
    if (autoPlay && !hasAutoPlayed.current) {
      hasAutoPlayed.current = true;
      scramble();
    }
  }, [autoPlay, scramble]);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  return (
    <div
      className={cn(
        "group relative inline-flex flex-col cursor-pointer select-none items-center",
        className
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span className="relative font-sans text-5xl md:text-7xl font-semibold tracking-[0.22em] uppercase">
        {displayText.split("").map((char, i) => (
          <span
            key={i}
            className={cn(
              "inline-block transition-all duration-150",
              isScrambling && char !== text[i]
                ? "text-white/35 scale-110"
                : "text-white"
            )}
            style={{ transitionDelay: `${i * 10}ms` }}
          >
            {char}
          </span>
        ))}
      </span>

      <span className="relative h-px w-full mt-4 overflow-hidden max-w-[min(100%,320px)]">
        <span className="absolute inset-0 bg-white/15" />
        <span
          className={cn(
            "absolute inset-0 bg-white transition-transform duration-500 ease-out origin-left",
            isHovering ? "scale-x-100" : "scale-x-0"
          )}
        />
      </span>

      <span
        className={cn(
          "absolute -inset-6 rounded-2xl bg-white/[0.04] transition-opacity duration-300 -z-10",
          isHovering ? "opacity-100" : "opacity-0"
        )}
      />
    </div>
  );
}
