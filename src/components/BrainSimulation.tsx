import { useEffect, useRef, useState } from "react";
import type { EmotionKey } from "../api";
import brainImg from "../assets/realistic_brain_clean.png";
import { cn } from "../lib/utils";

interface BrainSimulationProps {
  // The model's 5-emotion output (mean_probs), each in 0..1.
  emotions: Partial<Record<EmotionKey, number>>;
  dominantEmotion?: EmotionKey;
  // Optional: show a scanning sweep while a run is in flight.
  isActive?: boolean;
  /** Sidebar mode — hides legend and reduces on-canvas labels. */
  compact?: boolean;
  /** Large centerpiece layout with emotion sidebar overlay. */
  hero?: boolean;
  className?: string;
}

// Each model emotion maps to an anatomical region on the sagittal brain image
// so the panel reads as a brain lighting up rather than an abstract graph.
interface EmotionRegion {
  key: EmotionKey;
  label: string;
  /** Full anatomical name shown in lists and legend. */
  region: string;
  /** Shorter label for on-brain pills. */
  regionShort: string;
  blurb: string;
  x: number;
  y: number;
  rx: number;
  ry: number;
  angle?: number;
}

const EMOTION_REGIONS: EmotionRegion[] = [
  {
    key: "inspiration",
    label: "Inspiration",
    region: "Frontal Cortex",
    regionShort: "Frontal Cortex",
    blurb: "Aspiration & meaning-making",
    x: 0.225,
    y: 0.249,
    rx: 0.042,
    ry: 0.032,
    angle: -0.35,
  },
  {
    key: "curiosity",
    label: "Curiosity",
    region: "Dorsolateral Prefrontal Cortex",
    regionShort: "Dorsolateral PFC",
    blurb: "Exploration & information-seeking",
    x: 0.431,
    y: 0.126,
    rx: 0.038,
    ry: 0.028,
    angle: 0.25,
  },
  {
    key: "empathy",
    label: "Empathy",
    region: "Temporoparietal Junction",
    regionShort: "TP Junction",
    blurb: "Perspective-taking & concern",
    x: 0.683,
    y: 0.236,
    rx: 0.036,
    ry: 0.030,
    angle: 0.45,
  },
  {
    key: "relation",
    label: "Relation",
    region: "Superior Temporal Gyrus",
    regionShort: "Superior Temporal",
    blurb: "Social bonding & relatability",
    x: 0.546,
    y: 0.519,
    rx: 0.040,
    ry: 0.032,
    angle: -0.12,
  },
  {
    key: "joy",
    label: "Joy",
    region: "Limbic Reward Circuit",
    regionShort: "Limbic System",
    blurb: "Pleasure & positive affect",
    x: 0.385,
    y: 0.445,
    rx: 0.034,
    ry: 0.028,
    angle: 0.15,
  },
];

export { EMOTION_REGIONS };
export type { EmotionRegion };

// Faint neural tracts connecting regions (kept organic + subtle so the scene
// reads as cortical tissue, not a node diagram).
const TRACTS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 4], [0, 3],
];

interface Spark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
}

interface Ring {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  alpha: number;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function imageSourceSize(source: CanvasImageSource): { w: number; h: number } {
  if (source instanceof HTMLImageElement) {
    return { w: source.naturalWidth, h: source.naturalHeight };
  }
  if (source instanceof HTMLCanvasElement || source instanceof HTMLVideoElement) {
    return { w: source.width, h: source.height };
  }
  if (source instanceof ImageBitmap) {
    return { w: source.width, h: source.height };
  }
  return { w: 0, h: 0 };
}

export function BrainSimulation({
  emotions,
  dominantEmotion,
  isActive = false,
  compact = false,
  hero = false,
  className,
}: BrainSimulationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  /** Sizing container — hero mode attaches to the center column only, not the full grid. */
  const brainAreaRef = useRef<HTMLDivElement>(null);
  const [bgImage, setBgImage] = useState<CanvasImageSource | null>(null);
  // Hover is tracked via a ref so the animation loop isn't torn down on every
  // mouse move (which would reset the pulse phase). The canvas draws the tooltip.
  const hoveredRef = useRef<number | null>(null);
  const selectedRef = useRef<number | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  selectedRef.current = selectedIdx;
  const displayRef = useRef<number[]>(EMOTION_REGIONS.map(() => 0));

  const selectRegion = (idx: number) => {
    setSelectedIdx((prev) => (prev === idx ? null : idx));
  };

  const regionFocus = selectedIdx ?? hoveredIdx;
  const isRegionHighlighted = (idx: number, isDom: boolean) =>
    regionFocus === idx || (regionFocus === null && isDom);

  useEffect(() => {
    const img = new Image();
    img.src = brainImg;
    img.onload = () => setBgImage(img);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // ---- Resolve target intensities from the model's emotion output ----
    const rawVals = EMOTION_REGIONS.map((r) => clamp01(emotions[r.key] ?? 0));
    const maxV = Math.max(0.0001, ...rawVals);
    const hasData = rawVals.some((v) => v > 0);
    // Brightness emphasizes the strongest emotion while keeping the relative
    // profile readable; labels still show each emotion's absolute percentage.
    const targets = rawVals.map((v) => (hasData ? 0.06 + 0.94 * (v / maxV) : 0));
    const pcts = rawVals.map((v) => Math.round(v * 100));

    const computedDominant =
      dominantEmotion ??
      EMOTION_REGIONS[rawVals.indexOf(Math.max(...rawVals))]?.key;
    const dominantIdx = EMOTION_REGIONS.findIndex((r) => r.key === computedDominant);

    let animationId: number;
    let globalTime = 0;
    let sparks: Spark[] = [];
    let ambient: Spark[] = [];
    let rings: Ring[] = [];

    const dpr = () => Math.min(window.devicePixelRatio || 1, 2);

    const resizeCanvas = () => {
      const area = brainAreaRef.current;
      if (!area) return;
      const ratio = dpr();
      canvas.width = area.clientWidth * ratio;
      canvas.height = area.clientHeight * ratio;
      canvas.style.width = `${area.clientWidth}px`;
      canvas.style.height = `${area.clientHeight}px`;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(ratio, ratio);
    };
    resizeCanvas();
    const ro = new ResizeObserver(resizeCanvas);
    if (brainAreaRef.current) ro.observe(brainAreaRef.current);

    // The brain image is contain-fit + centered; regions are positioned
    // relative to that rect so they always sit on the cortex.
    const brainRect = (width: number, height: number) => {
      if (!bgImage) {
        return { x: width * 0.08, y: height * 0.06, w: width * 0.84, h: height * 0.88 };
      }
      const scaleMul = hero ? 0.98 : compact ? 0.92 : 0.96;
      const { w: imgW, h: imgH } = imageSourceSize(bgImage);
      const scale = Math.min(width / imgW, height / imgH) * scaleMul;
      const w = imgW * scale;
      const h = imgH * scale;
      return { x: (width - w) / 2, y: (height - h) / 2, w, h };
    };

    const regionPoint = (i: number, width: number, height: number) => {
      const r = EMOTION_REGIONS[i];
      const rect = brainRect(width, height);
      return { x: rect.x + r.x * rect.w, y: rect.y + r.y * rect.h };
    };

    const regionEllipse = (i: number, width: number, height: number, expand = 1) => {
      const r = EMOTION_REGIONS[i];
      const rect = brainRect(width, height);
      const { x, y } = regionPoint(i, width, height);
      return {
        cx: x,
        cy: y,
        rx: r.rx * rect.w * expand,
        ry: r.ry * rect.h * expand,
        angle: r.angle ?? 0,
      };
    };

    const strokeRegionEllipse = (
      i: number,
      width: number,
      height: number,
      expand = 1,
      dash: number[] = []
    ) => {
      const { cx, cy, rx, ry, angle } = regionEllipse(i, width, height, expand);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.restore();
    };

    const fillRegionEllipse = (
      i: number,
      width: number,
      height: number,
      expand: number,
      alpha: number
    ) => {
      const { cx, cy, rx, ry, angle } = regionEllipse(i, width, height, expand);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fill();
      ctx.restore();
    };

    const hitRegion = (mx: number, my: number, i: number, width: number, height: number) => {
      const { cx, cy, rx, ry, angle } = regionEllipse(i, width, height, 1.25);
      const dx = mx - cx;
      const dy = my - cy;
      const cos = Math.cos(-angle);
      const sin = Math.sin(-angle);
      const lx = dx * cos - dy * sin;
      const ly = dx * sin + dy * cos;
      return (lx / rx) ** 2 + (ly / ry) ** 2 < 1;
    };

    // Live intensity used for drawing: eased value with a soft ambient floor
    // so the brain always feels alive, brightest for the dominant emotion.
    const vizIntensity = (i: number) => {
      const eased = displayRef.current[i] ?? 0;
      const breathe = 0.012 + 0.018 * Math.sin(globalTime * 1.3 + i * 1.7);
      return Math.min(1, eased + breathe * (eased > 0.05 ? 1 : 0.35));
    };

    const spawnAmbient = (width: number, height: number) => {
      ambient = [];
      const rect = brainRect(width, height);
      for (let i = 0; i < 18; i++) {
        ambient.push({
          x: rect.x + Math.random() * rect.w,
          y: rect.y + rect.h * (0.12 + Math.random() * 0.66),
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          life: Math.random(),
          maxLife: 1,
          size: 0.6 + Math.random() * 1.4,
        });
      }
    };

    const draw = () => {
      const width = canvas.width / dpr();
      const height = canvas.height / dpr();
      globalTime += 0.02;

      if (ambient.length === 0) spawnAmbient(width, height);

      ctx.clearRect(0, 0, width, height);

      const focus = hoveredRef.current ?? selectedRef.current;

      // Ease displayed intensities toward target each frame.
      for (let i = 0; i < EMOTION_REGIONS.length; i++) {
        const cur = displayRef.current[i] ?? 0;
        displayRef.current[i] = cur + (targets[i] - cur) * 0.06;
      }

      // 1. Brain anatomy backdrop (transparent PNG — no dark frame)
      if (bgImage) {
        const rect = brainRect(width, height);
        ctx.save();
        ctx.globalAlpha = hero ? 0.55 : compact ? 0.32 : 0.38;
        ctx.drawImage(bgImage, rect.x, rect.y, rect.w, rect.h);
        ctx.restore();
      }

      // 2. Neural tracts between regions (subtle, brighten with activity)
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      TRACTS.forEach(([a, b], i) => {
        const pa = regionPoint(a, width, height);
        const pb = regionPoint(b, width, height);
        const act = (vizIntensity(a) + vizIntensity(b)) / 2;

        const mx = (pa.x + pb.x) / 2;
        const my = (pa.y + pb.y) / 2;
        const nx = -(pb.y - pa.y);
        const ny = pb.x - pa.x;
        const norm = Math.hypot(nx, ny) || 1;
        const bow = 16 + i * 2;
        const cx = mx + (nx / norm) * bow;
        const cy = my + (ny / norm) * bow;

        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.quadraticCurveTo(cx, cy, pb.x, pb.y);
        ctx.strokeStyle = `rgba(255,255,255,${0.03 + act * 0.16})`;
        ctx.lineWidth = 0.8 + act * 1.2;
        ctx.stroke();

        // Traveling signal pulse along the tract
        const t = (globalTime * 0.25 + i * 0.37) % 1;
        const it = 1 - t;
        const px = it * it * pa.x + 2 * it * t * cx + t * t * pb.x;
        const py = it * it * pa.y + 2 * it * t * cy + t * t * pb.y;
        ctx.beginPath();
        ctx.arc(px, py, 1.6 + act * 1.6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${0.18 + act * 0.5})`;
        ctx.fill();
      });
      ctx.restore();

      // 3. Ambient cortical firing (drifting micro-sparks)
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      ambient.forEach((s) => {
        s.x += s.vx;
        s.y += s.vy;
        s.life += 0.01;
        const tw = 0.25 + 0.55 * (0.5 + 0.5 * Math.sin(s.life * 6.28 + s.x));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${tw * 0.32})`;
        ctx.fill();
      });
      ctx.restore();

      // 4. Activation rings (spawned by strongly active regions)
      rings = rings.filter((ring) => ring.alpha > 0.01);
      rings.forEach((ring) => {
        ring.radius += 1.3;
        ring.alpha = Math.max(0, 1 - ring.radius / ring.maxRadius);
        ctx.save();
        ctx.beginPath();
        ctx.arc(ring.x, ring.y, ring.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255,255,255,${ring.alpha * 0.45})`;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.restore();
      });

      // 5. Focused sparks near active regions
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      sparks = sparks.filter((s) => s.life < s.maxLife);
      sparks.forEach((s) => {
        s.x += s.vx;
        s.y += s.vy;
        s.life += 0.02;
        const a = (1 - s.life / s.maxLife) * 0.9;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${a})`;
        ctx.fill();
      });
      ctx.restore();

      // 5b. Static anatomical region outlines (always visible)
      ctx.save();
      ctx.globalCompositeOperation = "screen";
      EMOTION_REGIONS.forEach((_, i) => {
        const inten = vizIntensity(i);
        const isFocused = focus === i;
        const isDominant = hasData && i === dominantIdx;
        const isHighlighted = isFocused || (focus === null && isDominant);
        ctx.lineWidth = isHighlighted ? 1.1 : 0.7;
        ctx.strokeStyle = `rgba(255,255,255,${0.1 + inten * 0.35 + (isFocused ? 0.25 : 0)})`;
        strokeRegionEllipse(i, width, height, 1, isHighlighted ? [] : [3, 4]);
      });
      ctx.restore();

      // 6. Emotion region glows + cores + labels
      EMOTION_REGIONS.forEach((r, i) => {
        const { x, y } = regionPoint(i, width, height);
        const inten = vizIntensity(i);
        const isFocused = focus === i;
        const isDominant = hasData && i === dominantIdx;
        const isHighlighted = isFocused || (focus === null && isDominant);
        const peak = Math.min(0.92, 0.08 + inten * 0.84);
        const pulse = Math.sin(globalTime * (1.6 + inten * 2) + i) * 0.015;

        // Spawn firing sparks proportional to intensity (tight to patch)
        if (Math.random() < inten * 0.22) {
          const { rx, ry, angle } = regionEllipse(i, width, height, 0.85);
          const ang = Math.random() * Math.PI * 2;
          const dist = Math.random() * 0.65;
          sparks.push({
            x: x + Math.cos(ang) * rx * dist,
            y: y + Math.sin(ang) * ry * dist,
            vx: Math.cos(ang + angle) * 0.2,
            vy: Math.sin(ang + angle) * 0.2 - 0.12,
            life: 0,
            maxLife: 0.45 + Math.random() * 0.35,
            size: 0.6 + Math.random() * 1.1,
          });
        }
        if (displayRef.current[i] > 0.45 && Math.random() < displayRef.current[i] * 0.025) {
          const { rx, ry } = regionEllipse(i, width, height, 1);
          rings.push({
            x,
            y,
            radius: Math.max(rx, ry) * 0.5,
            maxRadius: Math.max(rx, ry) * 1.8,
            alpha: 1,
          });
        }

        // Tight layered elliptical highlight (small anatomical patch)
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        fillRegionEllipse(i, width, height, 0.28 + pulse + inten * 0.08, peak * 0.95);
        fillRegionEllipse(i, width, height, 0.52 + pulse + inten * 0.12, peak * 0.42);
        fillRegionEllipse(i, width, height, 0.78 + pulse + inten * 0.1, peak * 0.14);
        ctx.restore();

        // Active region border
        ctx.save();
        ctx.lineWidth = isDominant ? 1.3 : isFocused ? 1.15 : 0.85;
        ctx.strokeStyle = `rgba(255,255,255,${0.18 + inten * 0.55 + (isFocused ? 0.2 : 0)})`;
        strokeRegionEllipse(i, width, height, 0.92 + inten * 0.08);
        ctx.restore();

        // Dominant emotion: tight accent arc on patch edge
        if (isDominant) {
          const { cx, cy, rx, ry, angle } = regionEllipse(i, width, height, 1.05);
          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.ellipse(0, 0, rx, ry, 0, globalTime * 1.4, globalTime * 1.4 + Math.PI * 1.1);
          ctx.strokeStyle = "rgba(255,255,255,0.65)";
          ctx.lineWidth = 1.2;
          ctx.stroke();
          ctx.restore();
        }

        // Bright core — pin-point at region center
        const coreR = 1.8 + inten * 2.2 + (isFocused ? 0.8 : 0);
        ctx.save();
        ctx.beginPath();
        ctx.arc(x, y, coreR, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.shadowBlur = 6 + inten * 10;
        ctx.shadowColor = "rgba(255,255,255,0.9)";
        ctx.fill();
        ctx.restore();

        // Hero: numbered patch markers only (lists live in sidebars — no text pills)
        if (hero) {
          const showNum = isHighlighted || inten > 0.2;
          if (showNum) {
            ctx.save();
            ctx.font = "bold 9px 'IBM Plex Mono', monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = `rgba(255,255,255,${0.35 + inten * 0.55 + (isFocused ? 0.2 : 0)})`;
            ctx.fillText(String(i + 1), x, y);
            ctx.restore();
          }
          return;
        }

        // Non-hero: compact on-brain label (hover / dominant only)
        const showLabel =
          !compact || isFocused || (hasData && i === dominantIdx) || inten > 0.55;
        if (!showLabel) return;

        const pctText = hasData ? `${pcts[i]}%` : "--";
        const regionLine = r.regionShort;
        ctx.font = "bold 9px 'IBM Plex Sans', sans-serif";
        const regionW = ctx.measureText(regionLine).width;
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        const pctW = ctx.measureText(pctText).width;
        const pillW = regionW + pctW + 20;
        const pillH = 18;
        const { ry } = regionEllipse(i, width, height, 1);
        const nearTop = y - ry - (pillH + 8) < 6;
        const pillX = Math.max(6, Math.min(width - pillW - 6, x - pillW / 2));
        const pillY = nearTop ? y + ry + 8 : y - ry - (pillH + 6);

        const labelAlpha = isFocused ? 1 : 0.5 + inten * 0.45;
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(pillX, pillY, pillW, pillH, 8);
        ctx.fillStyle = `rgba(8,12,18,${0.6 * labelAlpha + 0.25})`;
        ctx.fill();
        ctx.strokeStyle = isDominant
          ? `rgba(255,255,255,${0.45 + inten * 0.4})`
          : `rgba(255,255,255,${0.12 + inten * 0.3})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();

        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        ctx.font = "bold 9px 'IBM Plex Sans', sans-serif";
        ctx.fillStyle = `rgba(255,255,255,${labelAlpha})`;
        ctx.fillText(regionLine, pillX + 8, pillY + pillH / 2 + 0.5);
        ctx.textAlign = "right";
        ctx.font = "bold 9px 'IBM Plex Mono', monospace";
        ctx.fillStyle = `rgba(255,255,255,${Math.min(1, labelAlpha + 0.1)})`;
        ctx.fillText(pctText, pillX + pillW - 8, pillY + pillH / 2 + 0.5);
        ctx.restore();
      });

      // 7. Hover tooltip (non-hero — hero uses sidebars for region copy)
      const hi = hoveredRef.current;
      if (hi !== null && !hero) {
        const r = EMOTION_REGIONS[hi];
        const { x, y } = regionPoint(hi, width, height);
        const lines = [r.region, `${r.label} · ${r.blurb}`];
        ctx.save();
        const tipW = Math.min(220, Math.max(168, r.region.length * 6 + 40));
        const tipX = Math.max(6, Math.min(width - tipW - 6, x - tipW / 2));
        const tipY = Math.min(height - 48, y + 16);
        ctx.fillStyle = "rgba(8,10,14,0.9)";
        ctx.strokeStyle = "rgba(255,255,255,0.3)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tipX, tipY, tipW, 40, 6);
        ctx.fill();
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px 'IBM Plex Sans', sans-serif";
        ctx.fillText(lines[0], tipX + tipW / 2, tipY + 15);
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.font = "8px 'IBM Plex Sans', sans-serif";
        ctx.fillText(lines[1], tipX + tipW / 2, tipY + 29);
        ctx.restore();
      }

      // 8. Scanning sweep while the model is running (pre-results)
      if (isActive && !hasData) {
        const rect = brainRect(width, height);
        const sweepX = rect.x + ((globalTime * 90) % rect.w);
        const sg = ctx.createLinearGradient(sweepX - 30, 0, sweepX + 30, 0);
        sg.addColorStop(0, "rgba(255,255,255,0)");
        sg.addColorStop(0.5, "rgba(255,255,255,0.16)");
        sg.addColorStop(1, "rgba(255,255,255,0)");
        ctx.save();
        ctx.globalCompositeOperation = "screen";
        ctx.fillStyle = sg;
        ctx.fillRect(sweepX - 30, rect.y, 60, rect.h);
        ctx.restore();
      }

      animationId = requestAnimationFrame(draw);
    };

    const pickRegion = (e: MouseEvent): number | null => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const width = canvas.width / dpr();
      const height = canvas.height / dpr();
      for (let i = 0; i < EMOTION_REGIONS.length; i++) {
        if (hitRegion(mx, my, i, width, height)) return i;
      }
      return null;
    };

    const handleMouseMove = (e: MouseEvent) => {
      const i = pickRegion(e);
      hoveredRef.current = i;
      setHoveredIdx(i);
      canvas.style.cursor = i !== null ? "pointer" : "default";
    };
    const handleMouseLeave = () => {
      hoveredRef.current = null;
      setHoveredIdx(null);
    };
    const handleClick = (e: MouseEvent) => {
      const i = pickRegion(e);
      if (i !== null) setSelectedIdx((prev) => (prev === i ? null : i));
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);
    canvas.addEventListener("click", handleClick);
    draw();

    return () => {
      ro.disconnect();
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
      canvas.removeEventListener("click", handleClick);
      cancelAnimationFrame(animationId);
    };
  }, [bgImage, isActive, emotions, dominantEmotion, compact, hero]);

  const emotionBars = EMOTION_REGIONS.map((r) => ({
    ...r,
    pct: Math.round(clamp01(emotions[r.key] ?? 0) * 100),
  }));
  const maxPct = Math.max(1, ...emotionBars.map((e) => e.pct));

  return (
    <div
      ref={hero ? undefined : brainAreaRef}
      className={`w-full h-full relative overflow-hidden ${
        hero
          ? "bg-transparent"
          : compact
            ? "bg-transparent"
            : "rounded-xl ui-inset"
      } ${className ?? ""}`}
    >
      {/* Background + canvas — hero uses 3-column grid; brain fills center column */}
      {hero ? (
        <div className="relative z-10 grid h-full w-full grid-cols-1 md:grid-cols-[minmax(0,200px)_1fr_minmax(0,192px)] gap-3 md:gap-5 px-2 md:px-4 py-2 items-center">
          {/* Mobile region chips */}
          <div className="order-2 md:hidden flex gap-2 overflow-x-auto pb-1 -mx-1 px-1 scrollbar-none">
            {EMOTION_REGIONS.map((r, idx) => {
              const isDom = r.key === dominantEmotion;
              const active = isRegionHighlighted(idx, isDom);
              return (
                <button
                  key={r.key}
                  type="button"
                  onClick={() => selectRegion(idx)}
                  className={cn(
                    "shrink-0 rounded-full px-3 py-1.5 border text-[10px] transition-all",
                    active
                      ? "bg-white/10 border-white/25 text-white"
                      : "ui-inset border-white/[0.08] text-white/70"
                  )}
                >
                  {idx + 1}. {r.regionShort}
                </button>
              );
            })}
          </div>

          {/* Left — region buttons (desktop) */}
          <div className="order-3 md:order-1 hidden md:block">
            <div className="text-[9px] uppercase tracking-[0.18em] text-white/35 font-mono-data mb-3">
              Brain regions
            </div>
            <ol className="space-y-2 list-none">
              {EMOTION_REGIONS.map((r, idx) => {
                const isDom = r.key === dominantEmotion;
                const active = isRegionHighlighted(idx, isDom);
                return (
                  <li key={r.key}>
                    <button
                      type="button"
                      onClick={() => selectRegion(idx)}
                      onMouseEnter={() => setHoveredIdx(idx)}
                      onMouseLeave={() => setHoveredIdx(null)}
                      className={cn(
                        "w-full text-left rounded-lg px-2.5 py-2 border transition-all duration-200",
                        "hover:bg-white/[0.07] hover:border-white/15 active:scale-[0.99]",
                        "focus:outline-none focus-visible:ring-1 focus-visible:ring-white/40",
                        active
                          ? "bg-white/[0.1] border-white/25 shadow-[0_0_16px_rgba(255,255,255,0.05)]"
                          : "ui-inset"
                      )}
                    >
                      <div className="flex items-start gap-2">
                        <span
                          className={cn(
                            "text-[10px] font-mono-data tabular-nums w-4 shrink-0 pt-0.5",
                            active ? "text-white/70" : "text-white/30"
                          )}
                        >
                          {idx + 1}
                        </span>
                        <div className="min-w-0">
                          <div
                            className={cn(
                              "text-[11px] font-semibold leading-snug",
                              active ? "text-white" : "text-white/80"
                            )}
                          >
                            {r.region}
                          </div>
                          <div className="text-[10px] text-white/40 capitalize mt-0.5">{r.label}</div>
                        </div>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ol>
          </div>

          {/* Center — brain canvas (sized to this column only so anatomy stays centered) */}
          <div
            ref={brainAreaRef}
            className="order-1 md:order-2 relative min-h-[min(420px,55vh)] h-full w-full min-w-0"
          >
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full cursor-pointer" />
          </div>

          {/* Right — activation buttons */}
          <div className="order-3 hidden sm:block md:order-3">
            <div className="text-[9px] uppercase tracking-[0.18em] text-white/35 font-mono-data mb-3">
              Activation
            </div>
            <div className="space-y-2">
              {emotionBars.map((e, idx) => {
                const isDom = e.key === dominantEmotion;
                const active = isRegionHighlighted(idx, isDom);
                const barW = Math.max(4, (e.pct / maxPct) * 100);
                return (
                  <button
                    key={e.key}
                    type="button"
                    onClick={() => selectRegion(idx)}
                    onMouseEnter={() => setHoveredIdx(idx)}
                    onMouseLeave={() => setHoveredIdx(null)}
                    className={cn(
                      "w-full text-left rounded-lg px-2.5 py-2 border transition-all duration-200",
                      "hover:bg-white/[0.07] hover:border-white/15 active:scale-[0.99]",
                      "focus:outline-none focus-visible:ring-1 focus-visible:ring-white/40",
                      active
                        ? "bg-white/10 border-white/25 shadow-[0_0_16px_rgba(255,255,255,0.05)]"
                        : "ui-inset"
                    )}
                  >
                    <div className="text-[10px] font-semibold text-white/85 leading-snug mb-0.5">
                      {e.region}
                    </div>
                    <div className="flex justify-between items-baseline gap-2 mb-1.5">
                      <span className={cn("text-[10px] capitalize", active ? "text-white/70" : "text-white/45")}>
                        {e.label}
                      </span>
                      <span
                        className={cn(
                          "text-[11px] font-mono-data tabular-nums",
                          active ? "text-white" : "text-white/50"
                        )}
                      >
                        {e.pct}%
                      </span>
                    </div>
                    <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden pointer-events-none">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${barW}%`,
                          background: active
                            ? "linear-gradient(90deg, #ffffff, #a3a3a3)"
                            : "rgba(255,255,255,0.25)",
                          boxShadow: active ? "0 0 12px rgba(255,255,255,0.35)" : undefined,
                        }}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div
            className={`absolute inset-0 pointer-events-none ${
              compact
                ? "bg-[radial-gradient(circle_at_50%_42%,rgba(255,255,255,0.04),transparent_62%)]"
                : "bg-[radial-gradient(circle_at_50%_42%,rgba(255,255,255,0.04),transparent_62%)]"
            }`}
          />
          <canvas ref={canvasRef} className="block w-full h-full relative z-10" />
        </>
      )}

      {/* Title overlay — full / hero layout */}
      {!compact && !hero && (
        <div className="absolute top-2.5 left-2.5 z-20 flex items-center gap-1.5 pointer-events-none">
          <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)] animate-pulse" />
          <span className="text-[10px] font-semibold text-white/80">Neural Emotional Response</span>
          {dominantEmotion && (
            <span className="text-[9px] text-white/45 px-1.5 py-0.5 rounded-full bg-white/[0.05] capitalize">
              {dominantEmotion}
            </span>
          )}
        </div>
      )}

      {/* Legend — full layout only */}
      {!compact && !hero && (
        <div className="absolute bottom-3 left-3 z-20 ui-inset rounded-xl px-3 py-2.5 space-y-1.5 text-[10.5px] pointer-events-none max-w-[240px]">
          <div className="font-semibold text-white/90 text-[11px] mb-1">Brain regions</div>
          {EMOTION_REGIONS.map((r, idx) => (
            <div key={r.key} className="flex items-start gap-2">
              <span className="text-[9px] font-mono-data text-white/30 w-3 shrink-0">{idx + 1}</span>
              <div>
                <div className="font-medium text-white/85 leading-snug">{r.region}</div>
                <div className="text-white/40 capitalize text-[10px]">{r.label}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
