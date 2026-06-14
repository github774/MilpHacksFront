import { useEffect, useRef, useState } from "react";
import type { Agent, PersonaEntry, ShareEdge } from "../api";
import { ACTION_META, fmtPct, topEmotion } from "../lib/simulation";
import { REACTION_STYLE } from "../lib/theme";

interface NetworkGraphProps {
  agents: Agent[];
  shareEdges: ShareEdge[];
  personas: Record<string, PersonaEntry>;
  runKey: number;
  onSelect: (agent: Agent | null) => void;
  selectedExposureId: number | null;
  /** Full simulation agent count (when graph is sampled for performance). */
  displayTotal?: number;
  sampled?: boolean;
}

interface GNode {
  exposureId: number;
  agent: Agent;
  x: number;
  y: number;
  vx: number;
  vy: number;
  parent: number | null;
  wave: number;
  revealAt: number;
  revealed: boolean;
  bornAt: number;
  radius: number;
  color: string;
  glow: string;
  shareCount: number;
}

const ACTION_STYLE: Record<string, { color: string; glow: string; label: string }> = {
  like: { ...REACTION_STYLE.like, label: "Liked" },
  like_share: { ...REACTION_STYLE.like_share, label: "Liked + Shared" },
  neutral: { ...REACTION_STYLE.neutral, label: "Passed" },
  dislike: { ...REACTION_STYLE.dislike, label: "Disliked" },
  dislike_share: { ...REACTION_STYLE.dislike_share, label: "Disliked + Shared" },
};

const SEED_COLOR = "#ffffff";
const SEED_GLOW = "rgba(255,255,255,0.9)";

export function NetworkGraph({
  agents,
  shareEdges,
  personas,
  runKey,
  onSelect,
  selectedExposureId,
  displayTotal,
  sampled = false,
}: NetworkGraphProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const nodesRef = useRef<GNode[]>([]);
  const nodeByExposure = useRef<Map<number, GNode>>(new Map());
  const transformRef = useRef({ scale: 1, x: 0, y: 0 });
  const manualRef = useRef(false);
  const startRef = useRef(0);
  const pointerRef = useRef<{ x: number; y: number; down: boolean; moved: boolean; lastX: number; lastY: number }>({
    x: 0, y: 0, down: false, moved: false, lastX: 0, lastY: 0,
  });
  const selectedRef = useRef<number | null>(selectedExposureId);

  const [hover, setHover] = useState<{ node: GNode; sx: number; sy: number } | null>(null);
  const [stats, setStats] = useState({ shown: 0, total: 0, wave: 0, shares: 0 });

  useEffect(() => {
    selectedRef.current = selectedExposureId;
  }, [selectedExposureId]);

  // Build the graph topology whenever the simulation data changes.
  useEffect(() => {
    if (!agents.length) {
      nodesRef.current = [];
      nodeByExposure.current = new Map();
      return;
    }

    // Group exposures by persona so we can resolve "who shared to me".
    const byPersona = new Map<number, Agent[]>();
    for (const a of agents) {
      const arr = byPersona.get(a.persona_index) || [];
      arr.push(a);
      byPersona.set(a.persona_index, arr);
    }
    for (const arr of byPersona.values()) arr.sort((a, b) => a.exposure_id - b.exposure_id);

    const shareCounts = new Map<number, number>();
    for (const e of shareEdges) {
      shareCounts.set(e.from_exposure_id, (shareCounts.get(e.from_exposure_id) || 0) + 1);
    }

    const resolveParent = (a: Agent): number | null => {
      if (a.exposed_by_index == null) return null;
      const candidates = byPersona.get(a.exposed_by_index);
      if (!candidates) return null;
      let best: Agent | null = null;
      for (const c of candidates) {
        if (c.exposure_id >= a.exposure_id) break;
        if (c.wave === a.wave - 1) best = c;
      }
      if (!best) {
        for (const c of candidates) {
          if (c.exposure_id >= a.exposure_id) break;
          best = c;
        }
      }
      return best ? best.exposure_id : null;
    };

    const sorted = [...agents].sort(
      (a, b) => a.wave - b.wave || a.exposure_id - b.exposure_id
    );

    const N = sorted.length;
    const fullTotal = displayTotal ?? N;
    const cadence =
      N > 3000 ? 2 : N > 1200 ? 6 : Math.max(8, Math.min(50, 4000 / Math.max(1, N)));
    const wavePause = N > 1200 ? 80 : 280;

    const nodes: GNode[] = [];
    const map = new Map<number, GNode>();
    const cx = 0;
    const cy = 0;

    sorted.forEach((a, i) => {
      const style = a.wave === 0
        ? { color: SEED_COLOR, glow: SEED_GLOW }
        : ACTION_STYLE[a.sampled_action] || ACTION_STYLE.neutral;
      const sc = shareCounts.get(a.exposure_id) || 0;
      // Seeds spread on a wide ring, others spawn near parent (set later).
      const ang = (i / Math.max(1, N)) * Math.PI * 2;
      const node: GNode = {
        exposureId: a.exposure_id,
        agent: a,
        x: cx + Math.cos(ang) * (a.wave === 0 ? 120 : 40) + (Math.random() - 0.5) * 30,
        y: cy + Math.sin(ang) * (a.wave === 0 ? 120 : 40) + (Math.random() - 0.5) * 30,
        vx: 0,
        vy: 0,
        parent: null,
        wave: a.wave,
        revealAt: i * cadence + a.wave * wavePause,
        revealed: false,
        bornAt: 0,
        radius: 4 + Math.min(7, sc * 1.6) + (a.wave === 0 ? 1.5 : 0),
        color: style.color,
        glow: style.glow,
        shareCount: sc,
      };
      nodes.push(node);
      map.set(a.exposure_id, node);
    });

    // Resolve parents and seed child positions near their parent.
    for (const node of nodes) {
      const p = resolveParent(node.agent);
      node.parent = p;
      if (p != null) {
        const parent = map.get(p);
        if (parent) {
          node.x = parent.x + (Math.random() - 0.5) * 50;
          node.y = parent.y + (Math.random() - 0.5) * 50;
        }
      }
    }

    nodesRef.current = nodes;
    nodeByExposure.current = map;
    transformRef.current = { scale: 1, x: 0, y: 0 };
    manualRef.current = false;
    startRef.current = performance.now();
    setStats({ shown: 0, total: fullTotal, wave: 0, shares: shareEdges.length });
  }, [agents, shareEdges, runKey, displayTotal]);

  // Render + physics loop.
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let W = 0;
    let H = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      W = container.clientWidth;
      H = container.clientHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    const physics = (revealed: GNode[]) => {
      const cell = 64;
      const grid = new Map<string, GNode[]>();
      for (const n of revealed) {
        const gx = Math.floor(n.x / cell);
        const gy = Math.floor(n.y / cell);
        const key = `${gx},${gy}`;
        const arr = grid.get(key) || [];
        arr.push(n);
        grid.set(key, arr);
      }

      // Repulsion (local, via spatial hash).
      for (const n of revealed) {
        const gx = Math.floor(n.x / cell);
        const gy = Math.floor(n.y / cell);
        for (let dx = -1; dx <= 1; dx++) {
          for (let dy = -1; dy <= 1; dy++) {
            const arr = grid.get(`${gx + dx},${gy + dy}`);
            if (!arr) continue;
            for (const m of arr) {
              if (m === n) continue;
              let ddx = n.x - m.x;
              let ddy = n.y - m.y;
              let d2 = ddx * ddx + ddy * ddy;
              if (d2 === 0) {
                ddx = Math.random() - 0.5;
                ddy = Math.random() - 0.5;
                d2 = 0.01;
              }
              if (d2 < 60 * 60) {
                const f = 520 / d2;
                const d = Math.sqrt(d2);
                n.vx += (ddx / d) * f;
                n.vy += (ddy / d) * f;
              }
            }
          }
        }
      }

      // Springs to parent.
      for (const n of revealed) {
        if (n.parent == null) continue;
        const p = nodeByExposure.current.get(n.parent);
        if (!p || !p.revealed) continue;
        const dx = p.x - n.x;
        const dy = p.y - n.y;
        const dist = Math.hypot(dx, dy) || 0.01;
        const target = 46;
        const f = (dist - target) * 0.012;
        n.vx += (dx / dist) * f;
        n.vy += (dy / dist) * f;
        p.vx -= (dx / dist) * f * 0.5;
        p.vy -= (dy / dist) * f * 0.5;
      }

      // Gentle gravity toward origin keeps the cloud compact.
      for (const n of revealed) {
        n.vx += -n.x * 0.0016;
        n.vy += -n.y * 0.0016;
        n.vx *= 0.86;
        n.vy *= 0.86;
        const sp = Math.hypot(n.vx, n.vy);
        if (sp > 6) {
          n.vx = (n.vx / sp) * 6;
          n.vy = (n.vy / sp) * 6;
        }
        n.x += n.vx;
        n.y += n.vy;
      }
    };

    const fitView = (revealed: GNode[]) => {
      if (manualRef.current || revealed.length === 0) return;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const n of revealed) {
        if (n.x < minX) minX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.x > maxX) maxX = n.x;
        if (n.y > maxY) maxY = n.y;
      }
      const pad = 90;
      const bw = maxX - minX + pad * 2;
      const bh = maxY - minY + pad * 2;
      const targetScale = Math.min(2.2, Math.max(0.25, Math.min(W / bw, H / bh)));
      const cxw = (minX + maxX) / 2;
      const cyw = (minY + maxY) / 2;
      const t = transformRef.current;
      const tx = W / 2 - cxw * targetScale;
      const ty = H / 2 - cyw * targetScale;
      t.scale += (targetScale - t.scale) * 0.06;
      t.x += (tx - t.x) * 0.06;
      t.y += (ty - t.y) * 0.06;
    };

    let lastStat = 0;
    const draw = (now: number) => {
      const nodes = nodesRef.current;
      const t = transformRef.current;

      let curWave = 0;
      let shownCount = 0;
      const elapsed = now - startRef.current;
      for (const n of nodes) {
        if (!n.revealed && elapsed >= n.revealAt) {
          n.revealed = true;
          n.bornAt = now;
        }
        if (n.revealed) {
          shownCount++;
          if (n.wave > curWave) curWave = n.wave;
        }
      }
      const revealed = nodes.filter((n) => n.revealed);

      if (revealed.length <= 1800) physics(revealed);
      fitView(revealed);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // subtle vignette grid
      ctx.save();
      ctx.translate(t.x, t.y);
      ctx.scale(t.scale, t.scale);

      // Edges
      ctx.lineWidth = 1 / t.scale;
      for (const n of revealed) {
        if (n.parent == null) continue;
        const p = nodeByExposure.current.get(n.parent);
        if (!p || !p.revealed) continue;
        const age = Math.min(1, (now - n.bornAt) / 600);
        const sharer = n.agent.exposed_by_share_type?.includes("dislike");
        ctx.strokeStyle = sharer
          ? `rgba(115,115,115,${0.06 + age * 0.16})`
          : `rgba(255,255,255,${0.05 + age * 0.18})`;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(n.x, n.y);
        ctx.stroke();

        // traveling signal pulse on fresh edges
        if (age < 1) {
          const px = p.x + (n.x - p.x) * age;
          const py = p.y + (n.y - p.y) * age;
          ctx.beginPath();
          ctx.arc(px, py, 2 / t.scale, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(255,255,255,0.8)";
          ctx.fill();
        }
      }

      // Nodes
      const sel = selectedRef.current;
      for (const n of revealed) {
        const pop = Math.min(1, (now - n.bornAt) / 320);
        const r = n.radius * (0.3 + 0.7 * pop);
        const isSel = sel === n.exposureId;

        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.shadowBlur = (n.shareCount > 0 ? 18 : 8) + (isSel ? 16 : 0);
        ctx.shadowColor = n.glow;
        ctx.globalAlpha = 0.92 * pop + 0.08;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.shadowBlur = 0;

        if (n.wave === 0) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 2.5 / t.scale + 2, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,255,255,0.35)";
          ctx.lineWidth = 1.2 / t.scale;
          ctx.stroke();
        }
        if (isSel) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r + 7, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.lineWidth = 1.5 / t.scale;
          ctx.stroke();
        }
      }
      ctx.restore();

      if (now - lastStat > 120) {
        lastStat = now;
        setStats((s) =>
          s.shown === shownCount && s.wave === curWave
            ? s
            : { ...s, shown: shownCount, wave: curWave }
        );
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    // ---- interaction ----
    const screenToWorld = (sx: number, sy: number) => {
      const t = transformRef.current;
      return { x: (sx - t.x) / t.scale, y: (sy - t.y) / t.scale };
    };
    const pick = (sx: number, sy: number): GNode | null => {
      const w = screenToWorld(sx, sy);
      let best: GNode | null = null;
      let bestD = Infinity;
      for (const n of nodesRef.current) {
        if (!n.revealed) continue;
        const d = Math.hypot(n.x - w.x, n.y - w.y);
        const hit = n.radius + 6 / transformRef.current.scale;
        if (d < hit && d < bestD) {
          bestD = d;
          best = n;
        }
      }
      return best;
    };

    const onMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const pt = pointerRef.current;
      if (pt.down) {
        const ddx = sx - pt.lastX;
        const ddy = sy - pt.lastY;
        if (Math.abs(ddx) + Math.abs(ddy) > 2) pt.moved = true;
        transformRef.current.x += ddx;
        transformRef.current.y += ddy;
        manualRef.current = true;
        pt.lastX = sx;
        pt.lastY = sy;
        return;
      }
      const node = pick(sx, sy);
      canvas.style.cursor = node ? "pointer" : "grab";
      if (node) setHover({ node, sx, sy });
      else setHover(null);
    };
    const onDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const pt = pointerRef.current;
      pt.down = true;
      pt.moved = false;
      pt.lastX = e.clientX - rect.left;
      pt.lastY = e.clientY - rect.top;
    };
    const onUp = (e: MouseEvent) => {
      const pt = pointerRef.current;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      if (pt.down && !pt.moved) {
        const node = pick(sx, sy);
        onSelect(node ? node.agent : null);
      }
      pt.down = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const t = transformRef.current;
      const factor = Math.exp(-e.deltaY * 0.0012);
      const newScale = Math.min(4, Math.max(0.15, t.scale * factor));
      const wx = (sx - t.x) / t.scale;
      const wy = (sy - t.y) / t.scale;
      t.scale = newScale;
      t.x = sx - wx * newScale;
      t.y = sy - wy * newScale;
      manualRef.current = true;
    };
    const onLeave = () => {
      setHover(null);
      pointerRef.current.down = false;
    };

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("mouseleave", onLeave);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mousedown", onDown);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [onSelect]);

  const hoverAgent = hover?.node.agent;
  const hoverAction = hoverAgent ? ACTION_META[hoverAgent.sampled_action] : null;
  const hoverEmotion = hoverAgent ? topEmotion(hoverAgent.emotion_probs) : null;

  const growth = stats.total ? Math.round((stats.shown / agents.length) * 100) : 0;

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden rounded-2xl">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_40%,rgba(255,255,255,0.04),transparent_60%)] pointer-events-none" />
      <canvas ref={canvasRef} className="block w-full h-full relative z-10" />

      <div className="absolute top-3 left-3 z-20 flex flex-wrap items-center gap-2 pointer-events-none max-w-[90%]">
        <div className="px-2.5 py-1.5 rounded-lg ui-inset text-[11px] text-white/85 font-medium tabular-nums">
          {stats.shown.toLocaleString()}
          {sampled ? ` shown · ${stats.total.toLocaleString()} simulated` : ` / ${stats.total.toLocaleString()}`}
        </div>
        <div className="px-2.5 py-1.5 rounded-lg ui-inset text-[11px] text-white/70 font-medium tabular-nums">
          wave {stats.wave}
        </div>
        {growth < 100 && agents.length > 80 && (
          <div className="px-2.5 py-1.5 rounded-lg bg-white/10 backdrop-blur border border-white/15 text-[11px] text-white font-semibold tabular-nums">
            growing {growth}%
          </div>
        )}
        {sampled && (
          <div className="px-2.5 py-1.5 rounded-lg ui-inset text-[10px] text-white/75">
            sampled view — metrics use full run
          </div>
        )}
      </div>

      {/* Compact legend */}
      <div className="absolute bottom-3 left-3 z-20 ui-inset rounded-lg px-2.5 py-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-white/65 pointer-events-none max-w-[85%]">
        {[
          { c: SEED_COLOR, l: "Seed" },
          { c: ACTION_STYLE.like_share.color, l: "Share" },
          { c: ACTION_STYLE.like.color, l: "Like" },
          { c: ACTION_STYLE.dislike.color, l: "Dislike" },
        ].map((it) => (
          <span key={it.l} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: it.c }} />
            {it.l}
          </span>
        ))}
        <span className="text-white/30">· click node for vote</span>
      </div>

      {/* Hover tooltip */}
      {hover && hoverAgent && hoverAction && hoverEmotion && (
        <div
          className="absolute z-30 pointer-events-none px-3 py-2.5 rounded-lg bg-black/90 backdrop-blur border border-white/15 shadow-xl max-w-[240px]"
          style={{
            left: Math.min(
              Math.max(8, hover.sx + 12),
              (containerRef.current?.clientWidth || 400) - 248
            ),
            top: Math.min(
              Math.max(8, hover.sy + 12),
              (containerRef.current?.clientHeight || 300) - 100
            ),
          }}
        >
          <div className="text-[12px] font-semibold text-white capitalize leading-tight">
            {personas[String(hoverAgent.persona_index)]?.occupation?.replace(/_/g, " ") ||
              hoverAgent.archetype}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span
              className="text-[10px] font-semibold px-2 py-0.5 rounded-full"
              style={{ background: `${hoverAction.color}22`, color: hoverAction.color }}
            >
              {hoverAction.label}
            </span>
            <span className="text-[10px] text-white/70 capitalize">
              {hoverEmotion.key} {fmtPct(hoverEmotion.value, 0)}
            </span>
          </div>
          <div className="text-[10px] text-white/40 mt-1">wave {hoverAgent.wave} · click to inspect</div>
        </div>
      )}
    </div>
  );
}
