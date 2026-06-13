import React, { useEffect, useRef, useState } from "react";
import brainGlowImg from "../assets/realistic_brain_monochrome.png";

interface BrainSimulationProps {
  isActive: boolean;
  selectedGroups: string[];
  riskScore: number;
}

interface BrainNode {
  id: number;
  x: number;
  y: number;
  label: string;
  region: string;
  isHub?: boolean;
  group?: string;
  pulsePhase?: number;
  currentRadius?: number;
}

interface BrainEdge {
  from: number;
  to: number;
}

interface Particle {
  x: number;
  y: number;
  fromNode: number;
  toNode: number;
  progress: number;
  speed: number;
  color: string;
  size: number;
  isSignal: boolean;
  pulsePhase?: number;
}

interface Ripple {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  color: string;
  alpha: number;
}

interface FloatingText {
  x: number;
  y: number;
  text: string;
  color: string;
  alpha: number;
  speedY: number;
}

const brainNodes: BrainNode[] = [
  // Input Node (Left edge, Frontal lobe input)
  { id: 1, x: 0.18, y: 0.50, label: "Content Input Gate", region: "input" },
  
  // Frontal Lobe (Teens / Cognitive Processing)
  { id: 2, x: 0.28, y: 0.38, label: "Prefrontal Intake", region: "frontal" },
  { id: 3, x: 0.32, y: 0.25, label: "Attention Regulation", region: "frontal" },
  { id: 4, x: 0.42, y: 0.22, label: "Teens Hub", region: "frontal", isHub: true, group: "teens" },
  { id: 5, x: 0.35, y: 0.42, label: "Reward Estimation", region: "frontal" },
  
  // Temporal Lobe (Anxious Users / Emotional Amplification)
  { id: 6, x: 0.44, y: 0.54, label: "Amygdala Reactivity", region: "temporal" },
  { id: 7, x: 0.42, y: 0.68, label: "Anxious Users Hub", region: "temporal", isHub: true, group: "anxious" },
  { id: 8, x: 0.50, y: 0.62, label: "Stress Integration", region: "temporal" },
  { id: 9, x: 0.58, y: 0.68, label: "Vulnerability Filter", region: "temporal" },

  // Parietal Lobe (Caregivers / Empathy / Sensory Integration)
  { id: 10, x: 0.54, y: 0.30, label: "Sensory Hub", region: "parietal" },
  { id: 11, x: 0.65, y: 0.24, label: "Caregivers Hub", region: "parietal", isHub: true, group: "caregivers" },
  { id: 12, x: 0.70, y: 0.35, label: "Cognitive Empathy", region: "parietal" },
  { id: 13, x: 0.62, y: 0.45, label: "Perspective Synthesis", region: "parietal" },

  // Occipital Lobe (General Public / Spread Network)
  { id: 14, x: 0.82, y: 0.42, label: "Transmission Node", region: "occipital" },
  { id: 15, x: 0.85, y: 0.52, label: "General Public Hub", region: "occipital", isHub: true, group: "general" },
  { id: 16, x: 0.75, y: 0.48, label: "Viral Amplification", region: "occipital" },
  
  // Cerebellum (Action / Sharing Output)
  { id: 17, x: 0.78, y: 0.70, label: "Engagement Trigger", region: "cerebellum" },
  { id: 18, x: 0.70, y: 0.78, label: "Behavioral Mimicry", region: "cerebellum" },
  { id: 19, x: 0.62, y: 0.80, label: "Sentiment Echo", region: "cerebellum" }
];

const brainEdges: BrainEdge[] = [
  { from: 1, to: 2 }, { from: 1, to: 5 }, { from: 2, to: 3 }, { from: 2, to: 5 },
  { from: 3, to: 4 }, { from: 5, to: 4 }, { from: 5, to: 6 }, { from: 6, to: 7 },
  { from: 6, to: 8 }, { from: 7, to: 8 }, { from: 7, to: 9 }, { from: 8, to: 9 },
  { from: 4, to: 10 }, { from: 8, to: 10 }, { from: 10, to: 11 }, { from: 11, to: 12 },
  { from: 10, to: 13 }, { from: 12, to: 13 }, { from: 12, to: 16 }, { from: 13, to: 16 },
  { from: 16, to: 14 }, { from: 14, to: 15 }, { from: 16, to: 15 }, { from: 9, to: 19 },
  { from: 19, to: 18 }, { from: 18, to: 17 }, { from: 15, to: 17 }, { from: 17, to: 18 },
  { from: 8, to: 13 }, { from: 15, to: 16 }, { from: 6, to: 13 }
];

export function BrainSimulation({ isActive, selectedGroups, riskScore }: BrainSimulationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [hoveredNode, setHoveredNode] = useState<BrainNode | null>(null);
  
  // Load brain background image
  useEffect(() => {
    const img = new Image();
    img.src = brainGlowImg;
    img.onload = () => setBgImage(img);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let particles: Particle[] = [];
    let ripples: Ripple[] = [];
    let floatingTexts: FloatingText[] = [];
    let globalTime = 0;

    // Set canvas resolution
    const resizeCanvas = () => {
      const container = containerRef.current;
      if (!container) return;
      canvas.width = container.clientWidth * window.devicePixelRatio;
      canvas.height = container.clientHeight * window.devicePixelRatio;
      canvas.style.width = `${container.clientWidth}px`;
      canvas.style.height = `${container.clientHeight}px`;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Get color based on risk score
    const getRiskColor = (alpha = 1) => {
      if (riskScore >= 75) return `rgba(255, 255, 255, ${alpha})`; // Bright White
      if (riskScore >= 45) return `rgba(180, 180, 180, ${alpha})`; // Light Silver
      return `rgba(110, 110, 110, ${alpha})`; // Slate Grey
    };

    // Get color for specific group
    const getGroupColor = (group: string, alpha = 1) => {
      switch (group) {
        case "teens": return `rgba(255, 255, 255, ${alpha})`; // White
        case "anxious": return `rgba(210, 210, 210, ${alpha})`; // Light Grey
        case "caregivers": return `rgba(150, 150, 150, ${alpha})`; // Mid Grey
        case "general": return `rgba(100, 100, 100, ${alpha})`; // Dark Grey
        default: return `rgba(255, 255, 255, ${alpha})`;
      }
    };

    // Helper to translate brain relative coordinates to canvas space
    const getNodeCoords = (node: BrainNode, width: number, height: number) => {
      // Offset slightly to center the brain shape inside the viewport
      const offsetX = 50;
      const offsetY = 20;
      const brainWidth = width - offsetX * 2;
      const brainHeight = height - offsetY * 2;
      return {
        x: offsetX + node.x * brainWidth,
        y: offsetY + node.y * brainHeight
      };
    };

    // Initialize ambient particles
    const initAmbientParticles = () => {
      particles = [];
      // Spawn 15-20 subtle ambient particles traversing random edges
      for (let i = 0; i < 20; i++) {
        const edgeIndex = Math.floor(Math.random() * brainEdges.length);
        const edge = brainEdges[edgeIndex];
        const isForward = Math.random() > 0.5;
        particles.push({
          x: 0,
          y: 0,
          fromNode: isForward ? edge.from : edge.to,
          toNode: isForward ? edge.to : edge.from,
          progress: Math.random(),
          speed: 0.001 + Math.random() * 0.002,
          color: "rgba(200, 200, 200, 0.4)", // Translucent grey
          size: 1.5 + Math.random() * 2,
          isSignal: false
        });
      }
    };

    initAmbientParticles();

    // Trigger simulation signal blast
    const triggerSimulationBlast = () => {
      // Clear ambient particles to focus on the content flow
      particles = [];
      ripples = [];
      floatingTexts = [];

      // Determine active hubs based on selected groups
      const activeHubs = brainNodes.filter(n => n.isHub && n.group && selectedGroups.includes(n.group));
      
      // If no groups are selected, route to general public
      const destinations = activeHubs.length > 0 
        ? activeHubs 
        : [brainNodes.find(n => n.group === "general")!];

      // Spawn signal particles from Node 1 (Input Gate)
      destinations.forEach(hub => {
        // Find a path of edges from Node 1 to the Hub
        // To keep it simple and visual, we spawn multiple particles and direct them along connected nodes
        for (let burst = 0; burst < 12; burst++) {
          setTimeout(() => {
            const path = findPath(1, hub.id);
            if (path.length > 1) {
              spawnPathParticle(path, hub.group || "general", burst * 0.1);
            }
          }, burst * 150); // Staggered stream
        }
      });
    };

    // A simple DFS/BFS to find visual routing path
    const findPath = (startId: number, endId: number): number[] => {
      const queue: number[][] = [[startId]];
      const visited = new Set<number>();

      while (queue.length > 0) {
        const path = queue.shift()!;
        const node = path[path.length - 1];

        if (node === endId) return path;

        if (!visited.has(node)) {
          visited.add(node);
          // Get neighbors
          const neighbors = brainEdges
            .filter(e => e.from === node || e.to === node)
            .map(e => e.from === node ? e.to : e.from);

          for (const neighbor of neighbors) {
            queue.push([...path, neighbor]);
          }
        }
      }
      return [];
    };

    // Spawn a particle that follows a list of nodes
    const spawnPathParticle = (nodePath: number[], group: string, delayOffset: number) => {
      let currentSegment = 0;
      const spawnNextSegment = () => {
        if (currentSegment >= nodePath.length - 1) {
          // Arrived at target hub! Create a ripple and float text
          const targetNode = brainNodes.find(n => n.id === nodePath[nodePath.length - 1]);
          if (targetNode) {
            const coords = getNodeCoords(targetNode, canvas.width / window.devicePixelRatio, canvas.height / window.devicePixelRatio);
            const groupColor = getGroupColor(group);
            
            // Add a ripple
            ripples.push({
              x: coords.x,
              y: coords.y,
              radius: 5,
              maxRadius: 40 + Math.random() * 30,
              color: groupColor,
              alpha: 1
            });

            // Add emotional text floating up
            const emotionalKeywords: Record<string, string[]> = {
              teens: ["Impulsive Sharing", "Peer Influence", "Dopamine Response", "Identity Shift"],
              anxious: ["Panic Trigger", "Stress Spike", "Rumination Loop", "Hypervigilance"],
              caregivers: ["Hyper-Vigilance", "Protective Instinct", "Burnout Risk", "Empathy Fatigue"],
              general: ["Passive Spread", "Liking Behavior", "Muted Concern", "Echo Chamber"]
            };

            const keywords = emotionalKeywords[group] || ["Reaction Triggered"];
            const text = keywords[Math.floor(Math.random() * keywords.length)];
            
            floatingTexts.push({
              x: coords.x + (Math.random() * 30 - 15),
              y: coords.y - 10,
              text,
              color: groupColor,
              alpha: 1,
              speedY: -0.5 - Math.random() * 0.8
            });
          }
          return;
        }

        const fromId = nodePath[currentSegment];
        const toId = nodePath[currentSegment + 1];
        
        const speed = 0.03 + Math.random() * 0.02; // Speed of traversal
        const color = getGroupColor(group);

        const p: Particle = {
          x: 0,
          y: 0,
          fromNode: fromId,
          toNode: toId,
          progress: 0,
          speed,
          color,
          size: 3 + Math.random() * 3,
          isSignal: true
        };

        particles.push(p);

        // Track when this segment finishes
        const checkProgress = () => {
          if (p.progress >= 1) {
            // Remove particle
            particles = particles.filter(pt => pt !== p);
            currentSegment++;
            spawnNextSegment();
          } else {
            requestAnimationFrame(checkProgress);
          }
        };
        checkProgress();
      };

      spawnNextSegment();
    };

    if (isActive) {
      triggerSimulationBlast();
    }

    // Canvas drawing loop
    const draw = () => {
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;
      globalTime += 0.02;

      ctx.clearRect(0, 0, width, height);

      // 1. Draw glowing background brain image
      if (bgImage) {
        ctx.save();
        ctx.globalAlpha = 0.28; // Increased slightly for ultra-realistic detail visibility
        ctx.globalCompositeOperation = "screen";
        
        // Center-fit the image
        const imgWidth = bgImage.width;
        const imgHeight = bgImage.height;
        const scale = Math.min(width / imgWidth, height / imgHeight) * 0.95;
        const w = imgWidth * scale;
        const h = imgHeight * scale;
        const x = (width - w) / 2;
        const y = (height - h) / 2;
        
        ctx.drawImage(bgImage, x, y, w, h);
        ctx.restore();
      }

      // 2. Draw Edges
      ctx.lineWidth = 1;
      brainEdges.forEach(edge => {
        const fromNode = brainNodes.find(n => n.id === edge.from)!;
        const toNode = brainNodes.find(n => n.id === edge.to)!;
        const from = getNodeCoords(fromNode, width, height);
        const to = getNodeCoords(toNode, width, height);

        // Check if this connection belongs to an active path
        const isActiveEdge = isActive && (
          selectedGroups.some(group => {
            const isFromActiveHub = fromNode.isHub && fromNode.group === group;
            const isToActiveHub = toNode.isHub && toNode.group === group;
            return isFromActiveHub || isToActiveHub;
          }) || 
          (fromNode.region === "input" || toNode.region === "input")
        );

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);

        if (isActiveEdge) {
          // Glow active pathways
          ctx.strokeStyle = getRiskColor(0.35 + Math.sin(globalTime * 3) * 0.1);
          ctx.lineWidth = 1.5;
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      });

      // 3. Draw Ripples
      ripples.forEach((ripple, idx) => {
        ripple.radius += 1.2;
        ripple.alpha = 1 - (ripple.radius / ripple.maxRadius);
        if (ripple.alpha <= 0) {
          ripples.splice(idx, 1);
          return;
        }

        ctx.save();
        ctx.beginPath();
        ctx.arc(ripple.x, ripple.y, ripple.radius, 0, Math.PI * 2);
        ctx.strokeStyle = ripple.color.replace("1)", `${ripple.alpha})`);
        ctx.lineWidth = 2;
        ctx.shadowBlur = 10;
        ctx.shadowColor = ripple.color;
        ctx.stroke();
        ctx.restore();
      });

      // 4. Update and Draw Particles
      particles.forEach((p, idx) => {
        p.progress += p.speed;

        const fromNode = brainNodes.find(n => n.id === p.fromNode)!;
        const toNode = brainNodes.find(n => n.id === p.toNode)!;
        const from = getNodeCoords(fromNode, width, height);
        const to = getNodeCoords(toNode, width, height);

        // Linear interpolation
        p.x = from.x + (to.x - from.x) * p.progress;
        p.y = from.y + (to.y - from.y) * p.progress;

        // Draw particle
        ctx.save();
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowBlur = p.isSignal ? 10 : 0;
        ctx.shadowColor = p.color;
        ctx.fill();
        ctx.restore();

        // Ambient loop recycling
        if (!p.isSignal && p.progress >= 1) {
          p.progress = 0;
          const swap = p.fromNode;
          p.fromNode = p.toNode;
          p.toNode = swap;
        }
      });

      // 5. Draw Nodes
      brainNodes.forEach(node => {
        const { x, y } = getNodeCoords(node, width, height);
        const isHovered = hoveredNode?.id === node.id;

        // Hub status
        const isGroupActive = node.group && selectedGroups.includes(node.group);

        ctx.save();
        ctx.beginPath();

        let baseRadius = node.isHub ? 8 : 4;
        let radius = baseRadius;
        let color = "rgba(255, 255, 255, 0.2)";
        let glowColor = "rgba(255, 255, 255, 0)";

        if (node.isHub) {
          color = getGroupColor(node.group || "", 0.6);
          glowColor = getGroupColor(node.group || "", 0.8);
          // Pulsing animation
          const pulseSpeed = isGroupActive ? 4 : 1.5;
          const pulseScale = isGroupActive ? 2.5 : 1.2;
          radius = baseRadius + Math.sin(globalTime * pulseSpeed) * pulseScale;
        } else if (node.region === "input") {
          color = "rgba(255, 255, 255, 0.7)";
          glowColor = "rgba(255, 255, 255, 0.4)";
          radius = baseRadius + Math.sin(globalTime * 2.5) * 1.5;
        }

        // Draw core node
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = isHovered ? "#ffffff" : color;
        ctx.shadowBlur = node.isHub || isHovered ? 15 : 0;
        ctx.shadowColor = isHovered ? "#ffffff" : glowColor;
        ctx.fill();

        // Outermost border ring for active hubs
        if (node.isHub && isGroupActive) {
          ctx.beginPath();
          ctx.arc(x, y, radius + 5, 0, Math.PI * 2);
          ctx.strokeStyle = getGroupColor(node.group || "", 0.3);
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Display hub labels
        if (node.isHub) {
          ctx.fillStyle = isGroupActive ? "#ffffff" : "rgba(255, 255, 255, 0.4)";
          ctx.font = "bold 10px 'Outfit', sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(node.label, x, y - radius - 8);
        }

        ctx.restore();
      });

      // 6. Draw Hover details
      if (hoveredNode) {
        const { x, y } = getNodeCoords(hoveredNode, width, height);
        ctx.save();
        ctx.fillStyle = "rgba(10, 10, 11, 0.85)";
        ctx.strokeStyle = hoveredNode.isHub ? getGroupColor(hoveredNode.group || "") : "rgba(255,255,255,0.15)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x - 70, y + 12, 140, 36, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px 'Outfit', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(hoveredNode.label, x, y + 24);
        
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.font = "8px 'Outfit', sans-serif";
        ctx.fillText(hoveredNode.region.toUpperCase() + " REGION", x, y + 38);
        ctx.restore();
      }

      // 7. Update and Draw Floating Texts
      floatingTexts.forEach((ft, idx) => {
        ft.y += ft.speedY;
        ft.alpha -= 0.008; // Fade out slowly
        if (ft.alpha <= 0) {
          floatingTexts.splice(idx, 1);
          return;
        }

        ctx.save();
        ctx.fillStyle = ft.color.replace("1)", `${ft.alpha})`);
        ctx.font = "bold 9px 'Outfit', sans-serif";
        ctx.textAlign = "center";
        ctx.shadowBlur = 5;
        ctx.shadowColor = ft.color;
        ctx.fillText(ft.text, ft.x, ft.y);
        ctx.restore();
      });

      animationId = requestAnimationFrame(draw);
    };

    // Hover mouse detection
    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;

      let found: BrainNode | null = null;
      for (const node of brainNodes) {
        const { x, y } = getNodeCoords(node, width, height);
        const dist = Math.hypot(mouseX - x, mouseY - y);
        const hoverRadius = node.isHub ? 15 : 8;
        if (dist < hoverRadius) {
          found = node;
          break;
        }
      }
      setHoveredNode(found);
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    draw();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      canvas.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationId);
    };
  }, [isActive, selectedGroups, riskScore, bgImage, hoveredNode]);

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-black/40 rounded-2xl border border-white/[0.04]">
      {/* Background radial glow */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.02)_0%,transparent_70%)] pointer-events-none" />
      
      {/* Canvas for simulation */}
      <canvas ref={canvasRef} className="block w-full h-full relative z-10" />

      {/* Floating UI Legend overlay */}
      <div className="absolute bottom-4 left-4 z-20 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg p-2.5 space-y-1.5 text-xs text-white/60">
        <div className="font-semibold text-white/90 border-b border-white/5 pb-1 mb-1.5">Simulation Legend</div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-white shadow-[0_0_8px_#ffffff]" />
          <span>Teens Archetype Hub</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#d2d2d2] shadow-[0_0_8px_#d2d2d2]" />
          <span>Anxious Users Hub</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#969696] shadow-[0_0_8px_#969696]" />
          <span>Caregivers Hub</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-[#646464] shadow-[0_0_8px_#646464]" />
          <span>General Public Hub</span>
        </div>
        <div className="pt-1 text-[9px] text-white/40 italic">
          Hover over nodes to inspect neural region
        </div>
      </div>
    </div>
  );
}
