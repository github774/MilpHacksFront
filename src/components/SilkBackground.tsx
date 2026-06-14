import { useEffect, useRef } from "react";

/** Animated silk texture background for results / non-hero views. */
export function SilkBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let time = 0;
    let animationId = 0;
    const speed = 0.02;
    const scale = 2;
    const noiseIntensity = 0.8;

    const noise = (x: number, y: number) => {
      const G = 2.71828;
      const rx = G * Math.sin(G * x);
      const ry = G * Math.sin(G * y);
      return (rx * ry * (1 + x)) % 1;
    };

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    const animate = () => {
      const { width, height } = canvas;

      const gradient = ctx.createLinearGradient(0, 0, width * 0.85, height);
      gradient.addColorStop(0, "#121018");
      gradient.addColorStop(0.32, "#1c1824");
      gradient.addColorStop(0.58, "#16141e");
      gradient.addColorStop(0.82, "#1a1622");
      gradient.addColorStop(1, "#0e0c14");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      const imageData = ctx.createImageData(width, height);
      const { data } = imageData;

      for (let x = 0; x < width; x += 2) {
        for (let y = 0; y < height; y += 2) {
          const u = (x / width) * scale;
          const v = (y / height) * scale;
          const tOffset = speed * time;
          const tex_x = u;
          const tex_y = v + 0.03 * Math.sin(8.0 * tex_x - tOffset);

          const pattern =
            0.6 +
            0.4 *
              Math.sin(
                5.0 *
                  (tex_x +
                    tex_y +
                    Math.cos(3.0 * tex_x + 5.0 * tex_y) +
                    0.02 * tOffset) +
                  Math.sin(20.0 * (tex_x + tex_y - 0.1 * tOffset))
              );

          const rnd = noise(x, y);
          const intensity = Math.max(0, pattern - (rnd / 15.0) * noiseIntensity) * 0.55;
          const hue = 0.12 * Math.sin(tex_x * 2.4 + tex_y * 1.6 + tOffset * 0.04);

          const r = Math.floor((118 + hue * 18) * intensity);
          const g = Math.floor((110 - hue * 6) * intensity);
          const b = Math.floor((136 + hue * 28) * intensity);

          for (let dx = 0; dx < 2; dx++) {
            for (let dy = 0; dy < 2; dy++) {
              const px = x + dx;
              const py = y + dy;
              if (px >= width || py >= height) continue;
              const index = (py * width + px) * 4;
              data[index] = r;
              data[index + 1] = g;
              data[index + 2] = b;
              data[index + 3] = 255;
            }
          }
        }
      }

      ctx.putImageData(imageData, 0, 0);

      const overlayGradient = ctx.createRadialGradient(
        width * 0.45,
        height * 0.35,
        0,
        width * 0.5,
        height * 0.5,
        Math.max(width, height) * 0.65
      );
      overlayGradient.addColorStop(0, "rgba(0, 0, 0, 0.18)");
      overlayGradient.addColorStop(0.45, "rgba(0, 0, 0, 0.32)");
      overlayGradient.addColorStop(1, "rgba(0, 0, 0, 0.62)");
      ctx.fillStyle = overlayGradient;
      ctx.fillRect(0, 0, width, height);

      const cornerGlow = ctx.createRadialGradient(0, 0, 0, 0, 0, width * 0.55);
      cornerGlow.addColorStop(0, "rgba(100, 120, 200, 0.06)");
      cornerGlow.addColorStop(1, "transparent");
      ctx.fillStyle = cornerGlow;
      ctx.fillRect(0, 0, width, height);

      time += 1;
      animationId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none" aria-hidden>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-75" />
      <div
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(ellipse 90% 70% at 0% 0%, rgba(110, 130, 220, 0.09) 0%, transparent 52%),
            radial-gradient(ellipse 80% 60% at 100% 100%, rgba(180, 120, 240, 0.07) 0%, transparent 48%),
            linear-gradient(168deg, rgba(0, 0, 0, 0.38) 0%, rgba(0, 0, 0, 0.22) 42%, rgba(0, 0, 0, 0.36) 100%)
          `,
        }}
      />
      <div className="absolute inset-0 bg-gradient-to-b from-black/42 via-transparent to-black/48" />
      <div className="absolute inset-0 bg-gradient-to-r from-black/30 via-transparent to-black/30" />
    </div>
  );
}
