import { lazy, Suspense } from "react";

const Spline = lazy(() => import("@splinetool/react-spline"));

const SPLINE_SCENE = "https://prod.spline.design/us3ALejTXl6usHZ7/scene.splinecode";

/** Full-viewport Spline scene + vignette — background only, no pointer capture. */
export function HeroSplineBackground() {
  return (
    <div
      className="fixed inset-0 z-0 overflow-hidden pointer-events-none"
      aria-hidden
    >
      <Suspense fallback={null}>
        <Spline
          scene={SPLINE_SCENE}
          style={{
            width: "100%",
            height: "100%",
            minHeight: "100vh",
          }}
        />
      </Suspense>
      <div
        className="absolute inset-0 bg-black/40"
        style={{
          background: `
            linear-gradient(to right, rgba(0, 0, 0, 0.55), transparent 28%, transparent 72%, rgba(0, 0, 0, 0.55)),
            linear-gradient(to bottom, transparent 55%, rgba(0, 0, 0, 0.5))
          `,
        }}
      />
    </div>
  );
}
