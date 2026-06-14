import { useEffect, useRef } from "react";
import * as THREE from "three";

const FRAGMENT_SHADER = `
precision highp float;
uniform vec2 resolution;
uniform float time;

float random(float x) {
  return fract(sin(x) * 1e4);
}

float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main(void) {
  // True screen center (aspect-correct)
  vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy) / min(resolution.x, resolution.y);

  // Horizontal oval — bright core clustered in the middle (reference eye shape)
  float dist = length(uv * vec2(1.35, 0.88));

  // Vertical mosaic columns
  vec2 mosaic = uv;
  vec2 fMosaicScal = vec2(4.0, 2.0);
  vec2 vScreenSize = vec2(256.0, 256.0);
  mosaic.x = floor(mosaic.x * vScreenSize.x / fMosaicScal.x) / (vScreenSize.x / fMosaicScal.x);
  mosaic.y = floor(mosaic.y * vScreenSize.y / fMosaicScal.y) / (vScreenSize.y / fMosaicScal.y);

  float t = time * 0.06 + random(mosaic.x) * 0.4;
  float lineWidth = 0.0028;

  vec3 color = vec3(0.0);
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 5; i++) {
      float band = fract(t - 0.01 * float(j) + float(i) * 0.01);
      color[j] += lineWidth * float(i * i + 1) / abs(band - dist + 0.0006);
    }
  }

  // Grayscale output — no chromatic tint
  color = vec3(dot(color, vec3(0.299, 0.587, 0.114)));
  color *= 3.4;
  color = color / (color + vec3(0.28));

  // Soft radial falloff — keeps edges dark, core bright
  float vignette = smoothstep(1.05, 0.08, dist);
  color *= 0.35 + 0.65 * vignette;

  gl_FragColor = vec4(color, 1.0);
}
`;

const VERTEX_SHADER = `
void main() {
  gl_Position = vec4(position, 1.0);
}
`;

export function ShaderAnimation() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const scene = new THREE.Scene();
    const geometry = new THREE.PlaneGeometry(2, 2);

    const uniforms = {
      time: { value: 1.0 },
      resolution: { value: new THREE.Vector2() },
    };

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      renderer.setSize(width, height, false);
      uniforms.resolution.value.set(renderer.domElement.width, renderer.domElement.height);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    let animationId = 0;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      uniforms.time.value += 0.05;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationId);
      ro.disconnect();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 w-full h-full overflow-hidden"
      aria-hidden
    />
  );
}
