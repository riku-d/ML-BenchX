import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The API is proxied rather than called cross-origin, so the browser sees one
// origin in development and the app needs no knowledge of the backend's port.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  // `recharts` is imported by exactly one module — `workspace/PlanReview.tsx` — and that
  // module is `lazy()`-loaded from App.tsx. Vite's startup scan crawls from index.html
  // through main.tsx and App.tsx, so a dependency behind a dynamic import is never
  // reached: recharts was discovered mid-session, the first time someone opened the plan
  // screen and a chart actually rendered.
  //
  // Late discovery triggers an on-the-fly re-optimize, and in the window before the
  // reload lands the page can execute a half-written bundle. That surfaced as
  // `TypeError: import_decimal.default is not a constructor` out of
  // `.vite/deps/recharts.js` — recharts' own `decimal.js-light` interop — which threw
  // during render and unmounted the whole React tree, leaving a black page.
  //
  // Naming it here makes it part of the initial optimize, so it is bundled once at
  // server start and never discovered late. This affects the dev server only; the
  // production build resolves imports statically and was never subject to this.
  optimizeDeps: {
    include: ["recharts"],
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
