import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig(async () => ({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  // Tauri expects a fixed port and fails if the port is not available.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: "ws", host, port: 1421 }
      : undefined,
    watch: {
      // The Rust source is watched by `cargo tauri dev`; don't double-watch it.
      ignored: ["**/src-tauri/**", "**/python/**"],
    },
  },
  // Tauri targets modern webviews; smaller bundle.
  build: {
    target: process.env.TAURI_ENV_PLATFORM === "windows"
      ? "chrome105"
      : "safari15",
    minify: !process.env.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
}));
