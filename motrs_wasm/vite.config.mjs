import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    // Keep built asset URLs relative so the demo can live under a subdirectory.
    base: "./",
    server: {
        host: "127.0.0.1",
        port: 8080,
        strictPort: true,
    },
    preview: {
        host: "127.0.0.1",
        port: 8080,
        strictPort: true,
    },
});
