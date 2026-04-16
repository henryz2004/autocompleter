import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://autocompleter.dev",
  server: {
    port: 4321,
    host: true,
  },
  vite: {
    server: {
      fs: {
        strict: false,
      },
    },
  },
});
