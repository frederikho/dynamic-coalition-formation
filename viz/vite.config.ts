import { defineConfig } from 'vite'

export default defineConfig({
  // Use a relative base so built paths work whether served from root domain
  // or GitHub Pages project subpath.
  base: process.env.VITE_BASE_PATH || './',

  server: {
    port: 3000,
    open: true,
    allowedHosts: ['.ngrok-free.app']
  },
  build: {
    outDir: 'dist',
    assetsDir: 'viz/assets',
    emptyOutDir: true,
    sourcemap: true
  }
})
