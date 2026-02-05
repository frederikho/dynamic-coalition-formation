import { defineConfig } from 'vite'

export default defineConfig({
  // For GitHub Pages: set base to '/repo-name/' where repo-name is your repository name
  // For custom domain or root deployment, use '/'
  base: process.env.VITE_BASE_PATH || '/',

  server: {
    port: 3000,
    open: true,
    allowedHosts: ['.ngrok-free.app']
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
