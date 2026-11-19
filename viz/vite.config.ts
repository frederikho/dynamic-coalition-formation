import { defineConfig } from 'vite'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  // Use a relative base so built paths work whether served from root domain
  // or GitHub Pages project subpath.
  base: process.env.VITE_BASE_PATH || './',

  plugins: [
    {
      name: 'serve-viz-data',
      configureServer(server) {
        // Serve viz/data/ at /viz/data/ in dev so it matches the production URL
        server.middlewares.use('/viz/data', (req, res, next) => {
          const dataDir = path.resolve(__dirname, 'data')
          const filePath = path.join(dataDir, decodeURIComponent(req.url ?? ''))
          if (!filePath.startsWith(dataDir)) return next()
          if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
            const ext = path.extname(filePath)
            const contentType =
              ext === '.json' ? 'application/json' :
              ext === '.xlsx' ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' :
              'application/octet-stream'
            res.setHeader('Content-Type', contentType)
            res.end(fs.readFileSync(filePath))
          } else {
            next()
          }
        })
      }
    }
  ],

  server: {
    port: 3000,
    open: true,
    allowedHosts: ['.ngrok-free.app'],
    // Native filesystem change events don't reliably reach chokidar for files
    // on a Windows-mounted drive under WSL (/mnt/d/... via drvfs/9p), so HMR
    // silently stops picking up edits. Polling works around that.
    watch: {
      usePolling: true,
      interval: 300
    }
  },
  build: {
    // VITE_OUT_DIR (relative to viz/, e.g. 'dist') is what CI and `npm run build` use:
    // index.html + viz/assets/ in a self-contained folder that gets deployed to GitHub Pages.
    // Without it, output goes to the repo root (legacy behaviour).
    outDir: process.env.VITE_OUT_DIR || '../',
    assetsDir: 'viz/assets',
    emptyOutDir: !!process.env.VITE_OUT_DIR,
    sourcemap: true
  }
})
