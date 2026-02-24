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
    allowedHosts: ['.ngrok-free.app']
  },
  build: {
    // Output directly to repo root so built files are ready to commit for GitHub Pages.
    // index.html → repo root, JS/CSS → viz/assets/
    outDir: '../',
    assetsDir: 'viz/assets',
    emptyOutDir: false,
    sourcemap: true
  }
})
