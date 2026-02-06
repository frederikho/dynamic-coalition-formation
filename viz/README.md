# Coalition Transition Graph Visualizer

Interactive visualization tool for exploring Markov chain transition graphs derived from coalition formation strategy profiles.

## Overview

This visualization tool provides an interactive way to:
- Visualize state transition graphs as directed networks
- Explore transition probabilities between coalition structures
- Compare different governance scenarios in real-time
- Analyze equilibrium strategy profiles

## Architecture

The system consists of two components:

1. **Python Viz Service** (`../viz_service.py`): FastAPI server that computes transition graphs from XLSX strategy profiles
2. **Frontend** (this directory): Vite + TypeScript + Sigma.js web application for rendering and interaction

## Prerequisites

- Python 3.8+ with dependencies from `../requirements.txt`
- Node.js 18+ and npm

## Quick Start

### 1. Install Python Dependencies

From the repository root:

```bash
pip install fastapi uvicorn openpyxl
```

(Or add these to `requirements.txt` if not already present)

### 2. Start the Python Viz Service

From the repository root:

```bash
python -m viz_service
```

This starts the API server at `http://127.0.0.1:8000`

Options:
- `--host 127.0.0.1` - Host to bind to
- `--port 8000` - Port to bind to
- `--profiles-dir strategy_tables` - Directory containing XLSX profiles

API documentation available at: `http://127.0.0.1:8000/docs`

### 3. Install Frontend Dependencies

From the `viz/` directory:

```bash
npm install
```

### 4. Start the Frontend Dev Server

```bash
npm run dev
```

This starts the Vite dev server at `http://localhost:3000` and opens it in your browser.

## Usage

### Basic Workflow

1. **Select a profile**: Choose from available XLSX strategy profiles (e.g., `weak_governance.xlsx`)
2. **Configure parameters**:
   - Power Rule: `weak_governance` or `power_threshold`
   - Min Power: Threshold for power_threshold rule (0.0 - 1.0)
   - Require Unanimity: Toggle approval committee voting rule
   - Probability Threshold: Filter edges below this probability
3. **Click "Refresh"**: Recomputes the transition graph from the XLSX file
4. **Interact with the graph**:
   - Zoom/pan to navigate
   - Hover over nodes to highlight neighborhood
   - Click nodes to see detailed transition information
   - Click "Reset View" to center the graph

### Node and Edge Visualization

- **Nodes**: Represent coalition states (e.g., `( )`, `(TC)`, `(WTC)`)
- **Edges**: Directed arrows showing transition probabilities
  - Edge color indicates probability: green (high), yellow (medium), gray (low)
  - Edge labels show exact probability values
  - Self-loops indicate staying in the same state
- **Selected node**: Highlighted in red with detailed transition list in sidebar

### Metadata Display

The sidebar shows:
- Number of states and transitions
- Current governance configuration
- Selected state details:
  - Geoengineering deployment level
  - Outgoing transitions (sorted by probability)
  - Incoming transitions (sorted by probability)

## Adding New Strategy Profiles

1. Create a new XLSX file in the `strategy_tables/` directory
2. Follow the existing format (multi-level headers for proposers/states, proposition and acceptance rows)
3. Refresh the profiles list by reloading the page or restarting the viz service
4. Select your new profile from the dropdown

## Technical Details

### Graph Layout

- Initial positions: Circular layout
- Refinement: Force Atlas 2 algorithm (500 iterations)
- Manual override: Can provide `x`, `y` coordinates in the API response

### Self-Loop Rendering

Self-loops (transitions that stay in the same state) are rendered as curved edges. This is a common approach in Sigma.js for visualizing self-referential transitions.

### Performance

- The Python service recomputes the full transition graph on every request (no caching)
- Frontend applies probability threshold filtering client-side
- Suitable for graphs with up to ~100 states and ~500 edges

### API Endpoints

- `GET /`: API root information
- `GET /graph?profile=...&power_rule=...&min_power=...&unanimity=...`: Compute transition graph
- `GET /profiles`: List available XLSX strategy profiles

## Development

### Building for Production

```bash
npm run build
```

Output is in `dist/` directory.

To preview the production build:

```bash
npm run preview
```

### Project Structure

```
viz/
├── src/
│   ├── main.ts        # Application entry point
│   ├── types.ts       # TypeScript type definitions
│   ├── api.ts         # API client for viz service
│   └── graph.ts       # Sigma.js graph renderer
├── index.html         # HTML template
├── package.json       # Node dependencies
├── tsconfig.json      # TypeScript configuration
├── vite.config.ts     # Vite bundler configuration
└── README.md          # This file
```

## Troubleshooting

### Port Conflicts

If port 8000 or 3000 is already in use:
- Python service: Use `--port <PORT>` flag
- Frontend: Edit `vite.config.ts` to change the dev server port

### CORS Errors

The viz service enables CORS for `*` origins in development. For production, restrict this in `viz_service.py`.

### Profile Not Found

- Ensure the XLSX file exists in `strategy_tables/`
- Check the path in the dropdown matches the file location
- Lock files (`.~lock.*`) are automatically filtered out

### Graph Doesn't Render

- Check browser console for JavaScript errors
- Verify the Python service is running (`http://127.0.0.1:8000`)
- Try clicking "Reset View" or refreshing the page

## Extending the Visualizer

### Adding More Graph Metadata

Edit `viz_service.py` `compute_transition_graph()` to include additional node/edge metadata:

```python
nodes.append({
    "id": state_name,
    "label": state_name,
    "meta": {
        "geo_level": geo_levels[state_name],
        "your_custom_field": ...  # Add here
    }
})
```

### Custom Layouts

Edit `src/graph.ts` `applyLayout()` method to use different graph layout algorithms from `graphology-layout`.

### Styling

Modify colors, sizes, and styles in:
- `index.html` (CSS styles)
- `src/graph.ts` (node/edge rendering parameters)

## License

Same as the parent repository.
