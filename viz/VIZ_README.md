# Transition Graph Visualizer - Quick Start Guide

This repository includes an interactive visualization tool for exploring coalition formation transition graphs.

## What Does It Do?

The visualizer lets you:
- See coalition states as nodes in an interactive graph
- Explore transition probabilities between states as directed edges
- Edit strategy profile XLSX files and see results immediately
- Compare different governance scenarios (weak governance vs. power threshold)
- Analyze equilibrium strategies visually

## Quick Start

### Step 1: Install Dependencies

```bash
# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Install Node.js dependencies for the visualizer
cd viz
npm install
cd ..
```

### Step 2: Start the Backend Service

From the repository root:

```bash
python -m service_viz
```

You should see:
```
Starting Coalition Formation Visualizer API
  Host: 127.0.0.1
  Port: 8000
  Profiles dir: strategy_tables
  API docs: http://127.0.0.1:8000/docs
```

Keep this terminal running.

### Step 3: Start the Frontend

In a **new terminal**, from the repository root:

```bash
cd viz
npm run dev
```

Your browser should automatically open to `http://localhost:3000` showing the visualizer.

## Using the Visualizer

1. **Select a strategy profile** from the dropdown (e.g., `weak_governance`)
2. **Click "Refresh"** to compute and display the transition graph
3. **Interact with the graph**:
   - Zoom with mouse wheel
   - Pan by dragging
   - Hover over nodes to highlight connections
   - Click a node to see detailed transition probabilities in the sidebar
4. **Adjust parameters** and click Refresh to see how they affect transitions:
   - Power Rule: Switch between weak governance and power threshold
   - Probability Threshold: Filter out low-probability transitions

## Example Workflow: Editing Strategy Profiles

1. Open `strategy_tables/weak_governance.xlsx` in Excel/LibreOffice
2. Modify some proposition or acceptance probabilities
3. Save the file
4. In the visualizer, click **"Refresh"**
5. The graph updates immediately to reflect your changes

**No need to restart anything** - the system recomputes on every refresh.

## Understanding the Visualization

### Nodes (Coalition States)
- **( )**: All countries acting independently
- **(TC)**: Countries T and C in a coalition
- **(WC)**: Countries W and C in a coalition
- **(WT)**: Countries W and T in a coalition
- **(WTC)**: Grand coalition (all countries cooperating)

### Edges (Transitions)
- **Arrow direction**: Shows which state can transition to which
- **Edge color**:
  - Green: High probability transition (≥ 50%)
  - Yellow: Medium probability (10-50%)
  - Gray: Low probability (< 10%)
- **Edge label**: Exact transition probability
- **Self-loops** (curved edges back to same node): Probability of staying in current state

### Absorbing States
States with only self-loops (probability 1.0 of staying) are **absorbing states** - once reached, the system stays there in equilibrium.

## Files and Directories

```
.
├── service_viz.py          # Python backend API (new)
├── viz/                    # Frontend visualizer (new)
│   ├── src/               # TypeScript source
│   ├── index.html         # UI
│   ├── package.json       # Node dependencies
│   └── README.md          # Detailed frontend docs
├── strategy_tables/        # XLSX strategy profiles
│   ├── weak_governance.xlsx
│   ├── power_threshold.xlsx
│   └── ...
├── lib/                    # Research code (unchanged)
├── main.py                 # Original simulation (unchanged)
└── requirements.txt        # Python dependencies (updated)
```

## Integration with Research Code

The visualizer is designed as a **thin integration layer**:
- **No changes to `main.py`** or core research logic
- `service_viz.py` imports and calls existing computation functions from `lib/`
- Frontend is entirely separate in `viz/` directory
- You can still run `python main.py` as before to generate all results

## Troubleshooting

**"Failed to fetch profiles"**
- Make sure `python -m service_viz` is running
- Check that port 8000 is not blocked

**"Profile not found"**
- Verify XLSX files exist in `strategy_tables/`
- Close any lock files (`.~lock.*.xlsx`) that Excel may have created

**Graph doesn't update after editing XLSX**
- Make sure you saved the file
- Click "Refresh" in the visualizer (it doesn't auto-reload)

**Port already in use**
- Backend: `python -m service_viz --port 8001`
- Frontend: Edit `viz/vite.config.ts` to change port

## Advanced Usage

See `viz/README.md` for:
- API endpoint documentation
- Custom graph layouts
- Adding metadata fields
- Performance tuning
- Production deployment

## Questions?

For visualization-specific questions, see `viz/README.md`.

For research code questions, see the main `README.md` and `CLAUDE.md`.
