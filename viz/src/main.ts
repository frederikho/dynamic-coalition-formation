import { GraphRenderer } from './graph';
import { fetchProfiles, fetchGraph } from './api';
import type { GraphData } from './types';
import { computeAbsorbingSets } from './absorbing';

// UI Elements
const profileSelect = document.getElementById('profile-select') as HTMLSelectElement;
const downloadXlsxBtn = document.getElementById('download-xlsx-btn') as HTMLButtonElement;
const probThresholdInput = document.getElementById('prob-threshold') as HTMLInputElement;
const filterModeRadios = document.querySelectorAll('input[name="filter-mode"]') as NodeListOf<HTMLInputElement>;
const thresholdTooltip = document.getElementById('threshold-tooltip') as HTMLSpanElement;
const refreshBtn = document.getElementById('refresh-btn') as HTMLButtonElement;
const resetViewBtn = document.getElementById('reset-view-btn') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const metadataDiv = document.getElementById('metadata') as HTMLDivElement;
const nodeDetailsDiv = document.getElementById('node-details') as HTMLDivElement;
const selectedStateNameSpan = document.getElementById('selected-state-name') as HTMLSpanElement;
const selectedGeoLevelSpan = document.getElementById('selected-geo-level') as HTMLSpanElement;
const outgoingTransitionsDiv = document.getElementById('outgoing-transitions') as HTMLDivElement;
const incomingTransitionsDiv = document.getElementById('incoming-transitions') as HTMLDivElement;
const graphContainer = document.getElementById('graph-container') as HTMLDivElement;
const colorByAbsorbingCheckbox = document.getElementById('color-by-absorbing') as HTMLInputElement;
const absorbingLegendDiv = document.getElementById('absorbing-legend') as HTMLDivElement;

// State
let currentGraphData: GraphData | null = null;
let renderer: GraphRenderer | null = null;

// Initialize graph renderer
function initRenderer() {
  if (renderer) {
    renderer.destroy();
  }
  renderer = new GraphRenderer(graphContainer);
  renderer.setOnNodeSelect(handleNodeSelect);
}

// Get selected filter mode
function getFilterMode(): 'absolute' | 'cumulative' {
  const selected = Array.from(filterModeRadios).find(radio => radio.checked);
  return (selected?.value as 'absolute' | 'cumulative') || 'absolute';
}

// Update tooltip based on filter mode
function updateTooltip() {
  const mode = getFilterMode();
  if (mode === 'cumulative') {
    thresholdTooltip.textContent = 'For each node, hide smallest incoming edges until their sum reaches this value';
  } else {
    thresholdTooltip.textContent = 'Hide edges with probability below this value';
  }
}

// Status display
function showStatus(message: string, type: 'info' | 'error' | 'success') {
  statusDiv.textContent = message;
  statusDiv.className = type;
  setTimeout(() => {
    statusDiv.className = '';
    statusDiv.textContent = '';
  }, 5000);
}

// Load profiles list
async function loadProfiles() {
  try {
    const data = await fetchProfiles();

    profileSelect.innerHTML = '';

    if (data.profiles.length === 0) {
      profileSelect.innerHTML = '<option value="">No profiles found</option>';
      return;
    }

    data.profiles.forEach(profile => {
      const option = document.createElement('option');
      option.value = profile.path;
      option.textContent = profile.name;
      profileSelect.appendChild(option);
    });

    // Auto-load first profile
    if (data.profiles.length > 0) {
      await loadGraph();
    }
  } catch (error) {
    showStatus(`Failed to load profiles: ${error}`, 'error');
    console.error('Error loading profiles:', error);
  }
}

// Load and render graph
async function loadGraph() {
  if (!profileSelect.value) {
    showStatus('Please select a profile', 'error');
    return;
  }

  try {
    refreshBtn.disabled = true;
    showStatus('Computing transition graph...', 'info');

    const graphData = await fetchGraph({
      profile: profileSelect.value
    });

    currentGraphData = graphData;

    // Render graph
    initRenderer();
    const threshold = parseFloat(probThresholdInput.value) || 0;
    const filterMode = getFilterMode();
    renderer!.render(graphData, threshold, { colorByAbsorbing: colorByAbsorbingCheckbox?.checked ?? false, filterMode });

    // Update metadata display
    updateMetadata(graphData);

    // Update absorbing legend
    updateAbsorbingLegend(graphData, colorByAbsorbingCheckbox?.checked ?? false);

    // Show appropriate status message
    if (graphData.edges.length === 0) {
      showStatus('State nodes loaded. Transition probabilities not yet available for this player count.', 'info');
    } else {
      showStatus('Graph loaded successfully', 'success');
    }

    // Enable download button
    downloadXlsxBtn.disabled = false;
  } catch (error) {
    showStatus(`Error: ${error}`, 'error');
    console.error('Error loading graph:', error);
    downloadXlsxBtn.disabled = true;
  } finally {
    refreshBtn.disabled = false;
  }
}

function updateAbsorbingLegend(data: GraphData, enabled: boolean) {
  if (!absorbingLegendDiv) return;
  if (!enabled) {
    absorbingLegendDiv.innerHTML = '';
    return;
  }

  // Compute absorbing sets from graph structure
  const nodeToAbsorbing = computeAbsorbingSets(data);
  const absorbingSetIds = new Set<number>();
  nodeToAbsorbing.forEach(setId => {
    if (setId !== null) absorbingSetIds.add(setId);
  });

  if (absorbingSetIds.size === 0) {
    absorbingLegendDiv.innerHTML = '<div style="color:#999; padding:6px">No absorbing sets found</div>';
    return;
  }

  // Build set membership for display
  const setMembers = new Map<number, string[]>();
  nodeToAbsorbing.forEach((setId, nodeId) => {
    if (setId !== null) {
      if (!setMembers.has(setId)) {
        setMembers.set(setId, []);
      }
      setMembers.get(setId)!.push(nodeId);
    }
  });

  const palette = ['#e11d48','#06b6d4','#84cc16','#f59e0b','#7c3aed','#10b981','#0ea5e9','#f97316','#6366f1','#db2777','#14b8a6','#f43f5e'];
  const sortedIds = Array.from(absorbingSetIds).sort((a, b) => a - b);
  const items = sortedIds.map((setId, i) => {
    const color = palette[i % palette.length];
    const members = setMembers.get(setId) || [];
    const label = members.length === 1 ? members[0] : `Set ${setId + 1} (${members.length} states)`;
    return `<div class="absorbing-legend-item" data-set-id="${setId}" style="display:flex;align-items:center;gap:8px;margin:4px 0;cursor:default"><span style="width:16px;height:12px;background:${color};display:inline-block;border-radius:2px"></span><span>${label}</span></div>`;
  });
  absorbingLegendDiv.innerHTML = `<div style="font-weight:600;margin-bottom:6px">Absorbing sets</div>` + items.join('');

  // Attach hover listeners to legend items to highlight corresponding nodes
  const legendItems = Array.from(absorbingLegendDiv.querySelectorAll('.absorbing-legend-item')) as HTMLDivElement[];
  legendItems.forEach(el => {
    el.addEventListener('mouseover', () => {
      const sid = el.getAttribute('data-set-id');
      if (!sid) return;
      const setId = parseInt(sid, 10);
      const members = setMembers.get(setId) || [];
      // Highlight nodes and their edges via renderer
      if (renderer) renderer.highlightNodes(members);
    });
    el.addEventListener('mouseout', () => {
      if (renderer) renderer.clearHighlights();
    });
  });
}

// Update metadata display
function updateMetadata(data: GraphData) {
  const fileMetadata = data.metadata.file_metadata || {};

  // Helper to parse boolean values that might be strings
  const parseBool = (val: any): boolean => {
    if (typeof val === 'boolean') return val;
    if (typeof val === 'string') return val.toLowerCase() === 'true';
    return !!val;
  };

  // Determine unanimity value - prefer config over file_metadata as it's already parsed correctly
  const unanimityValue = data.metadata.config.unanimity_required;

  metadataDiv.innerHTML = `
    <div><strong>Profile:</strong> ${data.metadata.profile_path.split('/').pop()}</div>
    <div><strong>Players:</strong> ${fileMetadata.players || data.metadata.num_players || 'N/A'}</div>
    <div><strong>States:</strong> ${data.metadata.num_states}</div>
    <div><strong>Transitions:</strong> ${data.metadata.num_transitions}</div>
    <div><strong>Power Rule:</strong> ${fileMetadata.power_rule || data.metadata.config.power_rule}</div>
    ${fileMetadata.min_power ? `<div><strong>Min Power:</strong> ${fileMetadata.min_power}</div>` : ''}
    <div><strong>Unanimity:</strong> ${unanimityValue ? 'Yes' : 'No'}</div>
    ${fileMetadata.discounting ? `<div><strong>Discounting:</strong> ${fileMetadata.discounting}</div>` : ''}
    ${fileMetadata.converged !== undefined ? `<div><strong>Converged:</strong> ${fileMetadata.converged ? 'Yes' : 'No'}</div>` : ''}
    ${fileMetadata.outer_iterations ? `<div><strong>Iterations:</strong> ${fileMetadata.outer_iterations}</div>` : ''}
    ${fileMetadata.config_hash ? `<div style="font-size: 11px; color: #888;"><strong>Hash:</strong> ${fileMetadata.config_hash}</div>` : ''}
  `;
}

// Handle node selection
function handleNodeSelect(nodeId: string | null) {
  if (!nodeId || !renderer) {
    nodeDetailsDiv.style.display = 'none';
    return;
  }

  const nodeData = renderer.getNodeData(nodeId);
  if (!nodeData) return;

  selectedStateNameSpan.textContent = nodeData.label;
  selectedGeoLevelSpan.textContent = nodeData.meta?.geo_level?.toFixed(2) ?? 'N/A';

  // Outgoing transitions
  const outgoing = renderer.getOutgoingEdges(nodeId);
  outgoingTransitionsDiv.innerHTML = outgoing.length > 0
    ? outgoing.map(edge => `
        <div class="transition-item">
          <span>${edge.target}</span>
          <span class="prob">${(edge.probability * 100).toFixed(1)}%</span>
        </div>
      `).join('')
    : '<div style="color: #999; padding: 8px;">No outgoing transitions</div>';

  // Incoming transitions
  const incoming = renderer.getIncomingEdges(nodeId);
  incomingTransitionsDiv.innerHTML = incoming.length > 0
    ? incoming.map(edge => `
        <div class="transition-item">
          <span>${edge.source}</span>
          <span class="prob">${(edge.probability * 100).toFixed(1)}%</span>
        </div>
      `).join('')
    : '<div style="color: #999; padding: 8px;">No incoming transitions</div>';

  nodeDetailsDiv.style.display = 'block';
}

// Event listeners
refreshBtn.addEventListener('click', loadGraph);

resetViewBtn.addEventListener('click', () => {
  if (renderer) {
    renderer.resetView();
  }
});

downloadXlsxBtn.addEventListener('click', () => {
  if (!profileSelect.value) {
    showStatus('No profile selected', 'error');
    return;
  }

  // Create a temporary link to download the file
  const link = document.createElement('a');
  link.href = `http://127.0.0.1:8000/download?profile=${encodeURIComponent(profileSelect.value)}`;
  link.download = profileSelect.value.split('/').pop() || 'profile.xlsx';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showStatus('Downloading XLSX file...', 'success');
});

probThresholdInput.addEventListener('change', () => {
  if (currentGraphData && renderer) {
    const threshold = parseFloat(probThresholdInput.value) || 0;
    const filterMode = getFilterMode();
    initRenderer();
    renderer.render(currentGraphData, threshold, { colorByAbsorbing: colorByAbsorbingCheckbox?.checked ?? false, filterMode });
    updateAbsorbingLegend(currentGraphData, colorByAbsorbingCheckbox?.checked ?? false);
  }
});

// Profile change should auto-refresh
profileSelect.addEventListener('change', async () => {
  await loadGraph();
});

if (colorByAbsorbingCheckbox) {
  colorByAbsorbingCheckbox.addEventListener('change', async () => {
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      const filterMode = getFilterMode();
      initRenderer();
      renderer.render(currentGraphData, threshold, { colorByAbsorbing: colorByAbsorbingCheckbox.checked, filterMode });
      updateAbsorbingLegend(currentGraphData, colorByAbsorbingCheckbox.checked);
    }
  });
}

// Filter mode change
filterModeRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    updateTooltip();
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      const filterMode = getFilterMode();
      initRenderer();
      renderer.render(currentGraphData, threshold, { colorByAbsorbing: colorByAbsorbingCheckbox?.checked ?? false, filterMode });
      updateAbsorbingLegend(currentGraphData, colorByAbsorbingCheckbox?.checked ?? false);
    }
  });
});

// Initialize
async function init() {
  // Initialize tooltip
  updateTooltip();

  // Load profiles
  await loadProfiles();
}

init();
