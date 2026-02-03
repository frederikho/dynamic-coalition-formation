import { GraphRenderer } from './graph';
import { fetchProfiles, fetchGraph } from './api';
import type { GraphData } from './types';

// UI Elements
const numPlayersSelect = document.getElementById('num-players') as HTMLSelectElement;
const profileSelect = document.getElementById('profile-select') as HTMLSelectElement;
const powerRuleSelect = document.getElementById('power-rule') as HTMLSelectElement;
const minPowerInput = document.getElementById('min-power') as HTMLInputElement;
const unanimityCheckbox = document.getElementById('unanimity') as HTMLInputElement;
const probThresholdInput = document.getElementById('prob-threshold') as HTMLInputElement;
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

    // Select default (weak_governance if available)
    const defaultProfile = data.profiles.find(p => p.name === 'weak_governance');
    if (defaultProfile) {
      profileSelect.value = defaultProfile.path;
    }

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
      numPlayers: parseInt(numPlayersSelect.value),
      profile: profileSelect.value,
      powerRule: powerRuleSelect.value,
      minPower: powerRuleSelect.value === 'power_threshold' ? parseFloat(minPowerInput.value) : null,
      unanimity: unanimityCheckbox.checked
    });

    currentGraphData = graphData;

    // Render graph
    initRenderer();
    const threshold = parseFloat(probThresholdInput.value) || 0;
    renderer!.render(graphData, threshold);

    // Update metadata display
    updateMetadata(graphData);

    // Show appropriate status message
    if (graphData.edges.length === 0) {
      showStatus('State nodes loaded. Transition probabilities not yet available for this player count.', 'info');
    } else {
      showStatus('Graph loaded successfully', 'success');
    }
  } catch (error) {
    showStatus(`Error: ${error}`, 'error');
    console.error('Error loading graph:', error);
  } finally {
    refreshBtn.disabled = false;
  }
}

// Update metadata display
function updateMetadata(data: GraphData) {
  metadataDiv.innerHTML = `
    <div><strong>Players:</strong> ${data.metadata.num_players ?? 3}</div>
    <div><strong>Profile:</strong> ${data.metadata.profile_path.split('/').pop()}</div>
    <div><strong>States:</strong> ${data.metadata.num_states}</div>
    <div><strong>Transitions:</strong> ${data.metadata.num_transitions}</div>
    <div><strong>Power Rule:</strong> ${data.metadata.config.power_rule}</div>
    <div><strong>Min Power:</strong> ${data.metadata.config.min_power ?? 'N/A'}</div>
    <div><strong>Unanimity:</strong> ${data.metadata.config.unanimity_required ? 'Yes' : 'No'}</div>
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

probThresholdInput.addEventListener('change', () => {
  if (currentGraphData && renderer) {
    const threshold = parseFloat(probThresholdInput.value) || 0;
    initRenderer();
    renderer.render(currentGraphData, threshold);
  }
});

// Power rule change - enable/disable min power input
powerRuleSelect.addEventListener('change', async () => {
  minPowerInput.disabled = powerRuleSelect.value !== 'power_threshold';
  // Auto-refresh when power rule changes
  await loadGraph();
});

// Profile change should auto-refresh
profileSelect.addEventListener('change', async () => {
  await loadGraph();
});

// Min power input should auto-refresh when changed
minPowerInput.addEventListener('change', async () => {
  await loadGraph();
});

// Unanimity toggle should auto-refresh
unanimityCheckbox.addEventListener('change', async () => {
  await loadGraph();
});

// Number of players change - reload profiles
numPlayersSelect.addEventListener('change', async () => {
  await loadProfiles();
  // loadProfiles will auto-load the first profile which calls loadGraph,
  // but ensure we refresh if a profile was already selected
  if (profileSelect.value) {
    await loadGraph();
  }
});

// Initialize
async function init() {
  // Set initial state of min power input
  minPowerInput.disabled = powerRuleSelect.value !== 'power_threshold';

  // Load profiles
  await loadProfiles();
}

init();
