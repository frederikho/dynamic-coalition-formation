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
const outgoingTransitionsDiv = document.getElementById('outgoing-transitions') as HTMLDivElement;
const incomingTransitionsDiv = document.getElementById('incoming-transitions') as HTMLDivElement;
const graphContainer = document.getElementById('graph-container') as HTMLDivElement;
const nodeColoringRadios = document.querySelectorAll('input[name="node-coloring"]') as NodeListOf<HTMLInputElement>;
const layoutModeRadios = document.querySelectorAll('input[name="layout-mode"]') as NodeListOf<HTMLInputElement>;
const absorbingLegendDiv = document.getElementById('absorbing-legend') as HTMLDivElement;
const resultIndicatorDiv = document.getElementById('result-indicator') as HTMLDivElement;

// State
let currentGraphData: GraphData | null = null;
let renderer: GraphRenderer | null = null;
let previousMetadata: any = null;

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

// Get selected node coloring mode
function getNodeColoringMode(): 'none' | 'absorbing' | 'geoengineering' | 'deployer' {
  const selected = Array.from(nodeColoringRadios).find(radio => radio.checked);
  return (selected?.value as 'none' | 'absorbing' | 'geoengineering' | 'deployer') || 'none';
}

// Get selected layout mode
function getLayoutMode(): 'default' | 'connections' | 'deployer' | 'geo-level' {
  const selected = Array.from(layoutModeRadios).find(radio => radio.checked);
  return (selected?.value as 'default' | 'connections' | 'deployer' | 'geo-level') || 'default';
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
    const coloringMode = getNodeColoringMode();
    const layoutMode = getLayoutMode();
    renderer!.render(graphData, threshold, { coloringMode, filterMode, layoutMode });

    // Update metadata display
    updateMetadata(graphData);

    // Update result indicator
    updateResultIndicator(graphData);

    // Update legend
    updateLegend(graphData, coloringMode);

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

function updateLegend(data: GraphData, coloringMode: 'none' | 'absorbing' | 'geoengineering' | 'deployer') {
  if (!absorbingLegendDiv) return;

  if (coloringMode === 'none') {
    absorbingLegendDiv.innerHTML = '';
    return;
  }

  if (coloringMode === 'deployer') {
    // Get unique deploying coalitions
    const deployerSet = new Set<string>();
    data.nodes.forEach(node => {
      const deployer = node.meta?.deploying_coalition;
      if (deployer) deployerSet.add(deployer);
    });

    const deployers = Array.from(deployerSet).sort();
    const palette = ['#e11d48','#06b6d4','#84cc16','#f59e0b','#7c3aed','#10b981','#0ea5e9','#f97316','#6366f1','#db2777','#14b8a6','#f43f5e'];

    const items = deployers.map((deployer, i) => {
      const color = palette[i % palette.length];
      // Count states with this deployer
      const count = data.nodes.filter(n => n.meta?.deploying_coalition === deployer).length;
      return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:${color};display:inline-block;border-radius:2px"></span><span>${deployer} (${count} state${count !== 1 ? 's' : ''})</span></div>`;
    }).join('');

    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">Deploying Coalition</div>
      ${items}
      <div style="font-size:11px;color:#999;margin-top:8px">Border = This is the coalition which deploys within this group.</div>
    `;
  } else if (coloringMode === 'absorbing') {
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
  } else if (coloringMode === 'geoengineering') {
    // Find min/max G values for the scale
    const geoLevels = data.nodes.map(n => n.meta?.geo_level || 0);
    const minG = Math.min(...geoLevels);
    const maxG = Math.max(...geoLevels);

    // Create gradient legend
    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">Geoengineering Level (G)</div>
      <div style="margin:8px 0">
        <div style="width:100%;height:20px;background:linear-gradient(to right, #e0f2fe, #0369a1);border:1px solid #ccc;border-radius:3px"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#666">
        <span>${minG.toFixed(1)}°C</span>
        <span>${maxG.toFixed(1)}°C</span>
      </div>
      <div style="font-size:11px;color:#999;margin-top:4px">Light = Low cooling | Dark = High cooling</div>
    `;
  }
}

// State for result indicator expansion
let resultIndicatorExpanded = false;

// Update result indicator
function updateResultIndicator(data: GraphData) {
  const expectedG = data.metadata.expected_geo_level;
  const pi = data.metadata.stationary_distribution;
  const mixingTime = data.metadata.mixing_time;

  if (expectedG !== null && expectedG !== undefined) {
    renderResultIndicator(expectedG, pi, mixingTime, resultIndicatorExpanded);
    resultIndicatorDiv.style.display = 'block';
  } else {
    resultIndicatorDiv.style.display = 'none';
  }
}

function renderResultIndicator(expectedG: number, pi: any, mixingTime: number | null, expanded: boolean) {
  const expandIcon = expanded ? '▼' : '▶';

  // Format mixing time display with info icon
  let mixingTimeDisplay = '';
  if (mixingTime !== null) {
    const timeValue = mixingTime >= 0 ? `${mixingTime} steps` : `>10,000 steps`;
    const tooltipText = mixingTime >= 0
      ? `Mixing time: smallest t such that ||P^t - π|| < ε (ε=0.01). This chain converges in ${mixingTime} steps.`
      : `Mixing time: smallest t such that ||P^t - π|| < ε (ε=0.01). This chain requires >10,000 steps (slow mixing).`;

    mixingTimeDisplay = `
      <div style="font-size:12px;color:#64748b;margin-top:2px;display:flex;align-items:center;gap:4px;">
        <span>Mixing time: ${timeValue}</span>
        <span class="info-icon" style="font-size:10px;cursor:help;">
          i
          <span class="tooltip" style="width:280px;white-space:normal;text-align:left;">${tooltipText}</span>
        </span>
      </div>
    `;
  }

  let html = `
    <div class="result-summary">
      <div>
        <div><strong>E<sub>π</sub>[G]:</strong> ${expectedG.toFixed(2)}°C</div>
        ${mixingTimeDisplay}
      </div>
      <span class="expand-icon">${expandIcon}</span>
    </div>
  `;

  if (expanded && pi && currentGraphData) {
    // Collect all unique G levels from nodes
    const allGLevels = new Set<number>();
    const gToProb = new Map<number, number>();

    for (const node of currentGraphData.nodes) {
      const state = node.id;
      const gLevel = node.meta?.geo_level || 0;
      allGLevels.add(gLevel);

      const prob = (pi as Record<string, number>)[state] || 0;
      const currentProb = gToProb.get(gLevel) || 0;
      gToProb.set(gLevel, currentProb + prob);
    }

    // Sort G levels
    const sortedGLevels = Array.from(allGLevels).sort((a, b) => a - b);
    const minG = Math.min(...sortedGLevels);
    const maxG = Math.max(...sortedGLevels);
    const rangeG = maxG - minG || 1;
    const maxProb = Math.max(...Array.from(gToProb.values()));

    // Create bar chart with continuous axis
    const bars = sortedGLevels.map(g => {
      const prob = gToProb.get(g) || 0;
      const percentage = (prob * 100).toFixed(1);
      const leftPercent = ((g - minG) / rangeG * 100);
      const heightPercent = maxProb > 0 ? (prob / maxProb * 100) : 0;

      return `
        <div class="g-bar" style="left: ${leftPercent}%; height: ${heightPercent}%;">
          ${prob > 0 ? `<div class="g-bar-value">${percentage}%</div>` : ''}
        </div>
        <div class="g-bar-label" style="left: ${leftPercent}%;">${g.toFixed(1)}°C</div>
      `;
    }).join('');

    html += `
      <div class="result-details">
        <div style="font-weight:600;margin-bottom:8px">Distribution P(G)</div>
        <div class="g-chart">
          <div class="g-chart-bars">
            ${bars}
          </div>
          <div class="g-axis-label">Geoengineering Level (°C)</div>
        </div>
      </div>
    `;
  }

  resultIndicatorDiv.innerHTML = html;
  resultIndicatorDiv.className = expanded ? 'expanded' : '';
}

// Toggle result indicator expansion
resultIndicatorDiv.addEventListener('click', () => {
  resultIndicatorExpanded = !resultIndicatorExpanded;
  if (currentGraphData) {
    updateResultIndicator(currentGraphData);
  }
});

// Update metadata display
function updateMetadata(data: GraphData) {
  const fileMetadata = data.metadata.file_metadata || {};

  // Helper to parse boolean values that might be strings
  const parseBool = (val: any): boolean => {
    if (typeof val === 'boolean') return val;
    if (typeof val === 'string') return val.toLowerCase() === 'true';
    return !!val;
  };

  // Track changes from previous metadata
  const changedFields = new Set<string>();
  if (previousMetadata) {
    const prevMeta = previousMetadata.file_metadata || {};
    const prevConfig = previousMetadata.config || {};
    const currConfig = data.metadata.config;

    // Check basic fields
    if (prevMeta.power_rule !== fileMetadata.power_rule) changedFields.add('power_rule');
    if (prevMeta.min_power !== fileMetadata.min_power) changedFields.add('min_power');
    if (prevConfig.unanimity_required !== currConfig.unanimity_required) changedFields.add('unanimity');
    if (prevMeta.discounting !== fileMetadata.discounting) changedFields.add('discounting');
    if (prevMeta.players !== fileMetadata.players) changedFields.add('players');
    if (previousMetadata.scenario_name !== data.metadata.scenario_name) changedFields.add('scenario');

    // Check player parameters
    const players = typeof fileMetadata.players === 'string' ? fileMetadata.players.split(',').map(p => p.trim()) : [];
    const paramNames = ['base_temp', 'ideal_temp', 'delta_temp', 'm_damage', 'power', 'protocol'];

    for (const param of paramNames) {
      for (const player of players) {
        const key = `${param}_${player}`;
        if (prevMeta[key] !== fileMetadata[key]) {
          changedFields.add(`player_${param}_${player}`);
        }
      }
    }
  }

  // Store current metadata for next comparison
  previousMetadata = {
    file_metadata: { ...fileMetadata },
    config: { ...data.metadata.config },
    scenario_name: data.metadata.scenario_name,
    scenario_description: data.metadata.scenario_description
  };

  // Determine unanimity value - prefer config over file_metadata as it's already parsed correctly
  const unanimityValue = data.metadata.config.unanimity_required;

  // Scenario info (if available)
  const scenarioName = data.metadata.scenario_name || fileMetadata.scenario_name;
  const scenarioDescription = data.metadata.scenario_description || fileMetadata.scenario_description;

  // Helper to add highlight class if field changed
  const highlight = (fieldName: string): string => {
    return changedFields.has(fieldName) ? ' class="changed-field"' : '';
  };

  let scenarioSection = '';
  if (scenarioName) {
    scenarioSection = `
      <div style="margin-bottom: 12px; padding: 8px; background: #f8fafc; border-radius: 4px;"${highlight('scenario')}>
        <div><strong>Scenario:</strong> ${scenarioName}</div>
        ${scenarioDescription ? `<div style="font-size: 12px; color: #64748b; margin-top: 4px;">${scenarioDescription}</div>` : ''}
      </div>
    `;
  }

  // Extract player-specific parameters
  const playersStr = fileMetadata.players || data.metadata.num_players || 'N/A';
  const players = typeof playersStr === 'string' ? playersStr.split(',').map(p => p.trim()) : [];

  let playerParamsSection = '';
  if (players.length > 0) {
    const paramNames = ['base_temp', 'ideal_temp', 'delta_temp', 'm_damage', 'power', 'protocol'];
    const paramLabels: Record<string, string> = {
      'base_temp': 'Base Temp',
      'ideal_temp': 'Ideal Temp',
      'delta_temp': 'ΔTemp',
      'm_damage': 'Damage Coeff',
      'power': 'Power',
      'protocol': 'Protocol'
    };

    // Check if any player parameters exist
    const hasPlayerParams = paramNames.some(param =>
      players.some(player => fileMetadata[`${param}_${player}`] !== undefined)
    );

    if (hasPlayerParams) {
      // Create table header
      let tableHTML = `
        <div style="margin-top: 12px; padding: 8px; background: #f8fafc; border-radius: 4px;">
          <div style="font-weight: 600; margin-bottom: 8px;">PLAYER PARAMETERS</div>
          <table style="width: 100%; font-size: 12px; border-collapse: collapse;">
            <thead>
              <tr style="border-bottom: 1px solid #cbd5e1;">
                <th style="text-align: left; padding: 4px; font-weight: 600;">Param</th>
                ${players.map(p => `<th style="text-align: right; padding: 4px; font-weight: 600;">${p}</th>`).join('')}
              </tr>
            </thead>
            <tbody>
      `;

      // Helper to format numbers nicely
      const formatValue = (val: any): string => {
        if (val === undefined || isNaN(val)) return '—';
        const num = typeof val === 'number' ? val : parseFloat(val);

        // Check for common fractions
        if (Math.abs(num - 1/3) < 0.0001) return '0.333';
        if (Math.abs(num - 2/3) < 0.0001) return '0.667';
        if (Math.abs(num - 1/4) < 0.0001) return '0.25';
        if (Math.abs(num - 3/4) < 0.0001) return '0.75';
        if (Math.abs(num - 1/5) < 0.0001) return '0.2';
        if (Math.abs(num - 2/5) < 0.0001) return '0.4';
        if (Math.abs(num - 3/5) < 0.0001) return '0.6';
        if (Math.abs(num - 4/5) < 0.0001) return '0.8';

        // Round to appropriate precision
        if (Math.abs(num) >= 10) return num.toFixed(1);
        if (Math.abs(num) >= 1) return num.toFixed(2);
        return num.toFixed(3);
      };

      // Add rows for each parameter
      paramNames.forEach(param => {
        const values = players.map(player => {
          const key = `${param}_${player}`;
          return formatValue(fileMetadata[key]);
        });

        // Only show row if at least one value exists
        if (values.some(v => v !== '—')) {
          tableHTML += `
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 4px;">${paramLabels[param] || param}</td>
              ${players.map(player => {
                const val = formatValue(fileMetadata[`${param}_${player}`]);
                const changed = changedFields.has(`player_${param}_${player}`) ? ' class="changed-field"' : '';
                return `<td style="text-align: right; padding: 4px;"${changed}>${val}</td>`;
              }).join('')}
            </tr>
          `;
        }
      });

      tableHTML += `
            </tbody>
          </table>
        </div>
      `;

      playerParamsSection = tableHTML;
    }
  }

  metadataDiv.innerHTML = `
    ${scenarioSection}
    <div><strong>Profile:</strong> ${data.metadata.profile_path.split('/').pop()}</div>
    <div${highlight('players')}><strong>Players:</strong> ${playersStr}</div>
    <div><strong>States:</strong> ${data.metadata.num_states}</div>
    <div><strong>Transitions:</strong> ${data.metadata.num_transitions}</div>
    <div${highlight('power_rule')}><strong>Power Rule:</strong> ${fileMetadata.power_rule || data.metadata.config.power_rule}</div>
    ${fileMetadata.min_power ? `<div${highlight('min_power')}><strong>Min Power:</strong> ${fileMetadata.min_power}</div>` : ''}
    <div${highlight('unanimity')}><strong>Unanimity:</strong> ${unanimityValue ? 'Yes' : 'No'}</div>
    ${fileMetadata.discounting ? `<div${highlight('discounting')}><strong>Discounting:</strong> ${fileMetadata.discounting}</div>` : ''}
    ${fileMetadata.converged !== undefined ? `<div><strong>Converged:</strong> ${fileMetadata.converged ? 'Yes' : 'No'}</div>` : ''}
    ${fileMetadata.outer_iterations ? `<div><strong>Iterations:</strong> ${fileMetadata.outer_iterations}</div>` : ''}
    ${fileMetadata.config_hash ? `<div style="font-size: 11px; color: #888;"><strong>Hash:</strong> ${fileMetadata.config_hash}</div>` : ''}
    ${playerParamsSection}
  `;

  // Trigger animation by adding and removing class after a delay
  setTimeout(() => {
    const changedElements = metadataDiv.querySelectorAll('.changed-field');
    changedElements.forEach(el => {
      el.classList.remove('changed-field');
    });
  }, 5000);
}

// Handle node selection
function handleNodeSelect(nodeId: string | null) {
  if (!nodeId || !renderer) {
    nodeDetailsDiv.style.display = 'none';
    return;
  }

  const nodeData = renderer.getNodeData(nodeId);
  if (!nodeData) return;

  // Normalize state name for display
  const normalizeStateName = (window as any).normalizeStateName;
  const normalizedStateName = normalizeStateName ? normalizeStateName(nodeId) : nodeId;

  // Display state name with deploying coalition info if available
  const deployingCoalition = nodeData.data?.deploying_coalition;
  const geoLevel = nodeData.data?.geo_level;
  if (deployingCoalition !== undefined && geoLevel !== undefined) {
    selectedStateNameSpan.innerHTML = `
      ${normalizedStateName}
      <div style="font-size:11px;color:#64748b;margin-top:4px;">
        Deployed by: <strong>${deployingCoalition}</strong> (G = ${geoLevel.toFixed(3)}°C)
      </div>
    `;
  } else {
    selectedStateNameSpan.textContent = normalizedStateName;
  }

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

  // Extract filename from path
  const pathParts = profileSelect.value.split('/');
  const filename = pathParts[pathParts.length - 1];

  // Create a temporary link to download the file from static data
  const link = document.createElement('a');
  link.href = `/data/xlsx/${filename}`;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showStatus('Downloading XLSX file...', 'success');
});

probThresholdInput.addEventListener('change', () => {
  if (currentGraphData && renderer) {
    const threshold = parseFloat(probThresholdInput.value) || 0;
    const filterMode = getFilterMode();
    const coloringMode = getNodeColoringMode();
    const layoutMode = getLayoutMode();
    // Don't call initRenderer() - positions will be preserved
    renderer.render(currentGraphData, threshold, { coloringMode, filterMode, layoutMode });
    updateLegend(currentGraphData, coloringMode);
  }
});

// Profile change should auto-refresh
profileSelect.addEventListener('change', async () => {
  await loadGraph();
});

// Node coloring mode change
nodeColoringRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      const filterMode = getFilterMode();
      const coloringMode = getNodeColoringMode();
      const layoutMode = getLayoutMode();
      // Don't call initRenderer() - positions will be preserved
      renderer.render(currentGraphData, threshold, { coloringMode, filterMode, layoutMode });
      updateLegend(currentGraphData, coloringMode);
    }
  });
});

// Layout mode change
layoutModeRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      const filterMode = getFilterMode();
      const coloringMode = getNodeColoringMode();
      const layoutMode = getLayoutMode();
      // Don't preserve positions when changing layout mode - need fresh layout
      initRenderer();
      renderer.render(currentGraphData, threshold, { coloringMode, filterMode, layoutMode });
      updateLegend(currentGraphData, coloringMode);
    }
  });
});

// Filter mode change
filterModeRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    updateTooltip();
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      const filterMode = getFilterMode();
      const coloringMode = getNodeColoringMode();
      const layoutMode = getLayoutMode();
      // Don't call initRenderer() - positions will be preserved
      renderer.render(currentGraphData, threshold, { coloringMode, filterMode, layoutMode });
      updateLegend(currentGraphData, coloringMode);
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
