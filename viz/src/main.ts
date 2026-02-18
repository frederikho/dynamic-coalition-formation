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
const graphContainer = document.getElementById('graph-container') as HTMLDivElement;
const nodeColoringRadios = document.querySelectorAll('input[name="node-coloring"]') as NodeListOf<HTMLInputElement>;
const layoutModeRadios = document.querySelectorAll('input[name="layout-mode"]') as NodeListOf<HTMLInputElement>;
const absorbingLegendDiv = document.getElementById('absorbing-legend') as HTMLDivElement;
const resultIndicatorDiv = document.getElementById('result-indicator') as HTMLDivElement;
const menuToggleBtn = document.getElementById('menu-toggle') as HTMLButtonElement;
const sidebarBackdrop = document.getElementById('sidebar-backdrop') as HTMLDivElement;
const sidebar = document.getElementById('sidebar') as HTMLDivElement;
const sidebarCloseBtn = document.getElementById('sidebar-close') as HTMLButtonElement;

// State
let currentGraphData: GraphData | null = null;
let renderer: GraphRenderer | null = null;
let previousMetadata: any = null;
let highlightTimeoutId: number | null = null;

// Temporary display toggles (can toggle with keyboard shortcuts)
let showSelfLoops = true;
let showEdgeLabels = true;
let showNodeLabels = true;

function getRenderOptions() {
  const filterMode = getFilterMode();
  const coloringMode = getNodeColoringMode();
  const layoutMode = getLayoutMode();
  return { coloringMode, filterMode, layoutMode, showSelfLoops, showEdgeLabels, showNodeLabels };
}

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

    const sortedProfiles = [...data.profiles].sort((a, b) => (a.created_at ?? 0) - (b.created_at ?? 0));

    sortedProfiles.forEach(profile => {
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
    renderer!.render(graphData, threshold, getRenderOptions());

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
  const absorptionTime = (data.metadata as any).absorption_time;

  if (expectedG !== null && expectedG !== undefined) {
    renderResultIndicator(expectedG, pi, mixingTime, absorptionTime, resultIndicatorExpanded);
    resultIndicatorDiv.style.display = 'block';
  } else {
    resultIndicatorDiv.style.display = 'none';
  }
}

function renderResultIndicator(expectedG: number, pi: any, mixingTime: number | null, absorptionTime: any | null, expanded: boolean) {
  const expandIcon = expanded ? '▼' : '▶';

  // Format convergence time display
  let convergenceDisplay = '';
  if (absorptionTime && absorptionTime.max != null) {
    // Non-ergodic chain with absorbing sets: show absorption time
    const maxSteps = Math.round(absorptionTime.max);
    const meanSteps = Math.round(absorptionTime.mean);
    const tooltipText = `Expected steps to reach an absorbing state from a transient state. Mean: ${meanSteps}, worst-case: ${maxSteps}. Mixing time is undefined for chains with multiple absorbing sets.`;

    convergenceDisplay = `
      <div style="font-size:12px;color:#64748b;margin-top:2px;display:flex;align-items:center;gap:4px;">
        <span>Absorption time: ${maxSteps} steps (worst-case)</span>
        <span class="info-icon" style="font-size:10px;cursor:help;">
          i
          <span class="tooltip tooltip-mixing">${tooltipText}</span>
        </span>
      </div>
    `;
  } else if (mixingTime != null && mixingTime >= 0) {
    // Ergodic chain with finite mixing time
    const tooltipText = `Mixing time: smallest t such that ||P^t - \u03C0|| < \u03B5 (\u03B5=0.01). This chain converges in ${mixingTime} steps.`;

    convergenceDisplay = `
      <div style="font-size:12px;color:#64748b;margin-top:2px;display:flex;align-items:center;gap:4px;">
        <span>Mixing time: ${mixingTime} steps</span>
        <span class="info-icon" style="font-size:10px;cursor:help;">
          i
          <span class="tooltip tooltip-mixing">${tooltipText}</span>
        </span>
      </div>
    `;
  }

  let html = `
    <div class="result-summary">
      <div>
        <div><strong>E<sub>π</sub>[G]:</strong> ${expectedG.toFixed(2)}°C</div>
        ${convergenceDisplay}
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

  // Helper to compute derived player parameter
  const computeDerivedParam = (param: string, player: string, metadata: any): number | undefined => {
    if (param === 'temp_before_sg') {
      const baseTemp = metadata[`base_temp_${player}`];
      const deltaTemp = metadata[`delta_temp_${player}`];
      if (baseTemp !== undefined && deltaTemp !== undefined) {
        return parseFloat(baseTemp) + parseFloat(deltaTemp);
      }
    } else if (param === 'ideal_g') {
      const baseTemp = metadata[`base_temp_${player}`];
      const deltaTemp = metadata[`delta_temp_${player}`];
      const idealTemp = metadata[`ideal_temp_${player}`];
      if (baseTemp !== undefined && deltaTemp !== undefined && idealTemp !== undefined) {
        const tempBeforeSG = parseFloat(baseTemp) + parseFloat(deltaTemp);
        return tempBeforeSG - parseFloat(idealTemp);
      }
    }
    return undefined;
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

    // Check player parameters (including derived ones)
    const players = typeof fileMetadata.players === 'string' ? fileMetadata.players.split(',').map(p => p.trim()) : [];
    const allParamNames = ['base_temp', 'delta_temp', 'temp_before_sg', 'ideal_temp', 'ideal_g', 'm_damage', 'power', 'protocol'];

    for (const param of allParamNames) {
      for (const player of players) {
        let currentValue: any;
        let previousValue: any;

        // Check if it's a derived parameter
        if (param === 'temp_before_sg' || param === 'ideal_g') {
          currentValue = computeDerivedParam(param, player, fileMetadata);
          previousValue = computeDerivedParam(param, player, prevMeta);
        } else {
          const key = `${param}_${player}`;
          currentValue = fileMetadata[key];
          previousValue = prevMeta[key];
        }

        // Mark as changed if values differ
        if (currentValue !== previousValue) {
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
    const paramNames = ['base_temp', 'delta_temp', 'temp_before_sg', 'ideal_temp', 'ideal_g', 'm_damage', 'power', 'protocol'];
    const paramLabels: Record<string, string> = {
      'base_temp': 'Base Temp',
      'delta_temp': 'ΔTemp',
      'temp_before_sg': 'Temp before SG',
      'ideal_temp': 'Ideal Temp',
      'ideal_g': 'Ideal G',
      'm_damage': 'Damage Coeff',
      'power': 'Power',
      'protocol': 'Protocol'
    };

    const paramTooltips: Record<string, string> = {
      'power': 'Share of total world power. Determines influence within coalitions and whether a coalition meets the minimum power threshold to deploy geoengineering.',
      'protocol': 'Probability of being selected as the proposer for state transitions. Uniform protocol is the default and means equal probability for all countries.'
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
        let values: string[];

        // Get values (derived or direct from metadata)
        if (param === 'temp_before_sg' || param === 'ideal_g') {
          values = players.map(player => {
            const val = computeDerivedParam(param, player, fileMetadata);
            return formatValue(val);
          });
        } else {
          // Original parameters from file metadata
          values = players.map(player => {
            const key = `${param}_${player}`;
            return formatValue(fileMetadata[key]);
          });
        }

        // Only show row if at least one value exists
        if (values.some(v => v !== '—')) {
          const label = paramLabels[param] || param;
          const tooltip = paramTooltips[param];
          const tooltipClass = tooltip ? `tooltip tooltip-${param}` : '';
          const labelWithTooltip = tooltip
            ? `${label} <span class="info-icon" style="margin-left:4px;">i<span class="${tooltipClass}">${tooltip}</span></span>`
            : label;

          tableHTML += `
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 4px;">${labelWithTooltip}</td>
              ${players.map((player, i) => {
                const val = values[i];
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
    <div${highlight('unanimity')}>
      <strong>Unanimity:</strong> ${unanimityValue ? 'Yes' : 'No'}
      <span class="info-icon" style="margin-left:4px;">
        i
        <span class="tooltip tooltip-unanimity">If Yes, ALL members of the approval committee must approve a proposed transition. If No, only a simple majority is needed (all new members + majority of existing members).</span>
      </span>
    </div>
    ${fileMetadata.discounting ? `<div${highlight('discounting')}><strong>Discounting:</strong> ${fileMetadata.discounting}</div>` : ''}
    ${fileMetadata.converged !== undefined ? `<div><strong>Converged:</strong> ${fileMetadata.converged ? 'Yes' : 'No'}</div>` : ''}
    ${fileMetadata.outer_iterations ? `<div><strong>Iterations:</strong> ${fileMetadata.outer_iterations}</div>` : ''}
    ${fileMetadata.config_hash ? `<div style="font-size: 11px; color: #888;"><strong>Hash:</strong> ${fileMetadata.config_hash}</div>` : ''}
    ${fileMetadata.end_time ? `<div style="font-size: 11px; color: #888;"><strong>Computed:</strong> ${fileMetadata.end_time}</div>` : ''}
    ${playerParamsSection}
  `;

  // Clear any pending highlight removal from previous profile switch
  if (highlightTimeoutId !== null) {
    clearTimeout(highlightTimeoutId);
  }

  // Trigger animation by adding and removing class after a delay
  highlightTimeoutId = window.setTimeout(() => {
    const changedElements = metadataDiv.querySelectorAll('.changed-field');
    changedElements.forEach(el => {
      el.classList.remove('changed-field');
    });
    highlightTimeoutId = null;
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

  // Helper to render transition breakdown grouped by proposer
  const renderBreakdown = (breakdown: any[]) => {
    if (!breakdown || breakdown.length === 0) return '';

    // Group by proposer
    const byProposer: Record<string, { total: number; paths: any[] }> = {};
    for (const path of breakdown) {
      const p = path.proposer;
      if (!byProposer[p]) byProposer[p] = { total: 0, paths: [] };
      byProposer[p].total += path.path_prob;
      byProposer[p].paths.push(path);
    }

    // Sort proposers by total contribution descending
    const sorted = Object.entries(byProposer).sort((a, b) => b[1].total - a[1].total);

    return sorted.map(([proposer, data]) => {
      // Build individual path entries showing: target (prop_prob%) -> approval details
      const pathEntries = data.paths.map(path => {
        const approvedBy = path.committee.filter((p: string) => path.approvals[p] === 1);
        const rejectedBy = path.committee.filter((p: string) => path.approvals[p] === 0);
        const mixedBy = path.committee.filter((p: string) => {
          const prob = path.approvals[p];
          return prob !== undefined && prob > 0 && prob < 1;
        });
        const proposedTarget = path.proposed_target
          ? (normalizeStateName ? normalizeStateName(path.proposed_target) : path.proposed_target)
          : '';

        // Format approval/rejection details
        let approvalDetail = '';
        if (path.type === 'rejection') {
          // Show rejection: only show percentages for mixed strategies
          const details: string[] = [];
          if (mixedBy.length > 0) {
            // Mixed strategy: show all players with their approval probabilities
            rejectedBy.forEach(p => details.push(`${p} (0%)`));
            mixedBy.forEach(p => details.push(`${p} (${(path.approvals[p] * 100).toFixed(0)}%)`));
            approvalDetail = `rejected by ${details.join(', ')}`;
          } else {
            // Pure rejection: just list who rejected
            approvalDetail = rejectedBy.length > 0 ? `rejected by ${rejectedBy.join(', ')}` : 'rejected';
          }
        } else {
          // Show acceptance: only show percentages for mixed strategies
          const details: string[] = [];
          if (mixedBy.length > 0) {
            // Mixed strategy: show all players with their approval probabilities
            approvedBy.forEach(p => details.push(`${p} (100%)`));
            mixedBy.forEach(p => details.push(`${p} (${(path.approvals[p] * 100).toFixed(0)}%)`));
            approvalDetail = `approved by ${details.join(', ')}`;
          } else {
            // Pure approval: just list who approved
            approvalDetail = approvedBy.length > 0 ? `approved by ${approvedBy.join(', ')}` : 'approved';
          }
        }

        return {
          target: proposedTarget,
          propProb: path.prop_prob,
          approvalDetail: approvalDetail,
          pathProb: path.path_prob
        };
      });

      // Build the detail lines: target (prop_prob%) -> approval details
      const lines = pathEntries.map(entry => {
        return `<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>${entry.target} (${(entry.propProb * 100).toFixed(1)}%) \u2192 ${entry.approvalDetail}</span><span style="white-space:nowrap;margin-left:8px;color:#94a3b8;">${(entry.pathProb * 100).toFixed(1)}%</span></div>`;
      }).join('');

      // Single line: show inline
      if (pathEntries.length === 1) {
        return `
          <div style="font-size:10px;color:#64748b;margin-left:16px;padding:4px 0;border-top:1px solid #e5e7eb;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
              <strong>${proposer} proposes:</strong>
              <span>${(data.total * 100).toFixed(1)}%</span>
            </div>
            <div style="margin-left:8px;">${lines}</div>
          </div>
        `;
      }

      // Multiple lines: expandable
      const detailId = `proposer-detail-${proposer}-${Math.random().toString(36).slice(2, 8)}`;
      return `
        <div style="font-size:10px;color:#64748b;margin-left:16px;padding:4px 0;border-top:1px solid #e5e7eb;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>${proposer} proposes:</strong>
            <div style="display:flex;align-items:center;gap:6px;">
              <span>${(data.total * 100).toFixed(1)}%</span>
              <button class="breakdown-toggle" data-target="${detailId}" style="background:none;border:none;cursor:pointer;padding:2px;color:#64748b;font-size:10px;">\u25BC</button>
            </div>
          </div>
          <div id="${detailId}" class="breakdown-content" style="display:none;margin-left:8px;">
            ${lines}
          </div>
        </div>
      `;
    }).join('');
  };

  // Outgoing transitions
  const outgoing = renderer.getOutgoingEdges(nodeId);
  outgoingTransitionsDiv.innerHTML = outgoing.length > 0
    ? outgoing.map((edge, idx) => {
        const breakdownId = `out-breakdown-${idx}`;
        const hasBreakdown = edge.breakdown && edge.breakdown.length > 0;
        const targetName = normalizeStateName ? normalizeStateName(edge.target) : edge.target;
        const isSelfLoop = edge.target === nodeId;
        const label = isSelfLoop
          ? `${targetName} <span style="color:#b0b8c4;font-style:italic;font-weight:normal;">\u2014 status quo</span>`
          : targetName;
        return `
          <div class="transition-item" style="display:block;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span>${label}</span>
              <div style="display:flex;align-items:center;gap:8px;">
                <span class="prob">${(edge.probability * 100).toFixed(1)}%</span>
                ${hasBreakdown ? `<button class="breakdown-toggle" data-target="${breakdownId}" style="background:none;border:none;cursor:pointer;padding:4px;color:#64748b;font-size:12px;">▼</button>` : ''}
              </div>
            </div>
            ${hasBreakdown ? `<div id="${breakdownId}" class="breakdown-content" style="display:none;">${renderBreakdown(edge.breakdown)}</div>` : ''}
          </div>
        `;
      }).join('')
    : '<div style="color: #999; padding: 8px;">No outgoing transitions</div>';
  // Add event listeners for breakdown toggles
  document.querySelectorAll('.breakdown-toggle').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = (e.target as HTMLElement).closest('.breakdown-toggle');
      if (!target) return;
      const targetId = target.getAttribute('data-target');
      if (!targetId) return;
      const content = document.getElementById(targetId);
      if (!content) return;

      if (content.style.display === 'none') {
        content.style.display = 'block';
        target.textContent = '▲';
      } else {
        content.style.display = 'none';
        target.textContent = '▼';
      }
    });
  });

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
  // Use BASE_URL to handle GitHub Pages subdirectory deployment
  const baseUrl = import.meta.env.BASE_URL || '/';
  const xlsxBase = import.meta.env.DEV ? `${baseUrl}data/xlsx` : `${baseUrl}viz/data/xlsx`;
  const link = document.createElement('a');
  link.href = `${xlsxBase}/${filename}`;
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
    renderer.render(currentGraphData, threshold, getRenderOptions());
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
      renderer.render(currentGraphData, threshold, getRenderOptions());
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
      renderer.render(currentGraphData, threshold, getRenderOptions());
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
      renderer.render(currentGraphData, threshold, getRenderOptions());
      updateLegend(currentGraphData, coloringMode);
    }
  });
});

// Mobile menu toggle functionality
function openMobileMenu() {
  sidebar.classList.add('active');
  sidebarBackdrop.classList.add('active');
  document.body.style.overflow = 'hidden'; // Prevent scrolling when menu is open
}

function closeMobileMenu() {
  sidebar.classList.remove('active');
  sidebarBackdrop.classList.remove('active');
  document.body.style.overflow = ''; // Restore scrolling
}

// Toggle mobile menu
menuToggleBtn.addEventListener('click', () => {
  if (sidebar.classList.contains('active')) {
    closeMobileMenu();
  } else {
    openMobileMenu();
  }
});

// Close menu when clicking backdrop
sidebarBackdrop.addEventListener('click', closeMobileMenu);

// Close menu when clicking close button
sidebarCloseBtn.addEventListener('click', closeMobileMenu);

// Close menu when user selects an option (for better UX)
// This makes it easier to see the changes on the graph
profileSelect.addEventListener('change', () => {
  if (window.innerWidth <= 768) {
    closeMobileMenu();
  }
});

// Also close when user applies visualization changes on mobile
nodeColoringRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (window.innerWidth <= 768) {
      // Small delay so user can see what they selected
      setTimeout(closeMobileMenu, 300);
    }
  });
});

layoutModeRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (window.innerWidth <= 768) {
      setTimeout(closeMobileMenu, 300);
    }
  });
});

// Keyboard shortcuts for quick toggles (temporary display changes)
// s = toggle self-loops, e = toggle edge labels, l = toggle node labels
document.addEventListener('keydown', (ev) => {
  const key = ev.key.toLowerCase();
  let changed = false;
  if (key === 's') {
    showSelfLoops = !showSelfLoops;
    showStatus(`Self-loops ${showSelfLoops ? 'shown' : 'hidden'}`, 'info');
    changed = true;
  } else if (key === 'e') {
    showEdgeLabels = !showEdgeLabels;
    showStatus(`Edge probabilities ${showEdgeLabels ? 'shown' : 'hidden'}`, 'info');
    changed = true;
  } else if (key === 'l') {
    showNodeLabels = !showNodeLabels;
    showStatus(`Node labels ${showNodeLabels ? 'shown' : 'hidden'}`, 'info');
    changed = true;
  }

  if (changed && currentGraphData && renderer) {
    const threshold = parseFloat(probThresholdInput.value) || 0;
    renderer.render(currentGraphData, threshold, getRenderOptions());
    // Update legend to reflect coloring mode (labels toggles don't affect legend but keep consistent)
    updateLegend(currentGraphData, getNodeColoringMode());
  }
});

// Global tooltip system - renders tooltips as fixed overlays outside the sidebar
const globalTooltip = document.getElementById('global-tooltip') as HTMLDivElement;
let tooltipHideTimeout: number | null = null;

function setupGlobalTooltips() {
  document.addEventListener('mouseenter', (e) => {
    const icon = (e.target as HTMLElement).closest('.info-icon');
    if (!icon) return;
    const tooltipSource = icon.querySelector('.tooltip');
    if (!tooltipSource) return;

    if (tooltipHideTimeout !== null) {
      clearTimeout(tooltipHideTimeout);
      tooltipHideTimeout = null;
    }

    globalTooltip.textContent = '';
    globalTooltip.innerHTML = tooltipSource.innerHTML;

    const rect = icon.getBoundingClientRect();
    const tooltipWidth = 280;

    // Position above the icon by default
    let top = rect.top - 8;
    let left = rect.left + rect.width / 2 - tooltipWidth / 2;

    // Clamp horizontal position to viewport
    if (left < 8) left = 8;
    if (left + tooltipWidth > window.innerWidth - 8) left = window.innerWidth - tooltipWidth - 8;

    // Show tooltip, measure its height, then position above
    globalTooltip.style.width = tooltipWidth + 'px';
    globalTooltip.style.left = left + 'px';
    globalTooltip.style.top = '0px';
    globalTooltip.style.opacity = '1';

    // Measure actual height after rendering
    requestAnimationFrame(() => {
      const tooltipHeight = globalTooltip.offsetHeight;
      let finalTop = rect.top - tooltipHeight - 8;

      // If not enough space above, show below
      if (finalTop < 8) {
        finalTop = rect.bottom + 8;
      }

      globalTooltip.style.top = finalTop + 'px';
    });
  }, true);

  document.addEventListener('mouseleave', (e) => {
    const icon = (e.target as HTMLElement).closest('.info-icon');
    if (!icon) return;
    tooltipHideTimeout = window.setTimeout(() => {
      globalTooltip.style.opacity = '0';
    }, 100);
  }, true);
}

// Initialize
async function init() {
  // Initialize tooltip
  updateTooltip();
  setupGlobalTooltips();

  // Load profiles
  await loadProfiles();
}

init();
