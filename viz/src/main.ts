import { GraphRenderer, buildEmojiMoods, emojiModeSupported } from './graph';
import { fetchProfiles, fetchGraph, profileXlsxUrl } from './api';
import type { GraphData, NodeColoringMode, RingColoringMode } from './types';
import { computeAbsorbingSets } from './absorbing';
import { computeConceptRelations } from './relations';

// UI Elements
const profileSelect = document.getElementById('profile-select') as HTMLSelectElement;
const profileFilter = document.getElementById('profile-filter') as HTMLInputElement;
const downloadXlsxBtn = document.getElementById('download-xlsx-btn') as HTMLButtonElement;
const probThresholdInput = document.getElementById('prob-threshold') as HTMLInputElement;
const filterModeRadios = document.querySelectorAll('input[name="filter-mode"]') as NodeListOf<HTMLInputElement>;
const thresholdTooltip = document.getElementById('threshold-tooltip') as HTMLSpanElement;
const refreshBtn = document.getElementById('refresh-btn') as HTMLButtonElement;
const resetViewBtn = document.getElementById('reset-view-btn') as HTMLButtonElement;
const shareViewBtn = document.getElementById('share-view-btn') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const metadataDiv = document.getElementById('metadata') as HTMLDivElement;
const nodeDetailsDiv = document.getElementById('node-details') as HTMLDivElement;
const selectedStateNameSpan = document.getElementById('selected-state-name') as HTMLSpanElement;
const nodePayoffsDiv = document.getElementById('node-payoffs') as HTMLDivElement;
const outgoingTransitionsDiv = document.getElementById('outgoing-transitions') as HTMLDivElement;
const graphContainer = document.getElementById('graph-container') as HTMLDivElement;
const nodeColoringRadios = document.querySelectorAll('input[name="node-coloring"]') as NodeListOf<HTMLInputElement>;
const ringColoringRadios = document.querySelectorAll('input[name="ring-coloring"]') as NodeListOf<HTMLInputElement>;
const ringLegendDiv = document.getElementById('ring-legend') as HTMLDivElement;
const layoutModeRadios = document.querySelectorAll('input[name="layout-mode"]') as NodeListOf<HTMLInputElement>;
const absorbingLegendDiv = document.getElementById('absorbing-legend') as HTMLDivElement;
const resultIndicatorDiv = document.getElementById('result-indicator') as HTMLDivElement;
const menuToggleBtn = document.getElementById('menu-toggle') as HTMLButtonElement;
const sidebarBackdrop = document.getElementById('sidebar-backdrop') as HTMLDivElement;
const sidebar = document.getElementById('sidebar') as HTMLDivElement;
const sidebarCloseBtn = document.getElementById('sidebar-close') as HTMLButtonElement;
const relationsPanel = document.getElementById('relations-panel') as HTMLDivElement;
const relationsSubtitle = document.getElementById('relations-subtitle') as HTMLParagraphElement;
const relationsTbody = document.getElementById('relations-tbody') as HTMLTableSectionElement;
const relationsHandle = document.getElementById('relations-handle') as HTMLButtonElement;

// State
let currentGraphData: GraphData | null = null;
let renderer: GraphRenderer | null = null;
let previousMetadata: any = null;
let highlightTimeoutId: number | null = null;

// Temporary display toggles (can toggle with keyboard shortcuts)
let showSelfLoops = true;
let showEdgeLabels = true;
let showNodeLabels = true;
let showGeoLevel = true;
const emojiModeCheckbox = document.getElementById('emoji-mode') as HTMLInputElement;
// Both emoji switches are persisted in localStorage so they survive reloads.
const EMOJI_MODE_STORAGE_KEY = 'viz:emoji-mode';
const PAYOFF_EMOJI_STORAGE_KEY = 'viz:short-term-payoff-emoji';
function readStoredFlag(key: string): boolean {
  try { return localStorage.getItem(key) === '1'; } catch { return false; }
}
function storeFlag(key: string, on: boolean) {
  try { localStorage.setItem(key, on ? '1' : '0'); } catch { /* ignore localStorage errors */ }
}
let showEmojiMode = readStoredFlag(EMOJI_MODE_STORAGE_KEY);
emojiModeCheckbox.checked = showEmojiMode; // stored value wins over browser form restoration
// Also remembered across profile switches (the metadata panel is re-rendered each time).
let shortTermPayoffEmoji = readStoredFlag(PAYOFF_EMOJI_STORAGE_KEY);

function getRenderOptions() {
  const filterMode = getFilterMode();
  const coloringMode = getNodeColoringMode();
  const ringColoringMode = getRingColoringMode();
  const layoutMode = getLayoutMode();
  return { coloringMode, ringColoringMode, filterMode, layoutMode, showSelfLoops, showEdgeLabels, showNodeLabels, showGeoLevel, showEmojiMode };
}

// Initialize graph renderer
function initRenderer() {
  if (renderer) {
    renderer.destroy();
  }
  renderer = new GraphRenderer(graphContainer);
  renderer.setOnNodeSelect(handleNodeSelect);
  applyShareParamsFromUrl();
}

/** Share links look like `#<profile>&pos=<code>&emoji=1`: queue the positions and apply the view flags (not persisted). */
function applyShareParamsFromUrl() {
  const params = new URLSearchParams((window.location.hash.slice(1).split('&').slice(1)).join('&'));
  const pos = params.get('pos');
  if (pos && renderer && !renderer.setSharedPositions(pos)) console.warn('Malformed pos parameter in URL; ignored');
  if (params.get('emoji') === '1') {
    showEmojiMode = true;
    emojiModeCheckbox.checked = true;
  }
}

// Get selected filter mode
function getFilterMode(): 'absolute' | 'cumulative' {
  const selected = Array.from(filterModeRadios).find(radio => radio.checked);
  return (selected?.value as 'absolute' | 'cumulative') || 'absolute';
}

// Get selected node coloring mode
function getNodeColoringMode(): NodeColoringMode {
  const selected = Array.from(nodeColoringRadios).find(radio => radio.checked);
  return (selected?.value as NodeColoringMode) || 'none';
}

// Get selected ring coloring mode
function getRingColoringMode(): RingColoringMode {
  const selected = Array.from(ringColoringRadios).find(radio => radio.checked);
  return (selected?.value as RingColoringMode) || 'none';
}

// Persist the last-picked node/ring coloring mode across page loads. Kept to
// just these two radio groups (not every UI toggle) since they're the ones
// most worth not having to re-pick every time the page is reloaded.
const NODE_COLORING_STORAGE_KEY = 'viz:node-coloring-mode';
const RING_COLORING_STORAGE_KEY = 'viz:ring-coloring-mode';

function restoreColoringModeSelections() {
  try {
    const savedNodeMode = localStorage.getItem(NODE_COLORING_STORAGE_KEY);
    if (savedNodeMode) {
      const match = Array.from(nodeColoringRadios).find(radio => radio.value === savedNodeMode);
      if (match) match.checked = true;
    }
    const savedRingMode = localStorage.getItem(RING_COLORING_STORAGE_KEY);
    if (savedRingMode) {
      const match = Array.from(ringColoringRadios).find(radio => radio.value === savedRingMode);
      if (match) match.checked = true;
    }
  } catch { /* ignore localStorage errors (e.g. disabled in private browsing) */ }
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

// Formats a probability (0-1) as a percentage string for display, making a
// nonzero value that rounds to "0.0%" visually distinct from a true zero --
// otherwise e.g. a 0.021% path probability silently reads as exactly 0.
// The full-precision value goes in `title` so hovering reveals the real number.
function formatPercentDisplay(value: number, decimals: number = 1): { text: string; title: string } {
  const pct = value * 100;
  const rounded = pct.toFixed(decimals);
  const roundsToZero = pct > 0 && parseFloat(rounded) === 0;
  const text = roundsToZero ? `<${(1 / Math.pow(10, decimals)).toFixed(decimals)}%` : `${rounded}%`;
  // Show enough precision to be meaningful even for very small probabilities.
  const title = `${pct.toPrecision(3)}%`;
  return { text, title };
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

function getProfileSlug(profilePath: string): string {
  const filename = profilePath.split('/').pop() ?? profilePath;
  return filename.replace(/\.(xlsx|json)$/i, '');
}

function normalizeProfileKey(raw: string): string {
  const decoded = decodeURIComponent(raw.trim());
  const withoutPrefix = decoded.replace(/^#/, '');
  const withoutPath = withoutPrefix.includes('/') ? (withoutPrefix.split('/').pop() ?? withoutPrefix) : withoutPrefix;
  return withoutPath.replace(/\.(xlsx|json)$/i, '').toLowerCase();
}

function getRequestedProfileKeyFromUrl(): string | null {
  if (typeof window === 'undefined') return null;

  // Hash is `<profile>` or `<profile>&pos=...&emoji=1` (share links)
  const hash = window.location.hash?.slice(1)?.split('&')[0]?.trim();
  if (hash) {
    return normalizeProfileKey(hash);
  }

  const params = new URLSearchParams(window.location.search);
  const fromQuery = params.get('profile') || params.get('p') || params.get('xlsx');
  if (!fromQuery) return null;

  return normalizeProfileKey(fromQuery);
}

function findProfilePathByKey(requestedKey: string, profiles: Array<{ path: string; name: string; filename?: string }>): string | null {
  const key = normalizeProfileKey(requestedKey);

  for (const profile of profiles) {
    const candidates = [
      profile.name,
      profile.path,
      profile.filename || '',
      getProfileSlug(profile.path)
    ];
    if (candidates.some(candidate => candidate && normalizeProfileKey(candidate) === key)) {
      return profile.path;
    }
  }

  return null;
}

function syncUrlToSelectedProfile() {
  if (typeof window === 'undefined' || !profileSelect.value) return;

  const slug = getProfileSlug(profileSelect.value);
  const url = new URL(window.location.href);
  url.hash = slug;
  window.history.replaceState(null, '', url.toString());
}

let allProfiles: Array<{ path: string; name: string; filename?: string; category?: string; created_at?: number }> = [];

// Group profiles by category and render as <optgroup>s, newest first within
// each group. Profiles with no category (flat legacy files) go in a
// "general" group at the top.
function renderProfileOptions(profiles: typeof allProfiles, filterText: string) {
  const previousValue = profileSelect.value;
  profileSelect.innerHTML = '';

  const needle = filterText.trim().toLowerCase();
  const filtered = needle
    ? profiles.filter(p => p.name.toLowerCase().includes(needle))
    : profiles;

  if (filtered.length === 0) {
    profileSelect.innerHTML = '<option value="">No matching profiles</option>';
    return;
  }

  const groups = new Map<string, typeof allProfiles>();
  for (const profile of filtered) {
    const key = profile.category || 'general';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(profile);
  }

  const orderedKeys = [...groups.keys()].sort((a, b) => {
    if (a === 'general') return -1;
    if (b === 'general') return 1;
    return a.localeCompare(b);
  });

  for (const key of orderedKeys) {
    const container: HTMLElement = key === 'general'
      ? profileSelect
      : profileSelect.appendChild(Object.assign(document.createElement('optgroup'), { label: key }));
    for (const profile of groups.get(key)!) {
      const option = document.createElement('option');
      option.value = profile.path;
      option.textContent = profile.name;
      container.appendChild(option);
    }
  }

  // Keep the current selection if it still matches the filter.
  if (previousValue && filtered.some(p => p.path === previousValue)) {
    profileSelect.value = previousValue;
  }
}

// Load profiles list
async function loadProfiles() {
  try {
    const data = await fetchProfiles();

    if (data.profiles.length === 0) {
      profileSelect.innerHTML = '<option value="">No profiles found</option>';
      return;
    }

    allProfiles = [...data.profiles]
      .sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));

    renderProfileOptions(allProfiles, profileFilter.value);

    const requestedKey = getRequestedProfileKeyFromUrl();
    if (requestedKey) {
      const matchedPath = findProfilePathByKey(requestedKey, allProfiles);
      if (matchedPath) {
        profileSelect.value = matchedPath;
      } else {
        // Do NOT fall through to auto-loading the first profile: that would silently
        // show a different table (and rewrite the URL) instead of the requested one.
        showStatus(`Profile not found for URL key: ${requestedKey}`, 'error');
        return;
      }
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

    // Clear any previous node selection/details before loading a new profile.
    // Otherwise the sidebar can show stale values from the last graph until the
    // user clicks a node again.
    nodeDetailsDiv.style.display = 'none';
    selectedStateNameSpan.textContent = '';
    nodePayoffsDiv.innerHTML = '';
    outgoingTransitionsDiv.innerHTML = '';

    const graphData = await fetchGraph({
      profile: profileSelect.value
    });

    currentGraphData = graphData;

    // Auto-hide G values if all nodes have 0 geoengineering level
    const hasAnyG = graphData.nodes.some(node => (node.meta?.geo_level || 0) > 0);
    showGeoLevel = hasAnyG;

    // A 52-state graph has enough transitions that labels and self-loops turn
    // the initial view into visual noise (they remain one-key shortcuts: E/S).
    // Smaller profiles retain the detailed default view.
    const isN5 = graphData.nodes.length === 52;
    showEdgeLabels = !isN5;
    showSelfLoops = !isN5;

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
    updateRingLegend(graphData, getRingColoringMode());

    // Update solution-concepts-vs-MPE relations table
    updateRelationsPanel(graphData);

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

function updateLegend(data: GraphData, coloringMode: NodeColoringMode) {
  if (!absorbingLegendDiv) return;

  if (coloringMode === 'none') {
    absorbingLegendDiv.innerHTML = '';
    return;
  }

  if (coloringMode === 'internal' || coloringMode === 'external') {
    // Binary stability legend (same visual language as the combined mode:
    // soft-red stripes = stable on that dimension, gray = not).
    const field = coloringMode === 'internal' ? 'internal' : 'external_consent';
    const title = coloringMode === 'internal' ? 'Internal Stability' : 'External Stability';
    const stripe = coloringMode === 'internal'
      ? 'repeating-linear-gradient(45deg, #f87171 0 6px, #ffffff 6px 10px)'
      : 'repeating-linear-gradient(-45deg, #f87171 0 6px, #ffffff 6px 10px)';
    const stableCount = data.nodes.filter(n => n.meta?.stability?.[field] === true).length;
    const unstableCount = data.nodes.filter(n => n.meta?.stability?.[field] === false).length;

    const items = [];
    if (stableCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:${stripe};display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>Stable (${stableCount} state${stableCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (unstableCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:#cbd5e1;display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>Not stable (${unstableCount} state${unstableCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (items.length === 0) {
      items.push('<div style="color:#999">No stability data for this profile</div>');
    }

    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">${title}</div>
      ${items.join('')}
      <div style="font-size:11px;color:#999;margin-top:8px">Based on static one-period payoffs</div>
    `;
  } else if (coloringMode === 'internal-external') {
    // Combined 4-way legend
    const categorize = (node: typeof data.nodes[number]): 'both' | 'internal' | 'external' | 'neither' | 'unknown' => {
      const st = node.meta?.stability;
      if (!st || typeof st.internal !== 'boolean' || typeof st.external_consent !== 'boolean') return 'unknown';
      const internal = st.internal;
      const external = st.external_consent;
      if (internal && external) return 'both';
      if (internal) return 'internal';
      if (external) return 'external';
      return 'neither';
    };

    const categories: Array<{ key: 'both' | 'internal' | 'external' | 'neither'; label: string; swatch: string }> = [
      { key: 'both', label: 'Internally + externally stable', swatch: '#dc2626' },
      { key: 'internal', label: 'Internally stable only', swatch: 'repeating-linear-gradient(45deg, #f87171 0 6px, #ffffff 6px 10px)' },
      { key: 'external', label: 'Externally stable only', swatch: 'repeating-linear-gradient(-45deg, #f87171 0 6px, #ffffff 6px 10px)' },
      { key: 'neither', label: 'Neither', swatch: '#cbd5e1' },
    ];

    const counts: Record<string, number> = {};
    data.nodes.forEach(node => {
      const c = categorize(node);
      counts[c] = (counts[c] || 0) + 1;
    });

    const items = categories
      .filter(cat => (counts[cat.key] || 0) > 0)
      .map(cat => {
        const count = counts[cat.key] || 0;
        return `<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:${cat.swatch};display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>${cat.label} (${count} state${count !== 1 ? 's' : ''})</span></div>`;
      });

    if (counts['unknown']) {
      items.push(`<div style="color:#999">No stability data: ${counts['unknown']} state${counts['unknown'] !== 1 ? 's' : ''}</div>`);
    }

    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">Internal + External Stability</div>
      ${items.join('')}
      <div style="font-size:11px;color:#999;margin-top:8px">External = consent-based (approval committees)</div>
    `;
  } else if (coloringMode === 'gamma-core') {
    // Binary gamma-core legend (solid teal = in the NTU core, gray = not)
    const inCoreCount = data.nodes.filter(n => n.meta?.gamma_core?.in_core === true).length;
    const outCount = data.nodes.filter(n => n.meta?.gamma_core?.in_core === false).length;
    const unknownCount = data.nodes.filter(
      n => !n.meta?.gamma_core || typeof n.meta.gamma_core.in_core !== 'boolean'
    ).length;

    const items = [];
    if (inCoreCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:#0d9488;display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>In gamma core (${inCoreCount} state${inCoreCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (outCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:#cbd5e1;display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>Not in core (${outCount} state${outCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (items.length === 0) {
      items.push('<div style="color:#999">No gamma-core data for this profile</div>');
    }
    const unknownNote = unknownCount > 0
      ? `<div style="font-size:11px;color:#999;margin-top:6px">Multi-coalition states (no gamma verdict): ${unknownCount} state${unknownCount !== 1 ? 's' : ''}</div>`
      : '';

    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">Gamma Core (NTU)</div>
      ${items.join('')}
      ${unknownNote}
      <div style="font-size:11px;color:#999;margin-top:8px">NTU gamma-core (Chander &amp; Tulkens 1997): no coalition S can strictly improve all its members by breaking away to S + singletons.</div>
    `;
  } else if (coloringMode === 'ricke') {
    // Binary Ricke-winning-coalition legend (solid red = winning, gray = majority but not stable)
    const winningCount = data.nodes.filter(n => n.meta?.ricke?.winning === true).length;
    const majorityNotStableCount = data.nodes.filter(
      n => n.meta?.ricke?.majority === true && n.meta?.ricke?.winning === false
    ).length;
    const unknownCount = data.nodes.filter(
      n => !n.meta?.ricke || typeof n.meta.ricke.majority !== 'boolean' || n.meta.ricke.majority === false
    ).length;

    const items = [];
    if (winningCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:#7c3aed;display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>Ricke winning coalition (${winningCount} state${winningCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (majorityNotStableCount > 0) {
      items.push(`<div style="display:flex;align-items:center;gap:8px;margin:4px 0"><span style="width:16px;height:12px;background:#cbd5e1;display:inline-block;border-radius:2px;border:1px solid #e2e8f0"></span><span>Majority power, not stable (${majorityNotStableCount} state${majorityNotStableCount !== 1 ? 's' : ''})</span></div>`);
    }
    if (items.length === 0) {
      items.push('<div style="color:#999">No Ricke-stability data for this profile</div>');
    }
    const uniquenessNote = winningCount > 1
      ? `<div style="font-size:11px;color:#dc2626;margin-top:6px;font-weight:600">${winningCount} simultaneous winning coalitions -- falsifies Ricke et al.'s uniqueness claim for this game.</div>`
      : '';
    const unknownNote = unknownCount > 0
      ? `<div style="font-size:11px;color:#999;margin-top:6px">Below majority power (no Ricke verdict): ${unknownCount} state${unknownCount !== 1 ? 's' : ''}</div>`
      : '';

    absorbingLegendDiv.innerHTML = `
      <div style="font-weight:600;margin-bottom:6px">Ricke Winning Coalition</div>
      ${items.join('')}
      ${uniquenessNote}
      ${unknownNote}
      <div style="font-size:11px;color:#999;margin-top:8px">Ricke, Moreno-Cruz &amp; Caldeira (2013): majority power share + no member wants to leave (their game has only one active coalition, so leaving collapses onto internal stability).</div>
    `;
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

function updateRingLegend(data: GraphData, ringColoringMode: RingColoringMode) {
  if (!ringLegendDiv) return;

  if (ringColoringMode === 'none') {
    ringLegendDiv.innerHTML = '';
    return;
  }

  // ringColoringMode === 'absorbing'
  const nodeToAbsorbing = computeAbsorbingSets(data);
  const absorbingSetIds = new Set<number>();
  nodeToAbsorbing.forEach(setId => {
    if (setId !== null) absorbingSetIds.add(setId);
  });

  if (absorbingSetIds.size === 0) {
    ringLegendDiv.innerHTML = '<div style="color:#999; padding:6px">No absorbing sets found</div>';
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

  const sortedIds = Array.from(absorbingSetIds).sort((a, b) => a - b);
  const items = sortedIds.map(setId => {
    const members = setMembers.get(setId) || [];
    const label = members.length === 1 ? members[0] : `Set ${setId + 1} (${members.length} states)`;
    return `<div class="ring-legend-item" data-set-id="${setId}" style="display:flex;align-items:center;gap:8px;margin:4px 0;cursor:default"><span style="width:14px;height:10px;border:4px solid #dc2626;border-radius:50%;display:inline-block;background:transparent"></span><span>${label}</span></div>`;
  });
  ringLegendDiv.innerHTML = `<div style="font-weight:600;margin-bottom:6px">Absorbing sets (ring)</div>` + items.join('');

  // Attach hover listeners to legend items to highlight corresponding nodes
  const legendItems = Array.from(ringLegendDiv.querySelectorAll('.ring-legend-item')) as HTMLDivElement[];
  legendItems.forEach(el => {
    el.addEventListener('mouseover', () => {
      const sid = el.getAttribute('data-set-id');
      if (!sid) return;
      const setId = parseInt(sid, 10);
      const members = setMembers.get(setId) || [];
      if (renderer) renderer.highlightNodes(members);
    });
    el.addEventListener('mouseout', () => {
      if (renderer) renderer.clearHighlights();
    });
  });
}

let relationsPanelVisible = false;

function positionRelationsHandle() {
  if (!relationsHandle) return;
  // relationsPanel always sits immediately after the sidebar in flex order,
  // so its right edge IS the seam in both states: when collapsed (width: 0)
  // its right edge coincides exactly with the sidebar's right edge, and when
  // expanded it's the panel's own right edge. Anchoring to the panel alone
  // (rather than switching anchor by visibility) keeps this correct
  // continuously through the width transition, not just at its two endpoints.
  //
  // relationsHandle's containing block is #app (position: relative), so its
  // `left` must be expressed relative to #app's own box, not the viewport --
  // using a bare viewport-relative getBoundingClientRect().right here would
  // overshoot whenever #app itself is offset from the viewport's left edge,
  // which can push this absolutely-positioned element past the content edge
  // and introduce a spurious horizontal scrollbar.
  const appEl = document.getElementById('app') as HTMLDivElement;
  const appRect = appEl.getBoundingClientRect();
  const rect = relationsPanel.getBoundingClientRect();
  relationsHandle.style.left = `${rect.right - appRect.left}px`;
  relationsHandle.textContent = relationsPanelVisible ? '<<' : '>';
  relationsHandle.title = relationsPanelVisible ? 'Hide relations table' : 'Show relations table';
}

function setRelationsPanelVisible(visible: boolean) {
  relationsPanelVisible = visible;
  relationsPanel.classList.toggle('active', visible);
  // Track the handle continuously through the 0.2s width transition (rAF
  // ticks), plus a final correction once it ends, so it never lags behind
  // or snaps to a stale position mid-animation.
  const start = performance.now();
  const duration = 250; // matches styles.css's width transition + margin
  const tick = (now: number) => {
    positionRelationsHandle();
    if (now - start < duration) {
      requestAnimationFrame(tick);
    }
  };
  requestAnimationFrame(tick);
}

function updateRelationsPanel(data: GraphData) {
  if (!relationsPanel || !relationsTbody) return;

  // MPE reference set: nodes in some absorbing set, over the full (unfiltered)
  // transition graph -- the relations table should reflect the actual dynamic
  // equilibrium, not whatever edge-probability threshold is currently applied
  // for display.
  const nodeToAbsorbing = computeAbsorbingSets(data);
  const rows = computeConceptRelations(data, nodeToAbsorbing);

  const mpeCount = [...nodeToAbsorbing.values()].filter(v => v !== null).length;
  relationsSubtitle.textContent = mpeCount > 0
    ? `MPE absorbing set: ${mpeCount} state${mpeCount !== 1 ? 's' : ''}`
    : 'No absorbing set found for this profile';

  relationsTbody.innerHTML = rows.map(row => {
    const sizeStr = row.relation === 'UNDEFINED' ? '–' : String(row.conceptSetSize);
    const jaccardStr = row.jaccard === null ? '–' : row.jaccard.toFixed(3);
    const hammingStr = row.hamming === null ? '–' : String(row.hamming);
    return `
      <tr>
        <td>${row.concept}</td>
        <td>${sizeStr}</td>
        <td class="relation-${row.relation}">${row.relation}</td>
        <td>${jaccardStr}</td>
        <td>${hammingStr}</td>
      </tr>
    `;
  }).join('');

  setRelationsPanelVisible(relationsPanelVisible);
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

  // Check if G should be displayed
  const hasAnyG = currentGraphData?.nodes.some(node => (node.meta?.geo_level || 0) > 0);

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

  let gDisplay = '';
  if (hasAnyG) {
    gDisplay = `<div><strong>E<sub>π</sub>[G]:</strong> ${expectedG.toFixed(2)}°C</div>`;
  } else {
    gDisplay = `<div style="color:#64748b;font-style:italic;">No geoengineering deployment</div>`;
  }

  let html = `
    <div class="result-summary">
      <div>
        ${gDisplay}
        ${convergenceDisplay}
      </div>
      <span class="expand-icon">${expandIcon}</span>
    </div>
  `;

  if (expanded && pi && currentGraphData && hasAnyG) {
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
    if (prevMeta.effectivity_rule !== fileMetadata.effectivity_rule) changedFields.add('effectivity_rule');
    if (prevMeta.min_power !== fileMetadata.min_power) changedFields.add('min_power');
    if (prevConfig.unanimity_required !== currConfig.unanimity_required) changedFields.add('unanimity');
    if (prevMeta.discounting !== fileMetadata.discounting) changedFields.add('discounting');
    if (prevMeta.players !== fileMetadata.players) changedFields.add('players');
    if (previousMetadata.scenario_name !== data.metadata.scenario_name) changedFields.add('scenario');
    if (prevMeta.payoff_table !== fileMetadata.payoff_table) changedFields.add('payoff_table');

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
      <div class="metadata-block${changedFields.has('scenario') ? ' changed-field' : ''}">
        <div><strong>Scenario:</strong> ${scenarioName}</div>
        ${scenarioDescription ? `<div style="font-size: 12px; color: #64748b; margin-top: 4px;">${scenarioDescription}</div>` : ''}
      </div>
    `;
  }

  // Extract player-specific parameters
  const playersStr = fileMetadata.players || data.metadata.num_players || 'N/A';
  const players = typeof playersStr === 'string' ? playersStr.split(',').map(p => p.trim()) : [];

  // When payoffs come from a precomputed table (e.g. RICE50x), the temperature/damage
  // parameters stored in metadata are placeholder values and should not be shown.
  const payoffFromTable = fileMetadata.payoff_source === 'precomputed_table';

  let playerParamsSection = '';
  if (players.length > 0 && !payoffFromTable) {
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

  // Precomputed-table profiles have no meaningful temperature/damage inputs, so
  // show the one-period (short-term) payoffs per state instead, as served by the
  // API (normalised per player to [0,1]).
  if (payoffFromTable && players.length > 0) {
    const nodes = [...data.nodes].sort((a, b) => (a.meta?.index ?? 0) - (b.meta?.index ?? 0));
    // Per-player range of the NON-ZERO payoffs. Zero (the worst state) is pinned
    // to pure red; the non-zero values are stretched over orange -> green by their
    // own min/max, since on a plain 0..1 scale they all land in one narrow band.
    const nzRange: Record<string, [number, number]> = {};
    players.forEach(pl => {
      const nz = nodes.map(n => n.meta?.payoffs?.[pl]).filter((v): v is number => typeof v === 'number' && isFinite(v) && v > 1e-9);
      if (nz.length > 0) nzRange[pl] = [Math.min(...nz), Math.max(...nz)];
    });
    const payoffHue = (pl: string, u: number): number => {
      if (!(u > 1e-9)) return 0; // exact zero: red
      const range = nzRange[pl];
      if (!range || range[1] - range[0] < 1e-9) return 120;
      return 25 + ((u - range[0]) / (range[1] - range[0])) * 95; // 25 (orange) .. 120 (green)
    };
    // Emoji view (n=3 only): same mood scale as the graph's emoji mode.
    const emojiAvailable = emojiModeSupported(data);
    const moods = emojiAvailable ? buildEmojiMoods(data) : null;
    const rows = nodes.map(n => {
      const g = n.meta?.geo_level;
      const gStr = (typeof g === 'number' && isFinite(g)) ? g.toFixed(2) : '—';
      const cells = players.map(pl => {
        const u = n.meta?.payoffs?.[pl];
        if (typeof u !== 'number' || !isFinite(u)) return '<td style="text-align:right;padding:3px 4px;">—</td>';
        const bg = `background:hsl(${payoffHue(pl, u).toFixed(0)},75%,87%);`;
        const mood = moods ? (moods[n.id].find(m => m.player === pl)?.mood ?? (() => { throw new Error(`No mood for player ${pl}`); })()) : '';
        return `<td style="text-align:right;padding:3px 4px;font-variant-numeric:tabular-nums;${bg}"><span class="pay-num">${u.toFixed(3)}</span>${moods ? `<span class="pay-emoji" style="display:none;font-size:14px;">${mood}</span>` : ''}</td>`;
      }).join('');
      return `<tr data-node-id="${n.id}" style="border-bottom:1px solid #e2e8f0;"><td style="padding:3px 4px;white-space:nowrap;">${n.label ?? n.id}</td>${cells}<td style="text-align:right;padding:3px 4px;color:#64748b;">${gStr}</td></tr>`;
    }).join('');
    playerParamsSection = `
      <div style="margin-top:12px;">
        <details open>
          <summary style="font-weight:600;font-size:13px;cursor:pointer;">Short-term payoffs (one period)</summary>
          ${emojiAvailable ? `<label class="checkbox-group" style="margin-top:6px;"><input type="checkbox" id="payoff-emoji-toggle"${shortTermPayoffEmoji ? ' checked' : ''}> Show as emojis</label>` : ''}
          <table id="short-term-payoff-table" style="width:100%;font-size:11px;border-collapse:collapse;margin-top:6px;">
            <thead><tr style="border-bottom:2px solid #e2e8f0;">
              <th style="text-align:left;padding:3px 4px;">State</th>
              ${players.map(pl => `<th style="text-align:right;padding:3px 4px;">${pl}</th>`).join('')}
              <th style="text-align:right;padding:3px 4px;color:#64748b;">G</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <div style="font-size:11px;color:#999;margin-top:4px;">Payoffs normalised per player to [0, 1] (0 = worst, 1 = best); zero is red; non-zero values run orange to green across that country's own range. G = geoengineering level.</div>
        </details>
      </div>
    `;
  }

  metadataDiv.innerHTML = `
    ${scenarioSection}
    <div><strong>Profile:</strong> ${data.metadata.profile_path.split('/').pop()}</div>
    ${payoffFromTable ? `<div${highlight('payoff_table')}><strong>Payoff table:</strong> ${(fileMetadata.payoff_table as string)?.split('/').pop()}</div>` : ''}
    <div${highlight('players')}><strong>Players:</strong> ${playersStr}</div>
    <div><strong>States:</strong> ${data.metadata.num_states}</div>
    <div><strong>Transitions:</strong> ${data.metadata.num_transitions}</div>
    <div${highlight('power_rule')}><strong>Power Rule:</strong> ${fileMetadata.power_rule || data.metadata.config.power_rule}</div>
    ${fileMetadata.min_power ? `<div${highlight('min_power')}><strong>Min Power:</strong> ${fileMetadata.min_power}</div>` : ''}
    ${fileMetadata.effectivity_rule ? `<div${highlight('effectivity_rule')}><strong>Effectivity Rule:</strong> ${fileMetadata.effectivity_rule}</div>` : ''}
    <div${highlight('unanimity')}>
      <strong>Unanimity:</strong> ${unanimityValue ? 'Yes' : 'No'}
      <span class="info-icon" style="margin-left:4px;">
        i
        <span class="tooltip tooltip-unanimity">If Yes, ALL members of the approval committee must approve a proposed transition. If No, only a simple majority is needed (all new members + majority of existing members).</span>
      </span>
    </div>
    ${fileMetadata.discounting ? `<div${highlight('discounting')}><strong>Discounting:</strong> ${fileMetadata.discounting}</div>` : ''}
    ${fileMetadata.verification_success !== undefined ? `<div><strong>Equilibrium verified:</strong> ${fileMetadata.verification_success === true || fileMetadata.verification_success === 'True' ? '<span style="color:#16a34a;font-weight:600;">Yes</span>' : '<span style="color:#dc2626;font-weight:600;">No</span>'}</div>` : ''}
    ${fileMetadata.outer_iterations ? `<div><strong>Iterations:</strong> ${fileMetadata.outer_iterations}</div>` : ''}
    ${fileMetadata.config_hash ? `<div style="font-size: 11px; color: #888;"><strong>Hash:</strong> ${fileMetadata.config_hash}</div>` : ''}
    ${fileMetadata.end_time ? `<div style="font-size: 11px; color: #888;"><strong>Computed:</strong> ${fileMetadata.end_time}</div>` : ''}
    ${playerParamsSection}
  `;

  const payoffEmojiToggle = document.getElementById('payoff-emoji-toggle') as HTMLInputElement | null;
  const applyPayoffEmoji = () => {
    const on = payoffEmojiToggle?.checked ?? false;
    shortTermPayoffEmoji = on;
    storeFlag(PAYOFF_EMOJI_STORAGE_KEY, on);
    metadataDiv.querySelectorAll<HTMLElement>('#short-term-payoff-table .pay-num').forEach(el => { el.style.display = on ? 'none' : ''; });
    metadataDiv.querySelectorAll<HTMLElement>('#short-term-payoff-table .pay-emoji').forEach(el => { el.style.display = on ? '' : 'none'; });
  };
  if (payoffEmojiToggle) {
    payoffEmojiToggle.addEventListener('change', applyPayoffEmoji);
    applyPayoffEmoji();
  }

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
// Mark the selected state's row in the sidebar short-term payoff table (if shown).
function highlightPayoffRow(nodeId: string | null) {
  const rows = metadataDiv.querySelectorAll<HTMLElement>('#short-term-payoff-table tr[data-node-id]');
  rows.forEach(row => {
    const selected = nodeId !== null && row.dataset.nodeId === nodeId;
    row.classList.toggle('payoff-row-selected', selected);
    if (selected) row.scrollIntoView({ block: 'nearest' });
  });
}

function handleNodeSelect(nodeId: string | null) {
  highlightPayoffRow(nodeId);
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
  const payoffFromTableNode = currentGraphData?.metadata?.file_metadata?.payoff_source === 'precomputed_table';
  if (deployingCoalition !== undefined && geoLevel !== undefined) {
    const geoStr = payoffFromTableNode ? '' : ` (G = ${geoLevel.toFixed(3)}°C)`;
    selectedStateNameSpan.innerHTML = `
      ${normalizedStateName}
      <div style="font-size:11px;color:#64748b;margin-top:4px;">
        Deployed by: <strong>${deployingCoalition}</strong>${geoStr}
      </div>
    `;
  } else {
    selectedStateNameSpan.textContent = normalizedStateName;
  }

  // Render payoffs table
  const payoffs: Record<string, number> | undefined = nodeData.data?.payoffs;
  const values: Record<string, number> | undefined = nodeData.data?.values;
  if (payoffs && values) {
    const players = Object.keys(payoffs);
    const rows = players.map(p => {
      const u = payoffs[p];
      const v = values[p];
      const uStr = (u != null && isFinite(u)) ? u.toFixed(4) : '—';
      const vStr = (v != null && isFinite(v)) ? v.toFixed(4) : '—';
      return `<tr>
        <td style="padding:3px 6px;font-weight:600;">${p}</td>
        <td style="padding:3px 6px;text-align:right;font-variant-numeric:tabular-nums;">${uStr}</td>
        <td style="padding:3px 6px;text-align:right;font-variant-numeric:tabular-nums;">${vStr}</td>
      </tr>`;
    }).join('');
    const uHeader = `<th style="padding:3px 6px;text-align:right;color:#64748b;font-weight:600;">Current (u)</th>`;
    nodePayoffsDiv.innerHTML = `
      <table style="width:100%;border-collapse:collapse;font-size:12px;">
        <thead>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <th style="padding:3px 6px;text-align:left;color:#64748b;font-weight:600;">Player</th>
            ${uHeader}
            <th style="padding:3px 6px;text-align:right;color:#64748b;font-weight:600;">Long-term (V)</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } else {
    nodePayoffsDiv.innerHTML = '<div style="color:#999;font-size:12px;">No payoff data available</div>';
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
        const propPct = formatPercentDisplay(entry.propProb);
        const pathPct = formatPercentDisplay(entry.pathProb);
        return `<div style="display:flex;justify-content:space-between;margin-bottom:2px;"><span>${entry.target} (<span title="${propPct.title}">${propPct.text}</span>) \u2192 ${entry.approvalDetail}</span><span style="white-space:nowrap;margin-left:8px;color:#94a3b8;" title="${pathPct.title}">${pathPct.text}</span></div>`;
      }).join('');

      const totalPct = formatPercentDisplay(data.total);

      // Single line: show inline
      if (pathEntries.length === 1) {
        return `
          <div style="font-size:10px;color:#64748b;margin-left:16px;padding:4px 0;border-top:1px solid #e5e7eb;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
              <strong>${proposer} proposes:</strong>
              <span title="${totalPct.title}">${totalPct.text}</span>
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
              <span title="${totalPct.title}">${totalPct.text}</span>
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
        const edgePct = formatPercentDisplay(edge.probability);
        return `
          <div class="transition-item" style="display:block;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <span>${label}</span>
              <div style="display:flex;align-items:center;gap:8px;">
                <span class="prob" title="${edgePct.title}">${edgePct.text}</span>
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

relationsHandle.addEventListener('click', () => setRelationsPanelVisible(!relationsPanelVisible));
window.addEventListener('resize', positionRelationsHandle);
positionRelationsHandle();

shareViewBtn.addEventListener('click', async () => {
  const code = renderer?.exportShareCode();
  if (!code || !profileSelect.value) return;
  const url = new URL(window.location.href);
  url.hash = `${getProfileSlug(profileSelect.value)}&pos=${code}${showEmojiMode ? '&emoji=1' : ''}`;
  try {
    await navigator.clipboard.writeText(url.toString());
    const old = shareViewBtn.textContent;
    shareViewBtn.textContent = 'Link copied!';
    setTimeout(() => { shareViewBtn.textContent = old; }, 2000);
  } catch {
    window.prompt('Copy this link:', url.toString());
  }
});

resetViewBtn.addEventListener('click', () => {
  if (renderer) {
    // Reset also forgets hand-placed bubbles: re-layout from the default positions, then fit.
    renderer.clearUserPositions();
    if (currentGraphData) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      renderer.render(currentGraphData, threshold, getRenderOptions());
    }
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

  // Local mode downloads the real file from the backend; static hosting uses
  // the committed copy. (A missing static file used to fall through to the
  // dev server's index.html, which then got saved under an .xlsx name.)
  const link = document.createElement('a');
  link.href = profileXlsxUrl(profileSelect.value);
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
    updateRingLegend(currentGraphData, getRingColoringMode());
  }
});

// Profile change should auto-refresh
profileSelect.addEventListener('change', async () => {
  syncUrlToSelectedProfile();
  await loadGraph();
});

// Live-filter the profile dropdown by name as the user types.
profileFilter.addEventListener('input', () => {
  renderProfileOptions(allProfiles, profileFilter.value);
});

// Node coloring mode change
nodeColoringRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    try { localStorage.setItem(NODE_COLORING_STORAGE_KEY, getNodeColoringMode()); } catch { /* ignore */ }
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

// Ring coloring mode change
ringColoringRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    try { localStorage.setItem(RING_COLORING_STORAGE_KEY, getRingColoringMode()); } catch { /* ignore */ }
    if (currentGraphData && renderer) {
      const threshold = parseFloat(probThresholdInput.value) || 0;
      // Don't call initRenderer() - positions will be preserved
      renderer.render(currentGraphData, threshold, getRenderOptions());
      updateRingLegend(currentGraphData, getRingColoringMode());
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
      updateRingLegend(currentGraphData, getRingColoringMode());
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

ringColoringRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (window.innerWidth <= 768) {
      setTimeout(closeMobileMenu, 300);
    }
  });
});

emojiModeCheckbox.addEventListener('change', () => {
  if (emojiModeCheckbox.checked && currentGraphData && !emojiModeSupported(currentGraphData)) {
    emojiModeCheckbox.checked = false;
    showStatus('Emoji mode is only available for 3- and 4-player profiles', 'info');
    return;
  }
  showEmojiMode = emojiModeCheckbox.checked;
  storeFlag(EMOJI_MODE_STORAGE_KEY, showEmojiMode);
  if (currentGraphData && renderer) {
    const threshold = parseFloat(probThresholdInput.value) || 0;
    renderer.render(currentGraphData, threshold, getRenderOptions());
  }
});

// Keyboard shortcuts for quick toggles (temporary display changes)
// s = toggle self-loops, e = toggle edge labels, l = toggle node labels
document.addEventListener('keydown', (ev) => {
  if (ev.ctrlKey || ev.altKey || ev.metaKey) return;
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
  } else if (key === 'm') {
    if (currentGraphData && !emojiModeSupported(currentGraphData)) {
      showStatus('Emoji mode is only available for 3- and 4-player profiles', 'info');
    } else {
      showEmojiMode = !showEmojiMode;
      emojiModeCheckbox.checked = showEmojiMode;
      storeFlag(EMOJI_MODE_STORAGE_KEY, showEmojiMode);
      showStatus(`Emoji mode ${showEmojiMode ? 'on' : 'off'}`, 'info');
      changed = true;
    }
  } else if (key === 'g') {
    showGeoLevel = !showGeoLevel;
    showStatus(`G values ${showGeoLevel ? 'shown' : 'hidden'}`, 'info');
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
  // Restore the last-picked coloring modes before the first graph loads, so
  // it renders with the right selection from the start rather than flashing
  // the default and then re-rendering.
  restoreColoringModeSelections();

  // Initialize tooltip
  updateTooltip();
  setupGlobalTooltips();

  window.addEventListener('hashchange', () => {
    const requestedKey = getRequestedProfileKeyFromUrl();
    if (!requestedKey) return;

    const optionValues = Array.from(profileSelect.options).map(opt => opt.value);
    const matched = optionValues.find(value => normalizeProfileKey(getProfileSlug(value)) === requestedKey);
    if (matched && matched !== profileSelect.value) {
      profileSelect.value = matched;
      loadGraph();
    }
  });

  // Load profiles
  await loadProfiles();
}

init();
