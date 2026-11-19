import cytoscape, { Core, NodeSingular } from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import cola from 'cytoscape-cola';
import type { GraphData, NodeColoringMode, RingColoringMode } from './types';
import { computeAbsorbingSets } from './absorbing';

// Register layout algorithms
cytoscape.use(coseBilkent);
cytoscape.use(cola);

const PENTAGON_COORDS = [
  { x: 0,    y: -180 },
  { x: 171,  y: -55  },
  { x: 106,  y:  145 },
  { x: -106, y:  145 },
  { x: -171, y: -55  },
];

const STABILITY_RED = '#dc2626';
const STABILITY_RED_SOFT = '#f87171';
const STABILITY_NEUTRAL = '#cbd5e1';
// Distinct from STABILITY_RED: the red is reserved for the MPE absorbing-set
// ring, so the "both internal and external stable" solid fill (and its
// striped single-dimension variants) use orange instead, to avoid the two
// concepts reading as visually identical.
// Amber rather than a red-leaning orange, for better contrast against the
// red absorbing-set ring.
const STABILITY_ORANGE = '#f59e0b';
const STABILITY_ORANGE_SOFT = '#fbbf24';
const ABSORBING_RING_COLOR = '#dc2626';
// Ricke's winning-coalition verdict gets its own color, distinct from the red
// used by internal/external/gamma-core and the absorbing-set ring.
const RICKE_PURPLE = '#7c3aed';
// Gamma-core verdict also gets its own colour (teal): solid red is reserved for
// the MPE absorbing-set ring, so the core must not read as the same thing.
const GAMMA_CORE_TEAL = '#0d9488';

// Soft-color-with-white-diagonal-stripes tile, used to distinguish
// internal-only (rotate 45°) from external-only (rotate -45°) in the
// single-dimension and combined stability colourings. The fill is a lighter
// shade so the striped states recede visually behind the solid "both stable"
// states.
function stabilityStripeDataUri(rotateDeg: 45 | -45, fill: string, size = 70, overlay = ''): string {
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='${size}' height='${size}'>` +
    `<defs><pattern id='s' width='14' height='14' patternUnits='userSpaceOnUse' patternTransform='rotate(${rotateDeg})'>` +
    `<rect width='14' height='14' fill='${fill}'/>` +
    `<rect width='5' height='14' fill='#ffffff'/>` +
    `</pattern></defs>` +
    `<rect width='${size}' height='${size}' fill='url(#s)'/>` +
    overlay +
    `</svg>`;
  return 'data:image/svg+xml,' + encodeURIComponent(svg);
}

const TRIANGLE_COORDS = {
  N: { x: 0, y: -160 },
  A: { x: -140, y: 110 },
  B: { x: 140, y: 110 },
};

// Standard player order for normalization (single-character player names only)
const STANDARD_ORDER = ['H', 'W', 'T', 'C', 'F', 'A', 'B', 'D', 'E', 'G'];

function normalizeStateName(stateName: string): string {
  /**
   * Normalize state name to use standard player order (H, W, T, C, F, ...).
   * E.g., "(CTW)" -> "(WTC)", "(CF)(TW)" -> "(CF)(TW)"
   *
   * For multi-character player names (e.g. NDE, USA, RUS) the backend already
   * produces canonical names sorted alphabetically, so we return them unchanged.
   * Detection: if any character inside the parens is NOT in STANDARD_ORDER, the
   * coalition uses multi-char names and must not be re-split.
   */
  if (stateName === '( )') return stateName;

  // Match coalitions including those with digits (e.g. GOLF57)
  const coalitionMatches = stateName.match(/\([A-Z0-9]+\)/g);
  if (!coalitionMatches) return stateName;

  const normalizedCoalitions = coalitionMatches.map(coalition => {
    const content = coalition.slice(1, -1);
    const chars = content.split('');
    // Only re-sort if every character is a known single-char player
    if (!chars.every(c => STANDARD_ORDER.includes(c))) {
      return coalition; // Multi-char names: already canonical, return unchanged
    }
    const sorted = chars.sort((a, b) => STANDARD_ORDER.indexOf(a) - STANDARD_ORDER.indexOf(b));
    return `(${sorted.join('')})`;
  });

  return normalizedCoalitions.join('');
}

// Export for use in other modules
(window as any).normalizeStateName = normalizeStateName;

// Preset positions for 5-state case (3-country model)
// Pentagon layout: ( ) top, pair upper-right, pair lower-right,
//                  grand coalition lower-left, pair upper-left
function getPentagonPositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 5) return {};

  const emptyState = stateNames.find(s => s === '( )');
  if (!emptyState) return {};

  const others = stateNames.filter(s => s !== '( )');

  // Grand coalition = longest state name (contains all players concatenated)
  const maxLen = Math.max(...others.map(s => s.length));
  const grandCoalition = others.find(s => s.length === maxLen);
  if (!grandCoalition) return {};

  // Remaining are 2-player pairs — sort alphabetically for deterministic layout
  const pairs = others.filter(s => s !== grandCoalition).sort();
  if (pairs.length !== 3) return {};

  // Arrange: ( ) top, pairs at upper-right / lower-right / upper-left,
  //          grand coalition at lower-left (matches original paper layout)
  const ordered = [emptyState, pairs[0], pairs[1], grandCoalition, pairs[2]];
  const positionMap: Record<string, { x: number; y: number }> = {};
  ordered.forEach((state, i) => { positionMap[state] = PENTAGON_COORDS[i]; });
  return positionMap;
}

// Preset positions for reduced 4-state case (3-country model)
// Reuse the 5-state pentagon layout, omitting the grand coalition slot.
function getReducedPentagonPositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 4) return {};

  const emptyState = stateNames.find(s => s === '( )');
  if (!emptyState) return {};

  const pairs = stateNames.filter(s => s !== '( )').sort();
  if (pairs.length !== 3) return {};
  return {
    [emptyState]: PENTAGON_COORDS[0],
    [pairs[0]]: PENTAGON_COORDS[1],
    [pairs[1]]: PENTAGON_COORDS[2],
    [pairs[2]]: PENTAGON_COORDS[4],
  };
}

function getReducedTrianglePositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 3) return {};
  const required = ['N', 'A', 'B'];
  if (!required.every(state => stateNames.includes(state))) return {};
  return {
    N: TRIANGLE_COORDS.N,
    A: TRIANGLE_COORDS.A,
    B: TRIANGLE_COORDS.B,
  };
}

// Two-circle layout for n=4 case (15 states)
// Inner circle with 5 states, outer circle with 10 states
function getTwoCirclePositions(stateNames: string[]): Record<string, { x: number; y: number }> {
  if (stateNames.length !== 15) return {};
  const positions: Record<string, { x: number; y: number }> = {};

  // Inner preferred order (n=3 pentagon equivalents) - keep same layout
  const innerPreferred = ['( )', '(CT)', '(TW)', '(CTW)', '(CW)'];

  // Pick inner states as those not containing 'F' and matching preferred order first
  const innerStates: string[] = innerPreferred.filter(s => stateNames.includes(s));

  // If fewer than 5, add other states that do not include 'F'
  const nonFStates = stateNames.filter(s => !s.includes('F') && !innerStates.includes(s));
  for (const s of nonFStates) {
    if (innerStates.length >= 5) break;
    innerStates.push(s);
  }

  // If still fewer than 5, fill with any remaining states (shouldn't usually happen)
  for (const s of stateNames) {
    if (innerStates.length >= 5) break;
    if (!innerStates.includes(s)) innerStates.push(s);
  }

  const outerStates = stateNames.filter(s => !innerStates.includes(s)).sort();

  // Try to reuse pentagon positions for inner states (keeps identical layout to n=3)
  const pent = getPentagonPositions(innerStates);
  if (Object.keys(pent).length === innerStates.length) {
    // Use pentagon coordinates
    innerStates.forEach(state => { positions[state] = pent[state]; });
  } else {
    // Fallback: evenly spaced inner circle
    const innerRadius = 150;
    innerStates.forEach((state, i) => {
      const angle = (i / innerStates.length) * 2 * Math.PI - Math.PI / 2;
      positions[state] = { x: Math.cos(angle) * innerRadius, y: Math.sin(angle) * innerRadius };
    });
  }

  // Place outer circle states
  const outerRadius = 300;
  outerStates.forEach((state, i) => {
    const angle = (i / outerStates.length) * 2 * Math.PI - Math.PI / 2;
    positions[state] = { x: Math.cos(angle) * outerRadius, y: Math.sin(angle) * outerRadius };
  });

  return positions;
}

type TaxonomyLabel = { id: string; label: string; x: number; y: number; width?: number };

/**
 * Static, labelled taxonomy for all 52 partitions of five players.
 *
 * The primary progression is the number of countries participating in a
 * non-singleton coalition. The two ways to involve four or five countries
 * are deliberately given their own labelled subcolumns.
 */
function getN5Taxonomy(
  stateNames: string[], playerCodes: string[]
): { positions: Record<string, { x: number; y: number }>; labels: TaxonomyLabel[] } {
  const codes = playerCodes.slice().sort((a, b) => b.length - a.length);
  const coalitionSizes = (state: string): number[] => {
    const groups = state.match(/\([A-Z0-9]+\)/g) || [];
    return groups.map(group => {
      let rest = group.slice(1, -1);
      let size = 0;
      while (rest) {
        const code = codes.find(candidate => rest.startsWith(candidate));
        if (!code) return rest.length; // legacy defensive fallback
        rest = rest.slice(code.length);
        size += 1;
      }
      return size;
    });
  };
  const partition = (state: string): string => {
    if (state === '( )') return '0';
    return coalitionSizes(state).sort((a, b) => b - a).join('+');
  };

  // Each subcolumn has exactly one structural meaning. The y-grid within it
  // is alphabetical only, so it makes no claim about a transition ordering.
  const lanes: Record<string, { x: number; y: number; columns: number }> = {
    '0':   { x: 0,    y: 70, columns: 1 },
    '2':   { x: 220,  y: 70, columns: 2 },
    '3':   { x: 450,  y: 70, columns: 2 },
    '4':   { x: 680,  y: 70, columns: 1 },
    '2+2': { x: 900,  y: 70, columns: 3 },
    '5':   { x: 1130, y: 70, columns: 1 },
    '3+2': { x: 1350, y: 70, columns: 2 },
  };
  const byPartition = new Map<string, string[]>();
  stateNames.forEach(state => {
    const key = partition(state);
    const states = byPartition.get(key) || [];
    states.push(state);
    byPartition.set(key, states);
  });

  const positions: Record<string, { x: number; y: number }> = {};
  byPartition.forEach((states, key) => {
    const lane = lanes[key];
    if (!lane) return;
    states.sort();
    const rows = Math.ceil(states.length / lane.columns);
    states.forEach((state, index) => {
      const col = index % lane.columns;
      const row = Math.floor(index / lane.columns);
      positions[state] = {
        x: lane.x + (col - (lane.columns - 1) / 2) * 82,
        y: lane.y + (row - (rows - 1) / 2) * 76,
      };
    });
  });

  return {
    positions,
    labels: [
      { id: '__tax_0', label: '0 involved — none', x: 0, y: -230 },
      { id: '__tax_2', label: '2 involved — one pair', x: 220, y: -230 },
      { id: '__tax_3', label: '3 involved — one triple', x: 450, y: -230 },
      { id: '__tax_4', label: '4 involved', x: 790, y: -310, width: 350 },
      { id: '__tax_4a', label: 'one bloc of 4', x: 680, y: -230 },
      { id: '__tax_4b', label: 'two pairs (2 + 2)', x: 900, y: -230 },
      { id: '__tax_5', label: '5 involved', x: 1240, y: -310, width: 350 },
      { id: '__tax_5a', label: 'grand coalition (5)', x: 1130, y: -230 },
      { id: '__tax_5b', label: 'triple + pair (3 + 2)', x: 1350, y: -230 },
    ]
  };
}

// Full country names make five-player labels dominate the graph. Keep them
// available in the state-detail panel, but use unambiguous short forms in a
// node. (CHN needs two letters to avoid colliding with the C player.)
function compactN5StateLabel(stateName: string, playerCodes: string[]): string {
  const shortCode: Record<string, string> = {
    BRA: 'B', CAN: 'C', CHN: 'Ch', EUR: 'E', JPN: 'J',
    NDE: 'D', RUS: 'R', USA: 'U',
  };
  const codes = playerCodes.slice().sort((a, b) => b.length - a.length);
  return stateName.replace(/\(([A-Z0-9]+)\)/g, (_match, content: string) => {
    let rest = content;
    const compact: string[] = [];
    while (rest) {
      const code = codes.find(candidate => rest.startsWith(candidate));
      if (!code) return `(${content})`; // preserve rather than mislabel
      compact.push(shortCode[code] || code[0]);
      rest = rest.slice(code.length);
    }
    return `(${compact.join('')})`;
  });
}

const COUNTRY_FLAGS: Record<string, string> = {
  BRA: '\u{1F1E7}\u{1F1F7}', CAN: '\u{1F1E8}\u{1F1E6}', CHN: '\u{1F1E8}\u{1F1F3}',
  EUR: '\u{1F1EA}\u{1F1FA}', JPN: '\u{1F1EF}\u{1F1F5}', NDE: '\u{1F1EE}\u{1F1F3}',
  RUS: '\u{1F1F7}\u{1F1FA}', USA: '\u{1F1FA}\u{1F1F8}',
};

// Mood by the rank of a state's static payoff among the player's distinct
// payoffs (1 = best). Best = star-eyes; the worst distinct value (exactly 0
// once payoffs are normalised per player) = crying; ranks in between use
// MOOD_MIDDLE in order. n=3 has at most 3 middle ranks, n=4 at most 4.
const MOOD_BEST = '\u{1F929}';
const MOOD_WORST = '\u{1F62D}';
const MOOD_MIDDLE = ['\u{1F610}', '\u{1F615}', '\u{1F61F}', '\u{1F629}'];

/** Emoji mode needs a known player count (flag strip layouts exist for 3 and 4). */
export function emojiModeSupported(graphData: GraphData): boolean {
  const n = Object.keys(graphData.nodes[0]?.meta?.payoffs || {}).length;
  return n === 3 || n === 4;
}

type EmojiItem = { player: string; flag: string; mood: string; member: boolean };

export function buildEmojiMoods(graphData: GraphData): Record<string, EmojiItem[]> {
  const players = Object.keys(graphData.nodes[0]?.meta?.payoffs || {});
  if (!emojiModeSupported(graphData)) {
    throw new Error(`Emoji mode supports 3 or 4 players, got ${players.length}`);
  }
  const EPS = 1e-9;
  const rankOf: Record<string, number[]> = {};
  for (const pl of players) {
    const vals = graphData.nodes.map(n => n.meta!.payoffs![pl]);
    const distinct: number[] = [];
    for (const v of vals.slice().sort((a, b) => b - a)) {
      if (!distinct.length || distinct[distinct.length - 1] - v > EPS) distinct.push(v);
    }
    rankOf[pl] = vals.map(v => distinct.findIndex(d => Math.abs(d - v) <= EPS) + 1);
  }
  const out: Record<string, EmojiItem[]> = {};
  graphData.nodes.forEach((n, i) => {
    out[n.id] = players.map(pl => {
      const rank = rankOf[pl][i];
      const nRanks = Math.max(...rankOf[pl]);
      const payoff = n.meta!.payoffs![pl];
      let mood: string | undefined;
      if (rank === 1) mood = MOOD_BEST;
      else if (rank === nRanks || Math.abs(payoff) <= EPS) mood = MOOD_WORST;
      else mood = MOOD_MIDDLE[rank - 2];
      if (!mood) throw new Error(`No mood for payoff rank ${rank} of ${nRanks} (${pl})`);
      // Members = every player listed inside a coalition, e.g. "(CHNEUR)(NDEUSA)".
      const member = coalitionMembers(n.label, players).includes(pl);
      return { player: pl, flag: COUNTRY_FLAGS[pl] ?? pl, mood, member };
    });
  });
  return out;
}

// Split a coalition name such as "(EURUSA)" into player codes. A substring test
// is wrong here ("RUS" occurs inside "EURUSA"), so segment the whole string.
function coalitionMembers(stateLabel: string, players: string[]): string[] {
  const split = (rest: string): string[] | null => {
    if (!rest) return [];
    for (const pl of players) {
      if (rest.startsWith(pl)) {
        const tail = split(rest.slice(pl.length));
        if (tail) return [pl, ...tail];
      }
    }
    return null;
  };
  // A state can hold several coalitions, e.g. "(CHNEUR)(NDEUSA)".
  const members: string[] = [];
  for (const m of stateLabel.matchAll(/\(([^)]*)\)/g)) {
    const part = split(m[1].trim());
    if (!part) throw new Error(`Cannot parse coalition "${stateLabel}" into players ${players.join(',')}`);
    members.push(...part);
  }
  return members;
}

// One SVG strip per state: each country is a flag with its mood emoji
// overlapping the flag's lower-right corner, which saves width over text.
// Strip geometry per player count: 3 players in a row, 4 players in a 2x2 grid
// (a 4-wide row would need a much wider circle).
function emojiStripLayout(nPlayers: number) {
  const cols = nPlayers === 4 ? 2 : nPlayers;
  const rows = nPlayers === 4 ? 2 : 1;
  const pitchX = 27, pitchY = 18;
  const w = pitchX * cols + 4, h = pitchY * (rows - 1) + 32;
  // Node diameter, strip top, label offset below the node centre (all in px).
  const geom = nPlayers === 4
    ? { nodeSize: 90, stripTop: 12, labelMargin: 18 }
    : { nodeSize: 86, stripTop: 21, labelMargin: 17 };
  return { cols, pitchX, pitchY, w, h, ...geom };
}

function emojiStripInner(items: EmojiItem[]): string {
  const L = emojiStripLayout(items.length);
  const cells = items.map((it, i) => {
    const x = 2 + (i % L.cols) * L.pitchX;
    const yOff = Math.floor(i / L.cols) * L.pitchY;
    // 1px outline around the flag of each country in a coalition.
    const outline = it.member
      ? `<rect x="${x + 0.5}" y="${yOff + 3.5}" width="21" height="13.5" fill="none" stroke="#ffe600" stroke-width="1"/>`
      : '';
    return `<text x="${x}" y="${yOff + 17}" font-size="18">${it.flag}</text>${outline}`
      + `<text x="${x + 6}" y="${yOff + 17}" font-size="16">${it.mood}</text>`;
  }).join('');
  return cells;
}

function emojiSvgDataUri(items: EmojiItem[]): string {
  const L = emojiStripLayout(items.length);
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${L.w}" height="${L.h}">${emojiStripInner(items)}</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

// Stability-striped nodes already use the node's single background image for
// the stripes, so the emoji strip is baked into a node-sized stripe image
// (same stripe scale as normal nodes, emoji at the same position).
function emojiOnStripeDataUri(items: EmojiItem[], pattern: string): string {
  const L = emojiStripLayout(items.length);
  const rotate = pattern.startsWith('internal') ? 45 : -45;
  const fill = pattern.endsWith('red') ? STABILITY_RED_SOFT : STABILITY_ORANGE_SOFT;
  const overlay = `<svg x='${(L.nodeSize - L.w) / 2}' y='${L.stripTop}' width='${L.w}' height='${L.h}'>${emojiStripInner(items)}</svg>`;
  return stabilityStripeDataUri(rotate, fill, L.nodeSize, overlay);
}

function getN5PlayerCodes(graphData: GraphData): string[] {
  const listed = graphData.metadata.file_metadata?.players;
  if (typeof listed === 'string') {
    const codes = listed.split(',').map(code => code.trim()).filter(Boolean);
    if (codes.length === 5) return codes;
  }
  // Kept only for legacy data that predates the `players` metadata field.
  return ['CHN', 'EUR', 'NDE', 'RUS', 'USA'];
}

// Seed for deterministic force-directed layout (controls both initial positions and
// internal random decisions in cose-bilkent, ensuring identical results across devices).
const FORCE_LAYOUT_SEED = 42;

// Module-level cache for force-directed layout positions.
// Survives initRenderer() calls. Keyed by a string derived from node IDs + edge structure.
const forceDirectedCache = new Map<string, Record<string, { x: number; y: number }>>();

function getGraphCacheKey(graphData: GraphData): string {
  const nodeIds = graphData.nodes.map(n => n.id).sort().join(',');
  const edgeIds = graphData.edges.map(e => `${e.source}->${e.target}`).sort().join(',');
  return `${nodeIds}|${edgeIds}`;
}

// Seeded PRNG (mulberry32) to make force-directed layout deterministic across machines.
// Temporarily replaces Math.random while cose-bilkent runs.
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return function() {
    s = s + 0x6D2B79F5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function withSeededRandom<T>(seed: number, fn: () => T): T {
  const originalRandom = Math.random;
  Math.random = mulberry32(seed);
  try {
    return fn();
  } finally {
    Math.random = originalRandom;
  }
}

export class GraphRenderer {
  private cy: Core | null = null;
  private container: HTMLElement;
  private selectedNode: string | null = null;
  private onNodeSelect: ((nodeId: string | null) => void) | null = null;
  private savedPositions: Record<string, { x: number; y: number }> = {};
  // Positions the user dragged by hand. Persisted in localStorage (keyed by the
  // set of states, so the same layout is reused across profiles of one n) and
  // applied above every other position source until Reset View clears them.
  private userPositions: Record<string, { x: number; y: number }> = {};
  private userPositionsKey: string | null = null;
  // Set by clearUserPositions(): the next render must not re-save the live node
  // positions (that would immediately undo the reset).
  private discardLivePositions = false;
  private savedZoom: number | null = null;
  private savedPan: { x: number; y: number } | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  setOnNodeSelect(callback: (nodeId: string | null) => void) {
    this.onNodeSelect = callback;
  }

  private static storageKey(stateKey: string) {
    return `node-positions:${stateKey}`;
  }

  private loadUserPositions(stateKey: string) {
    this.userPositionsKey = stateKey;
    this.userPositions = {};
    try {
      const stored = localStorage.getItem(GraphRenderer.storageKey(stateKey));
      if (stored) this.userPositions = JSON.parse(stored);
    } catch { /* localStorage unavailable or corrupt: behave as if nothing was saved */ }
  }

  private persistUserPositions() {
    if (!this.userPositionsKey) return;
    try {
      localStorage.setItem(GraphRenderer.storageKey(this.userPositionsKey), JSON.stringify(this.userPositions));
    } catch { /* ignore localStorage errors */ }
  }

  /** Forget all hand-placed positions (memory + localStorage); used by Reset View. */
  clearUserPositions() {
    this.userPositions = {};
    this.savedPositions = {};
    this.discardLivePositions = true;
    if (!this.userPositionsKey) return;
    try {
      localStorage.removeItem(GraphRenderer.storageKey(this.userPositionsKey));
    } catch { /* ignore localStorage errors */ }
  }

  private saveCurrentPositions() {
    if (!this.cy) return;
    this.savedPositions = {};
    this.cy.nodes().forEach(node => {
      const pos = node.position();
      this.savedPositions[node.id()] = { x: pos.x, y: pos.y };
    });
    // Save zoom and pan state
    this.savedZoom = this.cy.zoom();
    this.savedPan = this.cy.pan();
  }

  private computeClusterPositions(
    layoutMode: 'default' | 'connections' | 'deployer' | 'geo-level',
    graphData: GraphData
  ): Record<string, { x: number; y: number }> {
    if (layoutMode !== 'deployer' && layoutMode !== 'geo-level') {
      return {};
    }

    // Group nodes by cluster key (deployer or G level)
    const nodesByCluster = new Map<string, string[]>();
    graphData.nodes.forEach(node => {
      let clusterKey: string;
      if (layoutMode === 'deployer') {
        clusterKey = node.meta?.deploying_coalition || 'None';
      } else {
        const gLevel = node.meta?.geo_level || 0;
        clusterKey = Math.round(gLevel).toString();
      }

      if (!nodesByCluster.has(clusterKey)) {
        nodesByCluster.set(clusterKey, []);
      }
      nodesByCluster.get(clusterKey)!.push(node.id);
    });

    // Position cluster centers in a large circle
    const clusters = Array.from(nodesByCluster.keys());
    const numClusters = clusters.length;
    const clusterRadius = 280; // Large radius for separation

    const clusterCenters = new Map<string, {x: number, y: number}>();
    clusters.forEach((cluster, i) => {
      const angle = (i / numClusters) * 2 * Math.PI - Math.PI / 2;
      clusterCenters.set(cluster, {
        x: Math.cos(angle) * clusterRadius,
        y: Math.sin(angle) * clusterRadius
      });
    });

    // Pre-position nodes near their cluster centers
    const nodePositions: Record<string, {x: number, y: number}> = {};
    const nodeDiameter = 70; // Node size from styles

    nodesByCluster.forEach((nodeIds, cluster) => {
      const center = clusterCenters.get(cluster)!;
      const numNodes = nodeIds.length;

      // Calculate minimum radius to prevent overlap
      // Chord length between adjacent nodes must be > nodeDiameter
      // Chord = 2R * sin(π/N), so R = (nodeDiameter * margin) / (2 * sin(π/N))
      let nodeRadius: number;
      if (numNodes === 1) {
        nodeRadius = 0; // Single node at center
      } else {
        const margin = 1.3; // 30% spacing margin
        const minChordLength = nodeDiameter * margin;
        nodeRadius = minChordLength / (2 * Math.sin(Math.PI / numNodes));
      }

      nodeIds.forEach((nodeId, i) => {
        const angle = (i / numNodes) * 2 * Math.PI;
        nodePositions[nodeId] = {
          x: center.x + Math.cos(angle) * nodeRadius,
          y: center.y + Math.sin(angle) * nodeRadius
        };
      });
    });

    return nodePositions;
  }

  render(graphData: GraphData, probThreshold: number = 0.001, options?: {
    coloringMode?: NodeColoringMode;
    ringColoringMode?: RingColoringMode;
    filterMode?: 'absolute' | 'cumulative';
    layoutMode?: 'default' | 'connections' | 'deployer' | 'geo-level';
    // UI toggles
    showSelfLoops?: boolean;
    showEdgeLabels?: boolean;
    showNodeLabels?: boolean;
    showGeoLevel?: boolean;
    showEmojiMode?: boolean;
  }) {
    // Save current positions before destroying
    if (this.cy) {
      if (this.discardLivePositions) {
        this.discardLivePositions = false;
      } else {
        this.saveCurrentPositions();
      }
      this.cy.destroy();
      this.cy = null;
    }

    const filterMode = options?.filterMode || 'absolute';
    const coloringMode = options?.coloringMode || 'none';
    const ringColoringMode = options?.ringColoringMode || 'none';
    const layoutMode = options?.layoutMode || 'default';
    const showSelfLoops = options?.showSelfLoops !== false; // default true
    const showEdgeLabels = options?.showEdgeLabels !== false; // default true
    const showNodeLabels = options?.showNodeLabels !== false; // default true
    const showGeoLevel = options?.showGeoLevel !== false; // default true
    // Emoji mode is n=3 only; anything else silently keeps the normal labels.
    const showEmojiMode = options?.showEmojiMode === true && emojiModeSupported(graphData);
    const emojiLayout = showEmojiMode ? emojiStripLayout(Object.keys(graphData.nodes[0].meta!.payoffs!).length) : null;
    const emojiMoods = showEmojiMode ? buildEmojiMoods(graphData) : null;

    // Filter edges by probability threshold
    let filteredEdges: typeof graphData.edges;

    if (filterMode === 'cumulative') {
      // Cumulative tail filtering: for each target node, remove smallest incoming edges until sum reaches threshold
      const edgesToRemove = new Set<string>();

      // Group edges by target node
      const edgesByTarget = new Map<string, typeof graphData.edges>();
      for (const edge of graphData.edges) {
        if (!edgesByTarget.has(edge.target)) {
          edgesByTarget.set(edge.target, []);
        }
        edgesByTarget.get(edge.target)!.push(edge);
      }

      // For each target node, identify edges to remove
      for (const [target, edges] of edgesByTarget) {
        // Sort by probability ascending (smallest first)
        const sorted = [...edges].sort((a, b) => a.p - b.p);

        let cumSum = 0;
        for (const edge of sorted) {
          if (cumSum + edge.p <= probThreshold) {
            // Remove this edge
            edgesToRemove.add(edge.id);
            cumSum += edge.p;
          } else {
            // Stop - keep this and all remaining edges
            break;
          }
        }
      }

      filteredEdges = graphData.edges.filter(e => !edgesToRemove.has(e.id));
    } else {
      // Absolute threshold filtering
      filteredEdges = graphData.edges.filter(e => e.p >= probThreshold);
    }

    const n5PlayerCodes = graphData.nodes.length === 52 ? getN5PlayerCodes(graphData) : [];

    const useN5Taxonomy = graphData.nodes.length === 52 && layoutMode === 'default';
    const n5Taxonomy = useN5Taxonomy ? getN5Taxonomy(graphData.nodes.map(n => n.id), n5PlayerCodes) : null;

    // Get preset positions for specific cases
    let presetPositions: Record<string, { x: number; y: number }> = {};
    if (graphData.nodes.length === 3) {
      presetPositions = getReducedTrianglePositions(graphData.nodes.map(n => n.id));
    } else if (graphData.nodes.length === 4) {
      presetPositions = getReducedPentagonPositions(graphData.nodes.map(n => n.id));
    } else if (graphData.nodes.length === 5) {
      presetPositions = getPentagonPositions(graphData.nodes.map(n => n.id));
    } else if (graphData.nodes.length === 15) {
      presetPositions = getTwoCirclePositions(graphData.nodes.map(n => n.id));
    } else if (n5Taxonomy) {
      presetPositions = n5Taxonomy.positions;
    }

    // Compute cluster positions for clustering layouts
    const clusterPositions = this.computeClusterPositions(layoutMode, graphData);

    // Check force-directed cache (in-memory, falls back to localStorage)
    const cacheKey = getGraphCacheKey(graphData);
    let cachedForcePositions = forceDirectedCache.get(cacheKey);
    if (!cachedForcePositions) {
      try {
        const stored = localStorage.getItem(`force-layout:${cacheKey}`);
        if (stored) {
          cachedForcePositions = JSON.parse(stored);
          forceDirectedCache.set(cacheKey, cachedForcePositions!);
        }
      } catch { /* ignore localStorage errors */ }
    }

    // Position priority: saved > cached force-directed > cluster/preset positions
    // This ensures that user interactions (dragging) or previous renders are preserved
    const finalPositions: Record<string, { x: number; y: number }> = {};

    const stateKey = graphData.nodes.map(n => n.id).sort().join(',');
    if (stateKey !== this.userPositionsKey) this.loadUserPositions(stateKey);

    for (const nodeId of graphData.nodes.map(n => n.id)) {
      if (this.userPositions[nodeId]) {
        // Hand-placed by the user: wins over everything, survives reloads
        finalPositions[nodeId] = this.userPositions[nodeId];
      } else if (this.savedPositions[nodeId]) {
        // Always prioritize saved positions from previous render
        finalPositions[nodeId] = this.savedPositions[nodeId];
      } else if (layoutMode === 'connections' && cachedForcePositions?.[nodeId]) {
        // Use cached force-directed positions
        finalPositions[nodeId] = cachedForcePositions[nodeId];
      } else if (layoutMode === 'deployer' || layoutMode === 'geo-level') {
        // For clustering modes: use computed cluster positions
        if (clusterPositions[nodeId]) {
          finalPositions[nodeId] = clusterPositions[nodeId];
        }
      } else if (layoutMode === 'default' && presetPositions[nodeId]) {
        // Only apply preset positions for the 'default' layout
        finalPositions[nodeId] = presetPositions[nodeId];
      }
    }

    const usePresetPositions = Object.keys(finalPositions).length > 0;

    // Compute absorbing sets if needed (ring coloring only; absorbing-set
    // membership is no longer a fill-coloring option).
    const nodeToAbsorbing = ringColoringMode === 'absorbing' ? computeAbsorbingSets(graphData, filteredEdges) : new Map<string, number | null>();

    // Compute G level range for geoengineering coloring
    let minG = 0, maxG = 1;
    if (coloringMode === 'geoengineering') {
      const geoLevels = graphData.nodes.map(n => n.meta?.geo_level || 0);
      minG = Math.min(...geoLevels);
      maxG = Math.max(...geoLevels);
    }

    // If requested, remove self-loop edges from display
    const displayedEdgesList = showSelfLoops ? filteredEdges : filteredEdges.filter(e => e.source !== e.target);

    // Convert to Cytoscape format
    const elements: cytoscape.ElementDefinition[] = [
      // N5 taxonomy headings are graph elements, so they remain correctly
      // positioned when the user pans or zooms the static layout.
      ...(n5Taxonomy?.labels.map(heading => ({
        group: 'nodes' as const,
        data: { id: heading.id, label: heading.label, is_taxonomy_label: true, label_width: heading.width || 150 },
        position: { x: heading.x, y: heading.y },
      })) || []),
      // Nodes
      ...graphData.nodes.map(node => {
        const geoLevel = node.meta?.geo_level || 0;
        const deployingCoalition = node.meta?.deploying_coalition || '';
        const normalizedLabel = normalizeStateName(node.label);
        const isN5 = graphData.nodes.length === 52;
        // N5 uses compact country codes, leaving enough room to retain the
        // G value directly on the state rather than requiring a click.
        const displayLabel = isN5 ? compactN5StateLabel(normalizedLabel, n5PlayerCodes) : normalizedLabel;
        let label = displayLabel;
        if (showGeoLevel) label += `\nG=${geoLevel.toFixed(2)}`;

        const element: cytoscape.ElementDefinition = {
          group: 'nodes' as const,
          data: {
            id: node.id,
            label: label,
            geo_level: geoLevel,
            deploying_coalition: deployingCoalition,
            payoffs: node.meta?.payoffs,
            values: node.meta?.values,
          }
        };

        if (emojiMoods) (element.data as any).emoji_img = emojiSvgDataUri(emojiMoods[node.id]);

        // Add preset position if available (prefers saved positions)
        if (usePresetPositions && node.id in finalPositions) {
          element.position = finalPositions[node.id];
        }

        // Assign color based on coloring mode
        let color = '#cbd5e1'; // Default gray

        if (coloringMode === 'internal') {
          const stability = node.meta?.stability as { internal?: boolean } | undefined;
          if (stability && typeof stability.internal === 'boolean') {
            if (stability.internal) {
              color = STABILITY_RED_SOFT; // internally stable -> soft red stripes
              (element.data as any).stab_pattern = 'internal-red';
            } else {
              color = STABILITY_NEUTRAL; // not stable -> no colour
            }
          }
        } else if (coloringMode === 'external') {
          const stability = node.meta?.stability as { external_consent?: boolean } | undefined;
          if (stability && typeof stability.external_consent === 'boolean') {
            if (stability.external_consent) {
              color = STABILITY_RED_SOFT; // externally stable -> soft red stripes
              (element.data as any).stab_pattern = 'external-red';
            } else {
              color = STABILITY_NEUTRAL; // not stable -> no colour
            }
          }
        } else if (coloringMode === 'internal-external') {
          const stability = node.meta?.stability as { internal?: boolean; external_consent?: boolean } | undefined;
          if (stability && typeof stability.internal === 'boolean' && typeof stability.external_consent === 'boolean') {
            if (stability.internal && stability.external_consent) {
              color = STABILITY_ORANGE; // both stable -> solid orange (kept distinct from the red absorbing-set ring)
            } else if (stability.internal) {
              color = STABILITY_ORANGE_SOFT; // internal only -> soft orange stripes
              (element.data as any).stab_pattern = 'internal-orange';
            } else if (stability.external_consent) {
              color = STABILITY_ORANGE_SOFT; // external only -> soft orange opposite stripes
              (element.data as any).stab_pattern = 'external-orange';
            } else {
              color = STABILITY_NEUTRAL; // neither -> no colour (default gray)
            }
          }
        } else if (coloringMode === 'gamma-core') {
          const gc = node.meta?.gamma_core as { in_core?: boolean } | undefined;
          if (gc && typeof gc.in_core === 'boolean') {
            color = gc.in_core ? GAMMA_CORE_TEAL : STABILITY_NEUTRAL;
          }
        } else if (coloringMode === 'ricke') {
          const rk = node.meta?.ricke as { majority?: boolean; winning?: boolean } | undefined;
          if (rk && typeof rk.majority === 'boolean' && rk.majority) {
            color = rk.winning ? RICKE_PURPLE : STABILITY_NEUTRAL;
          }
        } else if (coloringMode === 'geoengineering') {
          // Blue gradient based on G level
          const geoLevel = node.meta?.geo_level || 0;
          const range = maxG - minG || 1;
          const normalized = (geoLevel - minG) / range;

          // Interpolate from light blue (#e0f2fe) to dark blue (#0369a1)
          // Light blue for low G, dark blue for high G
          const r = Math.round(224 - normalized * (224 - 3));
          const g = Math.round(242 - normalized * (242 - 105));
          const b = Math.round(254 - normalized * (254 - 161));
          color = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        }

        (element.data as any).color = color;

        // Ring (border) coloring is a separate visual channel from the fill
        // above: a border is painted outside the node's width/height box, so
        // it does not change the node's apparent size.
        if (ringColoringMode === 'absorbing') {
          (element.data as any).in_absorbing_set = nodeToAbsorbing.get(node.id) !== null
            && nodeToAbsorbing.get(node.id) !== undefined;
        }

        const stabPattern = (element.data as any).stab_pattern as string | undefined;
        if (emojiMoods && stabPattern) {
          (element.data as any).stripe_img = emojiOnStripeDataUri(emojiMoods[node.id], stabPattern);
        }

        return element;
      }),
      // Edges
      ...displayedEdgesList.map(edge => ({
        group: 'edges' as const,
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          // keep probability data but label rendering can be toggled via styles
          label: this.formatProbability(edge.p),
          probability: edge.p,
          isSelfLoop: edge.source === edge.target,
          width: this.getEdgeWidth(edge.p, edge.source === edge.target, filteredEdges),
          breakdown: (edge as any).meta?.breakdown
        }
      }))
    ];

    // Build style array dynamically based on toggles
    const styles: any[] = [];

    // Node base style (use text-opacity to reliably hide/show labels)
    styles.push({
      selector: 'node',
      style: {
        'background-color': 'data(color)',
        'label': 'data(label)',
        'text-opacity': showNodeLabels ? 1 : 0,
        'color': '#1e293b',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': graphData.nodes.length === 52 ? '10px' : '12px',
        'font-weight': 'normal',
        'width': graphData.nodes.length === 52 ? 62 : (emojiLayout ? emojiLayout.nodeSize : 70),
        'height': graphData.nodes.length === 52 ? 62 : (emojiLayout ? emojiLayout.nodeSize : 70),
        'text-outline-width': 0,
        'text-wrap': 'wrap',
        'text-max-width': graphData.nodes.length === 52 ? '58px' : (emojiLayout ? `${emojiLayout.nodeSize - 4}px` : '65px'),
        'border-width': 0,
        'border-color': '#1e293b'
      }
    });

    if (emojiLayout) {
      // Emoji strip sits in the upper part of the node; text label moves below it.
      styles.push({
        selector: 'node[emoji_img]',
        style: {
          'background-image': 'data(emoji_img)',
          'background-fit': 'none',
          'background-width': `${emojiLayout.w}px`,
          'background-height': `${emojiLayout.h}px`,
          'background-position-x': '50%',
          'background-position-y': `${emojiLayout.stripTop}px`,
          'background-clip': 'none',
          'text-valign': 'center',
          'text-margin-y': emojiLayout.labelMargin,
        }
      });
    }


    styles.push({
      selector: 'node[?is_taxonomy_label]',
      style: {
        'background-opacity': 0,
        'border-width': 0,
        'width': 'data(label_width)',
        'height': 42,
        'font-size': '13px',
        'font-weight': 600,
        'color': '#475569',
        'text-wrap': 'wrap',
        'text-max-width': 'data(label_width)',
        'text-valign': 'center',
        'text-halign': 'center',
        'events': 'no'
      }
    });

    // Absorbing-set ring: a secondary coloring channel independent of the
    // fill color above. Cytoscape paints the border outside the node's
    // width/height box, so this never changes the circle's apparent size --
    // it reads as a ring around the existing fill, not a bigger node.
    styles.push({
      selector: 'node[?in_absorbing_set]',
      style: {
        'border-width': function(ele: any) {
          return ele.data('in_absorbing_set') ? 6 : 0;
        },
        'border-color': ABSORBING_RING_COLOR,
        'border-opacity': 1
      }
    });

    // Diagonal stripes for internal-only / external-only nodes, in the
    // single-dimension (red) and combined (orange) stability colourings.
    // (background-color is set via data(color) as the stripe tile's fill.)
    styles.push({
      selector: 'node[stab_pattern="internal-red"]',
      style: {
        'background-image': stabilityStripeDataUri(45, STABILITY_RED_SOFT),
        'background-fit': 'cover'
      }
    });
    styles.push({
      selector: 'node[stab_pattern="external-red"]',
      style: {
        'background-image': stabilityStripeDataUri(-45, STABILITY_RED_SOFT),
        'background-fit': 'cover'
      }
    });
    styles.push({
      selector: 'node[stab_pattern="internal-orange"]',
      style: {
        'background-image': stabilityStripeDataUri(45, STABILITY_ORANGE_SOFT),
        'background-fit': 'cover'
      }
    });
    styles.push({
      selector: 'node[stab_pattern="external-orange"]',
      style: {
        'background-image': stabilityStripeDataUri(-45, STABILITY_ORANGE_SOFT),
        'background-fit': 'cover'
      }
    });

    // Emoji + stripes: the combined node-sized image replaces both single images.
    if (emojiLayout) {
      styles.push({
        selector: 'node[stab_pattern][stripe_img]',
        style: {
          'background-image': 'data(stripe_img)',
          'background-fit': 'none',
          'background-width': `${emojiLayout.nodeSize}px`,
          'background-height': `${emojiLayout.nodeSize}px`,
          'background-position-x': '50%',
          'background-position-y': '50%',
          'background-clip': 'node',
        }
      });
    }

    // Selected node
    styles.push({
      selector: 'node:selected',
      style: {
        'overlay-color': '#000000',
        'overlay-opacity': 0.2,
        'overlay-padding': 6,
        'border-width': 3,
        'border-color': '#475569'
      }
    });

    // Edge base style (use text-opacity to hide labels without affecting layout)
    styles.push({
      selector: 'edge',
      style: {
        'width': 'data(width)',
        'line-color': '#334155',
        'target-arrow-color': '#334155',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        'label': 'data(label)',
        'text-opacity': showEdgeLabels ? 1 : 0,
        'font-size': '11px',
        'text-rotation': 'autorotate',
        'text-margin-y': -10,
        'color': '#1e293b',
        'text-background-color': 'transparent',
        'text-background-opacity': 0
      }
    });

    // Self-loop style
    styles.push({
      selector: 'edge[isSelfLoop]',
      style: {
        'curve-style': 'loop',
        'loop-direction': '0deg',
        'loop-sweep': '90deg',
        'control-point-step-size': 40
      }
    });

    // Highlight / dim styles
    styles.push({
      selector: '.highlighted',
      style: {
        'line-color': '#0f172a',
        'target-arrow-color': '#0f172a',
        'opacity': 1
      }
    });

    styles.push({
      selector: '.highlighted-node',
      style: {
        'opacity': 1
      }
    });

    styles.push({
      selector: '.dimmed',
      style: {
        'opacity': 0.2
      }
    });

    // Create Cytoscape instance.
    // For force-directed layout, seed Math.random so the result is identical across machines.
    const layoutConfig = this.getLayoutConfig(layoutMode, graphData, usePresetPositions);
    const needsSeededRandom = layoutMode === 'connections' && layoutConfig.name === 'cose-bilkent';
    const createCy = () => cytoscape({
      container: this.container,
      elements,
      style: styles,
      layout: layoutConfig,
      minZoom: 0.2,
      maxZoom: 3,
      wheelSensitivity: 1.5
    });
    this.cy = needsSeededRandom ? withSeededRandom(FORCE_LAYOUT_SEED, createCy) : createCy();

    // Cache force-directed positions after layout completes (memory + localStorage)
    if (layoutMode === 'connections' && !cachedForcePositions) {
      const positions: Record<string, { x: number; y: number }> = {};
      this.cy.nodes().forEach(node => {
        const pos = node.position();
        positions[node.id()] = { x: pos.x, y: pos.y };
      });
      forceDirectedCache.set(cacheKey, positions);
      try {
        localStorage.setItem(`force-layout:${cacheKey}`, JSON.stringify(positions));
      } catch { /* ignore localStorage errors */ }
    }

    this.setupInteractions();

    // Restore zoom and pan if previously saved
    if (this.savedZoom !== null && this.savedPan !== null) {
      this.cy.zoom(this.savedZoom);
      this.cy.pan(this.savedPan);
    } else {
      // For larger graphs (n=4 with 15 nodes), fit to view with padding on initial render
      if (graphData.nodes.length >= 15) {
        this.cy.fit(undefined, 50); // Fit all elements with 50px padding
      }
    }

    // Notify external tooling (e.g. the headless screenshot CLI in
    // viz/render_graph.py) that the graph finished rendering.
    (window as any).__graphRenderCount = ((window as any).__graphRenderCount || 0) + 1;
    window.dispatchEvent(new CustomEvent('graph-rendered'));
  }

  private formatProbability(p: number): string {
    // Convert to percentage and format with comma as decimal separator
    const percentage = (p * 100).toFixed(2);
    return percentage.replace('.', ',') + '%';
  }

  private getEdgeWidth(probability: number, isSelfLoop: boolean, allEdges: typeof import('./types').GraphData.edges): number {
    // Use separate scales for self-loops vs state transitions
    // This makes differences visible in both categories

    if (isSelfLoop) {
      // Self-loops typically have high probabilities (0.7-1.0)
      // Find min/max among self-loops for adaptive scaling
      const selfLoopProbs = allEdges.filter(e => e.source === e.target).map(e => e.p);
      const minSelfLoop = Math.min(...selfLoopProbs);
      const maxSelfLoop = Math.max(...selfLoopProbs);
      const range = maxSelfLoop - minSelfLoop || 0.1; // Avoid division by zero

      // Map to 2-6px range
      const normalized = (probability - minSelfLoop) / range;
      return Math.max(2, Math.min(6, 2 + 4 * normalized));
    } else {
      // Transitions typically have lower probabilities (0-0.3)
      // Find min/max among transitions for adaptive scaling
      const transitionProbs = allEdges.filter(e => e.source !== e.target).map(e => e.p);
      if (transitionProbs.length === 0) return 2;

      const minTransition = Math.min(...transitionProbs);
      const maxTransition = Math.max(...transitionProbs);
      const range = maxTransition - minTransition || 0.1;

      // Map to 1-5px range
      const normalized = (probability - minTransition) / range;
      return Math.max(1, Math.min(5, 1 + 4 * normalized));
    }
  }

  private getLayoutConfig(
    layoutMode: 'default' | 'connections' | 'deployer' | 'geo-level',
    graphData: GraphData,
    usePresetPositions: boolean
  ) {
    switch (layoutMode) {
      case 'default':
        // Use preset positions for compact graphs and the explicitly labelled
        // five-player taxonomy. "By connections" remains the force layout.
        // otherwise force-directed
        if (usePresetPositions || graphData.nodes.length === 3 || graphData.nodes.length === 4 || graphData.nodes.length === 5 || graphData.nodes.length === 15 || graphData.nodes.length === 52) {
          return { name: 'preset' };
        } else {
          return {
            name: 'cose-bilkent',
            animate: false,
            randomize: true,
            nodeRepulsion: 4500,
            idealEdgeLength: 100,
            edgeElasticity: 0.45,
            nestingFactor: 0.1,
            gravity: 0.25,
            numIter: 2500,
            tile: true,
            tilingPaddingVertical: 10,
            tilingPaddingHorizontal: 10
          };
        }

      case 'connections':
        // Use cached positions if available (deterministic across switches)
        if (usePresetPositions) {
          return { name: 'preset' };
        }
        // First time: run force-directed with seeded PRNG for deterministic result.
        // Positions are cached to localStorage after layout completes.
        return {
          name: 'cose-bilkent',
          animate: false,
          randomize: true,
          nodeRepulsion: 4500,
          idealEdgeLength: 100,
          edgeElasticity: 0.45,
          nestingFactor: 0.1,
          gravity: 0.25,
          numIter: 2500,
          tile: true,
          tilingPaddingVertical: 10,
          tilingPaddingHorizontal: 10
        };

      case 'deployer': {
        // Cluster by deploying coalition
        const nodesByDeployer = new Map<string, string[]>();
        graphData.nodes.forEach(node => {
          const deployer = node.meta?.deploying_coalition || 'None';
          if (!nodesByDeployer.has(deployer)) {
            nodesByDeployer.set(deployer, []);
          }
          nodesByDeployer.get(deployer)!.push(node.id);
        });

        const groups = Array.from(nodesByDeployer.values()).map(ids => ({ leaves: ids }));

        return {
          name: 'preset',  // Use preset with our computed cluster positions
          fit: true,
          padding: 50
        };
      }

      case 'geo-level': {
        // Cluster by geoengineering level
        return {
          name: 'preset',  // Use preset with our computed cluster positions
          fit: true,
          padding: 50
        };
      }

      default:
        return { name: 'cose-bilkent', animate: false };
    }
  }

  private setupInteractions() {
    if (!this.cy) return;

    // Click node to select
    this.cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      if (this.selectedNode === node.id()) {
        this.deselectNode();
      } else {
        this.selectNode(node.id());
      }
    });

    // Remember hand-dragged nodes (persisted so they survive reloads)
    this.cy.on('dragfree', 'node', (evt) => {
      const node = evt.target;
      if (node.data('is_taxonomy_label')) return;
      const pos = node.position();
      this.userPositions[node.id()] = { x: pos.x, y: pos.y };
      this.persistUserPositions();
    });

    // Click background to deselect
    this.cy.on('tap', (evt) => {
      if (evt.target === this.cy) {
        this.deselectNode();
      }
    });

    // Hover effects
    this.cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;

      // Highlight connected edges
      const connectedEdges = node.connectedEdges();
      connectedEdges.addClass('highlighted');

      // Dim everything else
      this.cy?.elements().not(node).not(connectedEdges).addClass('dimmed');
    });

    this.cy.on('mouseout', 'node', () => {
      this.cy?.elements().removeClass('highlighted dimmed');
    });
  }

  // Highlight a set of nodes and their connected edges; dim everything else
  highlightNodes(nodeIds: string[]) {
    if (!this.cy) return;
    // Clear previous
    this.cy.elements().removeClass('highlighted dimmed highlighted-node');

    const nodes = this.cy.nodes().filter((n) => nodeIds.includes(n.id()));
    if (nodes.length === 0) return;

    nodes.addClass('highlighted-node');

    const connectedEdges = nodes.connectedEdges();
    connectedEdges.addClass('highlighted');

    // Dim everything else
    this.cy.elements().not(nodes).not(connectedEdges).addClass('dimmed');
  }

  clearHighlights() {
    if (!this.cy) return;
    this.cy.elements().removeClass('highlighted dimmed highlighted-node');
  }

  private selectNode(nodeId: string) {
    if (!this.cy) return;

    this.selectedNode = nodeId;

    // Deselect all and select the clicked node
    this.cy.nodes().unselect();
    const node = this.cy.$id(nodeId);
    node.select();

    if (this.onNodeSelect) {
      this.onNodeSelect(nodeId);
    }
  }

  private deselectNode() {
    if (!this.cy) return;

    this.selectedNode = null;
    this.cy.nodes().unselect();

    if (this.onNodeSelect) {
      this.onNodeSelect(null);
    }
  }

  resetView() {
    if (!this.cy) return;
    this.cy.fit(undefined, 50);
  }

  zoomIn() {
    if (!this.cy) return;
    const currentZoom = this.cy.zoom();
    this.cy.zoom({
      level: currentZoom * 1.2,
      renderedPosition: {
        x: this.cy.width() / 2,
        y: this.cy.height() / 2
      }
    });
  }

  zoomOut() {
    if (!this.cy) return;
    const currentZoom = this.cy.zoom();
    this.cy.zoom({
      level: currentZoom / 1.2,
      renderedPosition: {
        x: this.cy.width() / 2,
        y: this.cy.height() / 2
      }
    });
  }

  getNodeData(nodeId: string): any {
    if (!this.cy) return undefined;
    const node = this.cy.$id(nodeId);
    if (node.length === 0) return undefined;
    return {
      label: node.data('label'),
      data: {
        geo_level: node.data('geo_level'),
        deploying_coalition: node.data('deploying_coalition'),
        payoffs: node.data('payoffs'),
        values: node.data('values'),
      },
      meta: {
        geo_level: node.data('geo_level'),
        deploying_coalition: node.data('deploying_coalition'),
        payoffs: node.data('payoffs'),
        values: node.data('values'),
      }
    };
  }

  getOutgoingEdges(nodeId: string): Array<{ target: string; probability: number; breakdown?: any }> {
    if (!this.cy) return [];

    const node = this.cy.$id(nodeId);
    if (node.length === 0) return [];

    const edges: Array<{ target: string; probability: number; breakdown?: any }> = [];
    node.outgoers('edge').forEach(edge => {
      edges.push({
        target: edge.target().id(),
        probability: edge.data('probability'),
        breakdown: edge.data('breakdown')
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  getIncomingEdges(nodeId: string): Array<{ source: string; probability: number; breakdown?: any }> {
    if (!this.cy) return [];

    const node = this.cy.$id(nodeId);
    if (node.length === 0) return [];

    const edges: Array<{ source: string; probability: number; breakdown?: any }> = [];
    node.incomers('edge').forEach(edge => {
      edges.push({
        source: edge.source().id(),
        probability: edge.data('probability'),
        breakdown: edge.data('breakdown')
      });
    });

    return edges.sort((a, b) => b.probability - a.probability);
  }

  destroy() {
    if (this.cy) {
      this.cy.destroy();
      this.cy = null;
    }
  }
}
