import type { GraphData, ProfilesResponse } from './types';

// Static data mode - fetch precomputed JSON files from /data directory
// Use Vite's BASE_URL to handle GitHub Pages subdirectory deployment
const DATA_BASE = `${import.meta.env.BASE_URL}data`;

// Choose live API when running on localhost; otherwise use static data
// (Hosted on GitHub Pages or similar should serve precomputed /data files.)
const API_BASE = (import.meta.env as any).VITE_API_BASE || 'http://127.0.0.1:8000';
let USE_API = false;
try {
  if (typeof window !== 'undefined' && window.location && window.location.hostname) {
    const host = window.location.hostname;
    // Treat local development hosts as live API mode
    if (host === 'localhost' || host === '127.0.0.1' || host.startsWith('192.168.') || host.startsWith('10.') || host.endsWith('.local')) {
      USE_API = true;
    }
  }
} catch (e) {
  // Default to static when host can't be determined
  USE_API = false;
}

export async function fetchProfiles(): Promise<ProfilesResponse> {
  // If configured, try the live backend first and fall back to static data
  if (USE_API) {
    const apiUrl = `${API_BASE.replace(/\/$/, '')}/profiles`;
    console.log('[API] Fetching profiles from live API:', apiUrl);
    try {
      const resp = await fetch(apiUrl);
      if (resp.ok) return resp.json();
      console.warn('[API] Live API responded with non-OK status:', resp.status, resp.statusText);
    } catch (e) {
      console.warn('[API] Failed to reach live API, falling back to static data:', e);
    }
  }

  // Static fallback
  const url = `${DATA_BASE}/profiles.json`;
  console.log('[API] Fetching profiles from static data:', url);

  const response = await fetch(url);
  if (!response.ok) {
    console.error('[API] Failed to fetch profiles:', response.status, response.statusText);
    throw new Error(`Failed to fetch profiles: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchGraph(params: {
  profile: string;
}): Promise<GraphData> {
  // When using live API, call /graph?profile=...; otherwise fetch precomputed JSON
  if (USE_API) {
    const apiUrl = `${API_BASE.replace(/\/$/, '')}/graph?profile=${encodeURIComponent(params.profile)}`;
    console.log('[API] Fetching graph from live API:', apiUrl);
    try {
      const resp = await fetch(apiUrl);
      if (resp.ok) return resp.json();
      console.warn('[API] Live API graph request failed:', resp.status, resp.statusText);
    } catch (e) {
      console.warn('[API] Failed to reach live API for graph, falling back to static data:', e);
    }
  }

  // Static mode: extract basename and fetch precomputed JSON
  const pathParts = params.profile.split('/');
  const filename = pathParts[pathParts.length - 1];
  const basename = filename.replace('.xlsx', '');

  const url = `${DATA_BASE}/${basename}.json`;
  console.log('[API] Fetching graph from static data:', url);

  const response = await fetch(url);
  if (!response.ok) {
    console.error('[API] Failed to fetch graph:', response.status, response.statusText);
    throw new Error(`Failed to fetch graph data: ${response.statusText}`);
  }
  return response.json();
}
