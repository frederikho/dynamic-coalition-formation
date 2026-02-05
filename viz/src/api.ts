import type { GraphData, ProfilesResponse } from './types';

// Static data mode - fetch precomputed JSON files from /data directory
// Use Vite's BASE_URL to handle GitHub Pages subdirectory deployment
const DATA_BASE = `${import.meta.env.BASE_URL}data`;

export async function fetchProfiles(): Promise<ProfilesResponse> {
  const url = `${DATA_BASE}/profiles.json`;
  console.log('[API] Fetching profiles from:', url);

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
  // Extract filename from path (e.g., "strategy_tables/foo.xlsx" -> "foo")
  const pathParts = params.profile.split('/');
  const filename = pathParts[pathParts.length - 1];
  const basename = filename.replace('.xlsx', '');

  const url = `${DATA_BASE}/${basename}.json`;
  console.log('[API] Fetching graph from:', url);

  const response = await fetch(url);
  if (!response.ok) {
    console.error('[API] Failed to fetch graph:', response.status, response.statusText);
    throw new Error(`Failed to fetch graph data: ${response.statusText}`);
  }
  return response.json();
}
