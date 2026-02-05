import type { GraphData, ProfilesResponse } from './types';

// Static data mode - fetch precomputed JSON files from /data directory
const DATA_BASE = '/data';

export async function fetchProfiles(): Promise<ProfilesResponse> {
  const response = await fetch(`${DATA_BASE}/profiles.json`);
  if (!response.ok) {
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

  const response = await fetch(`${DATA_BASE}/${basename}.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch graph data: ${response.statusText}`);
  }
  return response.json();
}
