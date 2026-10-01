import type { GraphData, GraphNode } from './types';

export type SetRelation =
  | 'EQUAL'
  | 'SUBSET'
  | 'SUPERSET'
  | 'INTERSECTION'
  | 'DISJOINT'
  | 'UNDEFINED';

export interface ConceptRelationRow {
  concept: string;
  relation: SetRelation;
  jaccard: number | null;
  hamming: number | null;
  conceptSetSize: number;
  mpeSetSize: number;
}

/** One named static solution concept, as a predicate over a node's meta. */
interface ConceptDef {
  name: string;
  /** Returns true/false if defined for this node, or null if undefined for this game. */
  predicate: (node: GraphNode) => boolean | null;
}

const CONCEPT_DEFS: ConceptDef[] = [
  {
    name: 'Int. stab.',
    predicate: (n) => {
      const v = n.meta?.stability?.internal;
      return typeof v === 'boolean' ? v : null;
    },
  },
  {
    // Consent-based reading only (incumbents must also agree to accession),
    // matching the approval-committee mechanism this framework's dynamics use
    // -- same field the sidebar's "By external stability" coloring reads.
    // The open-membership variant (external_open) is a different benchmark
    // not otherwise surfaced in this UI, so it's omitted here too.
    name: 'Ext. stab.',
    predicate: (n) => {
      const v = n.meta?.stability?.external_consent;
      return typeof v === 'boolean' ? v : null;
    },
  },
  {
    name: 'Int. + ext.',
    predicate: (n) => {
      const v = n.meta?.stability?.stable_consent;
      return typeof v === 'boolean' ? v : null;
    },
  },
  {
    name: 'Gamma core',
    predicate: (n) => {
      const v = n.meta?.gamma_core?.in_core;
      return typeof v === 'boolean' ? v : null;
    },
  },
  {
    name: 'Ricke MWC',
    predicate: (n) => {
      const v = n.meta?.ricke?.winning;
      return typeof v === 'boolean' ? v : null;
    },
  },
];

/**
 * Compares each static solution concept's predicted set of states against the
 * MPE's absorbing set (farsighted equilibrium outcome), as plain set relations.
 *
 * The MPE reference set is every node with a non-null entry in `absorbingSetIndex`
 * (as produced by `computeAbsorbingSets`, over the currently-displayed/filtered
 * edges, so this stays consistent with the "by absorbing set" node coloring). A
 * concept is "undefined" for this game if none of its nodes carry the relevant
 * meta field (e.g. Ricke under weak_governance, which has no majority-power
 * concept).
 */
export function computeConceptRelations(
  data: GraphData,
  absorbingSetIndex: Map<string, number | null>
): ConceptRelationRow[] {
  const mpeSet = new Set<string>(
    [...absorbingSetIndex.entries()].filter(([, idx]) => idx !== null).map(([id]) => id)
  );

  const rows: ConceptRelationRow[] = [];

  for (const def of CONCEPT_DEFS) {
    let anyDefined = false;
    const conceptSet = new Set<string>();

    for (const node of data.nodes) {
      const verdict = def.predicate(node);
      if (verdict === null) continue;
      anyDefined = true;
      if (verdict) conceptSet.add(node.id);
    }

    if (!anyDefined || mpeSet.size === 0) {
      rows.push({
        concept: def.name,
        relation: 'UNDEFINED',
        jaccard: null,
        hamming: null,
        conceptSetSize: conceptSet.size,
        mpeSetSize: mpeSet.size,
      });
      continue;
    }

    rows.push({
      concept: def.name,
      // MPE is the subject: e.g. SUBSET means "MPE's absorbing set is a
      // subset of this concept's predicted set," matching how the relation
      // column reads ("what is MPE, relative to this row's concept?").
      ...relateSets(mpeSet, conceptSet),
      conceptSetSize: conceptSet.size,
      mpeSetSize: mpeSet.size,
    });
  }

  return rows;
}

function relateSets(
  a: Set<string>,
  b: Set<string>
): { relation: SetRelation; jaccard: number; hamming: number } {
  const union = new Set<string>([...a, ...b]);
  let intersectionSize = 0;
  for (const x of a) if (b.has(x)) intersectionSize++;

  const symmetricDiff = a.size + b.size - 2 * intersectionSize;
  const jaccard = union.size === 0 ? 1 : intersectionSize / union.size;

  let relation: SetRelation;
  if (a.size === 0 && b.size === 0) {
    relation = 'EQUAL';
  } else if (intersectionSize === 0) {
    relation = 'DISJOINT';
  } else if (a.size === b.size && intersectionSize === a.size) {
    relation = 'EQUAL';
  } else if (intersectionSize === a.size) {
    relation = 'SUBSET'; // a ⊆ b
  } else if (intersectionSize === b.size) {
    relation = 'SUPERSET'; // a ⊇ b
  } else {
    relation = 'INTERSECTION';
  }

  return { relation, jaccard, hamming: symmetricDiff };
}
