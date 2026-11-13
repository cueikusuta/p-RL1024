import { productionCombatant } from './battle-production.js';

// A catalog is an atomic release. Never silently shrink the configured pool.
export function requirePlayableCatalog(sets, data) {
  const rejected = [];
  for (const set of sets) {
    try { productionCombatant({...set, instanceId: 'support-check'}, data); }
    catch (error) { rejected.push(`${set.setId}: ${error.message}`); }
  }
  if (rejected.length) throw new Error(`Catalog has ${rejected.length} unsupported set(s):\n${rejected.join('\n')}`);
  return sets;
}
