import { requireRule } from './draft.js';
const STATS = ['hp','atk','def','spa','spd','spe'];
// Reference: Champions scripts.ts/statModify at b8f23c99629132611ba59e99480cb148ea8c5d3d.
export function calculateChampionsStats(base, points = {}, nature = {}) {
  requireRule(nature && typeof nature === 'object', 'Invalid nature facts');
  for (const key of ['plus','minus']) requireRule(nature[key] === undefined || STATS.slice(1).includes(nature[key]), 'Invalid nature stat');
  requireRule((nature.plus === undefined) === (nature.minus === undefined), 'Incomplete nature facts');
  requireRule(!nature.plus || nature.plus !== nature.minus, 'Invalid nature pair');
  requireRule(Object.keys(points).every(stat => STATS.includes(stat)), 'Unknown stat points');
  let total = 0;
  const stats = Object.fromEntries(STATS.map(stat => {
    requireRule(Number.isInteger(base?.[stat]) && base[stat] >= 1 && base[stat] <= 255, 'Invalid base stats');
    const point = points[stat] ?? 0;
    requireRule(Number.isInteger(point) && point >= 0 && point <= 32, 'Invalid stat points');
    total += point;
    const value = base[stat] + point + (stat === 'hp' ? 75 : 20);
    const modifier = nature.plus === stat ? 110 : nature.minus === stat ? 90 : 100;
    return [stat, Math.floor(((value * modifier) % 65536) / 100)];
  }));
  requireRule(total <= 66, 'Too many stat points');
  return stats;
}
