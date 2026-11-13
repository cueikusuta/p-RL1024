import { resolvePivotEligibility } from './battle-pivot.js';

// The owning Battle resolves the hit first, then uses this result to decide
// whether to request a private mid-turn replacement choice.
export function resolveFlipTurnPivot({
  moveId,
  result,
  actorHp,
  hasBench,
  canSwitch = true,
  commanded = false,
}) {
  return resolvePivotEligibility(moveId, 'flipturn', result, actorHp, hasBench, canSwitch, commanded);
}
