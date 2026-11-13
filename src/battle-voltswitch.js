import { resolvePivotEligibility } from './battle-pivot.js';

// Resolve the Volt Switch-specific switch request after the hit pipeline has
// finished. The battle owner suspends the turn and owns switch validation.
export function resolveVoltSwitchPivot({
  moveId,
  result,
  actorHp,
  hasBench,
  canSwitch = true,
  commanded = false,
}) {
  return resolvePivotEligibility(moveId, 'voltswitch', result, actorHp, hasBench, canSwitch, commanded);
}
