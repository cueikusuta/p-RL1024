// Resolve the U-turn-specific switch request after the hit pipeline has
// finished. The battle owner remains responsible for suspending the turn,
// validating the private replacement choice, switching, and resuming actions.
const SUCCESSFUL_HIT_RESULTS = new Set(['hit', 'substitute']);

export function resolvePivotEligibility(moveId, expectedMoveId, result, actorHp, hasBench, canSwitch, commanded) {
  let reason = 'eligible';
  if (moveId !== expectedMoveId) reason = `not-${expectedMoveId}`;
  else if (!SUCCESSFUL_HIT_RESULTS.has(result)) reason = 'move-did-not-hit';
  else if (!(actorHp > 0)) reason = 'actor-fainted';
  else if (commanded) reason = 'actor-commanded';
  else if (!hasBench) reason = 'no-usable-bench';
  else if (!canSwitch) reason = 'switch-blocked';

  if (reason !== 'eligible') return { switch: false, reason };
  return { switch: true, reason, request: { kind: 'pivot', moveId } };
}

/**
 * Decide whether a completed U-turn creates a mid-turn replacement request.
 * `result` is the hit pipeline outcome, rather than damage dealt: a hit into
 * Substitute is still a successful hit. Target fainting is handled outside
 * this helper because it may end the battle before a request is issued.
 */
export function resolveUTurnPivot({
  moveId,
  result,
  actorHp,
  hasBench,
  canSwitch = true,
  commanded = false,
}) {
  return resolvePivotEligibility(moveId, 'uturn', result, actorHp, hasBench, canSwitch, commanded);
}
