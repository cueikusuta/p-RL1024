import {
  formatPublicMoveMessages,
  formatPublicDelayedMoveArrival,
  formatPublicBoostMessages,
  formatPublicHeldItemMessages,
  formatPublicMoveStatusMessages,
  formatPublicStatusAbilityMessages,
  formatPublicMoveHealMessages,
  formatPublicMoveHpLoss,
  formatPublicAbilityStatusCureMessages,
  formatPublicAbilityHealMessages,
  formatPublicFieldChangeMessages,
  formatPublicHarvestRestorationMessages,
  formatPublicEntryAbilityMessages,
  formatPublicDamagingHitAbility,
  formatPublicAbilityLifecycle,
  formatPublicEntryItemMessages,
  formatPublicConditionAbilityMessages,
  formatPublicNaturalCureMessages,
  formatPublicAbilityMoveBlockMessages,
  formatPublicNeutralStatusMessages,
  formatPublicNeutralMoveOutcomeMessages,
  formatPublicConfirmedConditionHeal,
  formatPublicConfirmedConditionTransition,
  formatPublicUnableToAct,
} from './battle-messages.js';
import { formatPublicMessages } from './battle-damage-messages.js';
import { applyPublicKnowledge } from './public-knowledge.js';

// This is a server-only publication boundary, not an event bus or a new
// persistence layer. Battle owns the event timing, state, and presentation
// frames; the public event owns disclosure, not its Japanese log formatter.
const messageFormatters=Object.freeze({
  'move-used':formatPublicMoveMessages,
  'delayed-move-arrival':formatPublicDelayedMoveArrival,
  'boost-changed':formatPublicBoostMessages,
  'held-item':formatPublicHeldItemMessages,
  'status-move':formatPublicMoveStatusMessages,
  status:formatPublicStatusAbilityMessages,
  'heal-move':formatPublicMoveHealMessages,
  'hp-loss-move':formatPublicMoveHpLoss,
  'status-cured-ability':formatPublicAbilityStatusCureMessages,
  heal:formatPublicAbilityHealMessages,
  'field-change':formatPublicFieldChangeMessages,
  'harvest-restored':formatPublicHarvestRestorationMessages,
  'entry-ability':formatPublicEntryAbilityMessages,
  'ability-activated':formatPublicDamagingHitAbility,
  'ability-lifecycle':formatPublicAbilityLifecycle,
  'entry-item-revealed':formatPublicEntryItemMessages,
  'condition-ability':formatPublicConditionAbilityMessages,
  'switchout-ability-cure':formatPublicNaturalCureMessages,
  'move-blocked-ability':formatPublicAbilityMoveBlockMessages,
  'move-absorbed-ability':formatPublicAbilityMoveBlockMessages,
  'status-transition':formatPublicNeutralStatusMessages,
  'move-result':formatPublicNeutralMoveOutcomeMessages,
  'heal-condition':formatPublicConfirmedConditionHeal,
  'condition-transition':formatPublicConfirmedConditionTransition,
  'action-result':formatPublicUnableToAct,
});

export function publishResolvedPublicEvent(event,{observations,appendMessage}) {
  // First subscribe the confirmed disclosures. A silent formatter, a changed
  // message string, or a formatter exception cannot revoke that disclosure.
  applyPublicKnowledge(observations,event);
  const formatter=messageFormatters[event.kind]??formatPublicMessages;
  for(const message of formatter(event))appendMessage(message);
}
