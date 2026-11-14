import { budgetFor, requireRule, validateOffer, validateTeam, teamCost } from './draft.js';
import {resolveCpuHandicap} from './cpu-handicap.js';
import { defaultTrainerAvatar, isTrainerAvatar, trainerAvatarId } from './trainer-avatars.js';
import { mergeCarryoverObservations, projectCarryoverPublicView, packCarryoverKnowledge, unpackCarryoverKnowledge } from './carryover-observations.js';

// Server-owned domain object. Never serialize this object to clients.
// offerProvider is a trusted server dependency, not a player command.
export class Match {
  #provider;
  #catalogVersion;
  #state;
  #battleFactory;
  #battle = null;
  // Server-only internal history. Checkpoint and public view retain legacy fields.
  #carryoverKnowledge = [{},{}];
  #nextReady = [false,false];
  #ruleset;
  #cpuDifficulty;
  #cpuHandicap;
  #offerAudits = [];
  constructor(offerProvider, { battleFactory = null, ruleset='fixture-v1', catalogVersion=null, cpuDifficulty=0 } = {}) {
    requireRule(typeof offerProvider === 'function', 'Offer provider required');
    this.#provider = offerProvider;
    this.#catalogVersion = catalogVersion;
    this.#battleFactory = battleFactory;
    this.#ruleset=ruleset;
    this.#cpuHandicap=resolveCpuHandicap(cpuDifficulty);
    this.#cpuDifficulty=cpuDifficulty;
    this.#state = { phase: 'waiting', round: 1, score: [0, 0], history: [], players: [0, 1].map(seat => ({ trainerAvatar: defaultTrainerAvatar(seat), ready: false, credits: budgetFor(1)+(seat===1?this.#cpuHandicap.initialCreditBonus:0), teamCost: 0, fresh: [], carry: [], team: null, selected: null })) };
  }
  #player(seat) {
    requireRule(seat === 0 || seat === 1, 'Invalid player');
    return this.#state.players[seat];
  }
  #offers(round, carries, creditBalances=null) {
    const offers=carries.map((carry, seat) => {
      const request={round,seat,carry:structuredClone(carry)};
      if(seat===1&&this.#cpuDifficulty>0)
        request.credits=creditBalances?.[seat]??this.#state.players[seat].credits;
      const fresh = structuredClone(this.#provider(request));
      validateOffer(fresh, carry, round, {cpuHandicap:seat===1&&this.#cpuDifficulty>0?this.#cpuHandicap:null});
      return fresh;
    });
    this.#offerAudits=carries.map((_,seat)=>this.#provider.auditFor?.(seat) ?? null);
    return offers;
  }
  setTrainerAvatar(seat,avatar,{developmentBattleLobby=false}={}) {
    const player=this.#player(seat);
    // A dev-only room has already restored its battle before the guest joins.
    // Only the explicitly gated Node server may opt into this lobby exception.
    requireRule(this.#state.phase==='waiting'
      || (developmentBattleLobby === true && this.#state.phase==='battle'),
      'Trainer image can only change in the lobby');
    requireRule(isTrainerAvatar(avatar),'Invalid trainer image');
    player.trainerAvatar=avatar;
  }
  ready(seat) {
    const p = this.#player(seat);
    requireRule(this.#state.phase === 'waiting', 'Not waiting');
    requireRule(!p.ready, 'Already ready');
    if (this.#state.players[1 - seat].ready) {
      const offers = this.#offers(1, [[], []]);
      this.#state.players.forEach((player, i) => { player.fresh = offers[i]; });
      this.#state.phase = 'draft';
    }
    p.ready = true;
  }
  confirmTeam(seat, ids) {
    const p = this.#player(seat);
    requireRule(this.#state.phase === 'draft' && p.team === null, 'Team cannot be confirmed');
    const team = validateTeam(ids, p.fresh, p.carry, this.#state.round, p.credits,
      {cpuHandicap:seat===1?this.#cpuHandicap:null});
    p.teamCost = teamCost(team, p.carry);
    p.credits -= p.teamCost;
    p.team = team;
    if (this.#state.players.every(player => player.team)) this.#state.phase = 'selection';
  }
  confirmSelection(seat, ids) {
    const p = this.#player(seat);
    requireRule(this.#state.phase === 'selection' && p.selected === null, 'Selection cannot be confirmed');
    requireRule(Array.isArray(ids) && ids.length === 3 && new Set(ids).size === 3, 'Expected three distinct members');
    requireRule(ids.every(id => p.team.some(member => member.instanceId === id)), 'Selection outside team');
    if (this.#state.players[1-seat].selected && this.#battleFactory) {
      const teams = this.#state.players.map((player, i) => (i === seat ? ids : player.selected)
        .map(id => player.team.find(member => member.instanceId === id)));
      this.#battle = this.#battleFactory(structuredClone(teams), { round: this.#state.round });
    }
    p.selected = [...ids];
    if (this.#state.players.every(player => player.selected)) this.#state.phase = 'battle';
  }
  // Internal engine integration only. This is never exposed as a player action.
  // Living selected members carry over, except for the surrendering player.
  finishBattle({ winner, survivors, reason = 'knockout' }) {
    requireRule(this.#state.phase === 'battle', 'Battle not active');
    requireRule(winner === 0 || winner === 1 || winner === null, 'Invalid winner');
    requireRule(Array.isArray(survivors) && survivors.length === 2, 'Missing survivors');
    const carries = survivors.map((rows, seat) => {
      const p = this.#state.players[seat];
      requireRule(Array.isArray(rows) && rows.every(row => row && typeof row === 'object'), 'Invalid survivors');
      requireRule(Array.isArray(rows) && new Set(rows.map(row => row.instanceId)).size === rows.length, 'Invalid survivors');
      return rows.map(row => {
        requireRule(p.selected.includes(row.instanceId) && row.alive === true, 'Invalid survivor');
        // Copy only the original immutable set; HP, PP, status and consumed items are never carried.
        return structuredClone(p.team.find(member => member.instanceId === row.instanceId));
      });
    });
    const previousKnowledge=unpackCarryoverKnowledge(this.#carryoverKnowledge);
    const carryoverObservations = carries.map((rows,seat)=>
      mergeCarryoverObservations(
        rows,this.#state.players[seat].team,this.#battle?.view(1-seat)?.revealed??[],
        previousKnowledge.usedMoves[seat],previousKnowledge.observedItemIds[seat],
        previousKnowledge.observedAbilityIds[seat]));
    if (reason === 'surrender') {
      requireRule(winner !== null, 'Surrender requires a winner');
      carries[1-winner] = [];
    }
    if (winner !== null) this.#state.score[winner]++;
    this.#state.history.push({ round: this.#state.round, winner, reason });
    this.#state.players.forEach((p, seat) => { p.carry = carries[seat]; });
    this.#carryoverKnowledge=packCarryoverKnowledge(
      carryoverObservations.map(row=>row.byInstance),
      carryoverObservations.map(row=>row.itemsByInstance),
      carryoverObservations.map(row=>row.abilitiesByInstance));
    this.#state.phase = winner !== null && this.#state.score[winner] === 4 ? 'complete' : 'result';
    this.#nextReady = [false,false];
  }
  battleAction(seat, action, round, request) {
    this.#player(seat);
    requireRule(this.#state.phase === 'battle' && this.#battle && round === this.#state.round, 'Battle not active or stale round');
    if (action?.kind === 'surrender') this.#battle.surrender(seat, request);
    else this.#battle.submit(seat, action, request);
    const result = this.#battle.result();
    if (result) this.finishBattle(result);
  }
  readyNext(seat, round) {
    this.#player(seat);
    requireRule(this.#state.phase === 'result' && round === this.#state.round && !this.#nextReady[seat], 'Next round unavailable');
    if (this.#nextReady[1-seat]) this.nextRound();
    else this.#nextReady[seat] = true;
  }
  leave(seat) {
    this.#player(seat);
    if (this.#state.phase === 'ended') return;
    this.#state.phase = 'ended';
    this.#state.endedBy = seat;
    this.#battle = null;
    this.#nextReady = [false,false];
  }
  checkpoint() {
    const history=unpackCarryoverKnowledge(this.#carryoverKnowledge);
    return structuredClone({ version: 1, ...(this.#catalogVersion ? {catalogVersion:this.#catalogVersion} : {}), ...(this.#cpuDifficulty?{cpuDifficulty:this.#cpuDifficulty}:{}), ruleset:this.#ruleset, offerAudits:this.#offerAudits, state: this.#state, nextReady: this.#nextReady, carryoverUsedMoves:history.usedMoves,carryoverObservedItemIds:history.observedItemIds,carryoverObservedAbilityIds:history.observedAbilityIds, battle: this.#battle?.checkpoint() ?? null });
  }
  static restore(saved, offerProvider, { battleFactory, battleRestore, catalogVersion=null, cpuDifficulty=0 } = {}) {
    requireRule(saved?.version === 1, 'Unsupported saved match');
    const match = new Match(offerProvider, {battleFactory,ruleset:saved.ruleset,catalogVersion,cpuDifficulty:saved.cpuDifficulty??cpuDifficulty});
    match.#state = structuredClone(saved.state); match.#nextReady = [...saved.nextReady];
    match.#carryoverKnowledge=packCarryoverKnowledge(
      saved.carryoverUsedMoves??[{},{}],
      saved.carryoverObservedItemIds??[{},{}],
      saved.carryoverObservedAbilityIds??[{},{}]);
    match.#offerAudits=structuredClone(saved.offerAudits);
    if (saved.battle) {
      requireRule(typeof battleRestore === 'function', 'Battle restore unavailable');
      match.#battle = battleRestore(saved.battle);
       if(match.#state.phase==='battle' && match.#battle.result())match.finishBattle(match.#battle.result());
    }
    return match;
  }
  nextRound() {
    requireRule(this.#state.phase === 'result', 'Next round unavailable');
    const round = this.#state.round + 1;
    const nextCredits=this.#state.players.map((p,seat)=>p.credits+budgetFor(round)+(seat===1?this.#cpuHandicap.perRoundCreditBonus:0));
    requireRule(nextCredits.every(Number.isSafeInteger),'Credit balance overflow');
    const offers = this.#offers(round, this.#state.players.map(p => p.carry), nextCredits);
    this.#state.players.forEach((p, seat) => {
      p.fresh = offers[seat]; p.team = null; p.selected = null; p.teamCost = 0; p.credits = nextCredits[seat];
    });
    this.#battle = null; this.#nextReady = [false,false];
    this.#state.round = round;
    this.#state.phase = 'draft';
  }
  // Server-only CPU integration. Never include this in API responses:
  // it exposes own active sets but no hidden opponent identities or moves.
  decisionViewForAI(seat) {
    this.#player(seat);
    requireRule(this.#state.phase==='battle'&&this.#battle,
      'CPU decision view requires an active battle');
    return this.#battle.decisionViewForAI(seat);
  }
  // By default include the full presentation snapshot. Headless local
  // simulations can omit it without changing the public state or rules.
  view(seat,{includeBattle=true}={}) {
    const p = this.#player(seat);
    if (this.#state.phase === 'ended') return { phase:'ended', endedBy:this.#state.endedBy };
    const opponent = this.#state.players[1 - seat];
    const reveal = !['waiting', 'draft'].includes(this.#state.phase);
    const hasOpponentCarryoverView=(this.#state.phase==='draft'&&this.#state.round>1)||['result','complete'].includes(this.#state.phase);
    const knowledge=hasOpponentCarryoverView?unpackCarryoverKnowledge(this.#carryoverKnowledge):null;
    const carryoverPublic=knowledge
      ?projectCarryoverPublicView(opponent.carry,
        knowledge.usedMoves[1-seat],knowledge.observedItemIds[1-seat],
        knowledge.observedAbilityIds[1-seat]):null;
    return structuredClone({ ruleset:this.#ruleset, phase: this.#state.phase, round: this.#state.round,
      budget: p.credits, grant: budgetFor(this.#state.round), score: this.#state.score, history: this.#state.history,
      nextReady: this.#nextReady[seat], opponentNextReady: this.#nextReady[1-seat],
      battle: includeBattle?(this.#battle?.view(seat)??null):null,
      self: {...p,trainerAvatar:trainerAvatarId(p.trainerAvatar,seat)}, opponent: { trainerAvatar:trainerAvatarId(opponent.trainerAvatar,1-seat), ready: opponent.ready, teamConfirmed: opponent.team !== null,
        selectionConfirmed: opponent.selected !== null,
        ...(carryoverPublic??{}),
        species: reveal ? opponent.team.map(member => member.publicSpeciesName ?? (member.mega ? member.speciesId : member.speciesName ?? member.speciesId)) : [] } });
  }
}
