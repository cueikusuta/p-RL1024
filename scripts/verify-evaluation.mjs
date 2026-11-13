import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';

const fields = ['block', 'seed', 'rlSeat', 'winner', 'won', 'rounds', 'actionCounts', 'score'];
const canonical = value => {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map(canonical);
  if (typeof value === 'object') return Object.fromEntries(Object.keys(value).sort((a,b) => {
    const x = Array.from(a, c => c.codePointAt(0));
    const y = Array.from(b, c => c.codePointAt(0));
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
  }).map(k => [k, canonical(value[k])]));
  throw new Error('Invalid value in evaluation record');
};

export function normalize(report) {
  if (report.schema !== 'i975-frozen-paired-four-win-evaluation-v1' || report.purpose !== 'smoke' || report.blocks !== 2 || report.completedMatches !== 4 || report.formalEligible !== false || !Array.isArray(report.records) || report.records.length !== 4) throw new Error('Invalid smoke evaluation');
  const records = report.records.map(row => {
    if (!fields.every(k => Object.hasOwn(row,k))) throw new Error('Missing result field');
    if (!Number.isSafeInteger(row.block) || row.block < 0 || row.block > 1 || !Number.isSafeInteger(row.seed) || row.seed < 0 || ![0,1].includes(row.rlSeat) || ![0,1].includes(row.winner) || row.won !== Number(row.winner === row.rlSeat) || !Number.isSafeInteger(row.rounds) || row.rounds < 4 || !Array.isArray(row.score) || row.score.length !== 2 || row.score[row.winner] !== 4) throw new Error('Invalid Match record');
    return Object.fromEntries(fields.map(k => [k, canonical(row[k])]));
  }).sort((a,b) => a.block-b.block || a.rlSeat-b.rlSeat);
  if (new Set(records.map(r => `${r.block}:${r.rlSeat}`)).size !== 4 || records[0].seed !== records[1].seed || records[2].seed !== records[3].seed || records[0].seed === records[2].seed) throw new Error('Invalid paired seeds or seats');
  return records;
}

export function compare(a,b) {
  if (![a.modelSha256,b.modelSha256,a.seedDigestSha256,b.seedDigestSha256,a.outcomeDigestSha256,b.outcomeDigestSha256].every(x => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x))) throw new Error('Invalid provenance digest');
  const left = JSON.stringify(normalize(a));
  const right = JSON.stringify(normalize(b));
  if (a.modelSha256 !== b.modelSha256 || a.seedDigestSha256 !== b.seedDigestSha256 || left !== right) throw new Error('Model, seeds, or results differ');
  return {schema:'p-rl1024-normalized-outcome-v1',modelSha256:a.modelSha256,seedDigestSha256:a.seedDigestSha256,normalizedOutcomeSha256:createHash('sha256').update(left,'utf8').digest('hex'),originalOutcomeDigests:[a.outcomeDigestSha256,b.outcomeDigestSha256]};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 4) throw new Error('Usage: node scripts/verify-evaluation.mjs <evaluation-a.json> <evaluation-b.json>');
  console.log(JSON.stringify(compare(...process.argv.slice(2).map(p => JSON.parse(readFileSync(p,'utf8')))),null,2));
}
