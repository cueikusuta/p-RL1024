import {readFileSync} from 'node:fs';

const entries=JSON.parse(readFileSync(new URL('./registry.json',import.meta.url),'utf8'));
if (!Array.isArray(entries) || new Set(entries.map(e=>e.id)).size!==entries.length) throw new Error('Invalid model registry');
export function getModel(id) {
  const entry=entries.find(e=>e.id===id);
  if (!entry) throw new Error('Unknown model: '+id);
  return entry;
}
export const models=Object.freeze(entries);
