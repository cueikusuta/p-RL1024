import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';

const data=JSON.parse(readFileSync('provenance/manifest.json','utf8'));
const entries=Object.entries(data.files);
if (entries.length !== 86) throw new Error('Unexpected source file count');
for (const [path,expected] of entries) {
  if (!/^(src|data\/catalog|scripts\/dev\/simulator|test)\/[a-zA-Z0-9_.\/-]+$/.test(path) || path.split('/').includes('..')) throw new Error(`Invalid source path: ${path}`);
  const bytes=readFileSync(path);
  const sha256=createHash('sha256').update(bytes).digest('hex');
  const gitBlob=createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
  if (sha256!==expected.sha256 || gitBlob!==expected.gitBlob) throw new Error(`Source mismatch: ${path}`);
}
console.log(`Verified ${entries.length} unchanged source files from ${data.sourceCommit}`);
