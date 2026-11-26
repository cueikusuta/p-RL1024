'use strict';
// Fixed synthetic-data-only verification. No plaintext, URL, archive, or credential output.
const fs=require('node:fs'),c=require('node:crypto'),assert=require('node:assert/strict');
try {
 const secret=process.env.LEAGUE_SYNTHETIC_KEY;
 if(!/^[0-9a-f]{64}$/.test(secret??''))throw new Error('Credential unavailable');
 const key=Buffer.from(secret,'hex'); delete process.env.LEAGUE_SYNTHETIC_KEY;
 const sealed=fs.readFileSync('dummy-ciphertext.bin');
 if(sealed.length!==65564)throw new Error('Wrong synthetic packet size');
 const d=c.createDecipheriv('aes-256-gcm',key,sealed.subarray(0,12));
 d.setAAD(Buffer.from('league-synthetic-handoff-v1'));d.setAuthTag(sealed.subarray(12,28));
 const plain=Buffer.concat([d.update(sealed.subarray(28)),d.final()]);
 key.fill(0);
 const result=c.createHash('sha256').update(plain).digest('hex');
 assert.equal(plain.length,65536);
 assert.equal(result,'4ce2a9d8a6db2067b5fb0f669e7b37125467ea64d2ce0bce89ac5c14ed7600ef');
 plain.fill(0);
 process.stdout.write(JSON.stringify({test:'fixed-synthetic-v1',verified_bytes:65536,exact_digest:result,all_real_data_excluded:true})+'\n');
} catch { process.stderr.write('Synthetic scope or integrity check was rejected\n');process.exitCode=1; }
