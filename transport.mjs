/* Only approved archived-public G422 inputs. No plaintext GitHub result. */
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash,randomBytes,createCipheriv,createDecipheriv,publicEncrypt,privateDecrypt,constants} from 'node:crypto';
import assert from 'node:assert/strict';
const [mode,src,dst,keyfile,actor,epoch]=process.argv.slice(2);
assert(/^(?:0|[1-9]|1[0-9])$/.test(actor)&&/^[1-9][0-9]{7,15}$/.test(epoch));
const context=Buffer.from(JSON.stringify(['g422-hosted-full-v1',epoch,Number(actor)]));
const digest=x=>createHash('sha256').update(x).digest('hex');
const b=readFileSync(src), pem=readFileSync(keyfile);
assert(b.length>0&&b.length<64*1024*1024);
const options={key:pem,padding:constants.RSA_PKCS1_OAEP_PADDING,oaepHash:'sha256'};
if(mode==='seal'){
 const k=randomBytes(32),iv=randomBytes(12);
 try{
  const wrapped=publicEncrypt(options,k), cipher=createCipheriv('aes-256-gcm',k,iv,{authTagLength:16});
  assert(wrapped.length===256);
  cipher.setAAD(context);
  const body=Buffer.concat([cipher.update(b),cipher.final()]);
  const z=Buffer.concat([wrapped,iv,cipher.getAuthTag(),body]);
  writeFileSync(dst,z,{flag:'wx',mode:0o600});
  console.log(JSON.stringify({actor:Number(actor),sealedBytes:z.length,encryptedSHA:digest(z),notPlaintext:true}));
 }finally{k.fill(0)}
}else if(mode==='open'){
 assert(b.length>284);
 const k=privateDecrypt(options,b.subarray(0,256));
 try{
  const decipher=createDecipheriv('aes-256-gcm',k,b.subarray(256,268),{authTagLength:16});
  decipher.setAAD(context);decipher.setAuthTag(b.subarray(268,284));
  const z=Buffer.concat([decipher.update(b.subarray(284)),decipher.final()]);
  writeFileSync(dst,z,{flag:'wx',mode:0o600});
  console.log(JSON.stringify({actor:Number(actor),sealedBytes:b.length,rawSHA:digest(z),verifiedAuthenticatedPacket:true}));
 }finally{k.fill(0)}
}else throw Error('Unapproved envelope mode');
