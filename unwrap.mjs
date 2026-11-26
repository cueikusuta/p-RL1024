import {readFileSync,writeFileSync} from 'node:fs';
import {createPrivateKey,privateDecrypt,createDecipheriv,createHash,constants} from 'node:crypto';
const [keyfile,source,out,digest]=process.argv.slice(2);
try{
 const pem=createPrivateKey(readFileSync(keyfile)), b=readFileSync(source);
 if(pem.asymmetricKeyType!=='rsa'||pem.asymmetricKeyDetails.modulusLength!==2048||
    b.length<285||b.length>24*1024**2||!/^[a-f0-9]{64}$/.test(digest)||
    createHash('sha256').update(b).digest('hex')!==digest)throw Error('Invalid seal');
 const key=privateDecrypt({key:pem,padding:constants.RSA_PKCS1_OAEP_PADDING,oaepHash:'sha256'},b.subarray(0,256));
 try{
  const d=createDecipheriv('aes-256-gcm',key,b.subarray(256,268));
  d.setAAD(Buffer.from('league-one-shot-bounded-v1'));d.setAuthTag(b.subarray(-16));
  const data=Buffer.concat([d.update(b.subarray(268,-16)),d.final()]);
  writeFileSync(out,data,{flag:'wx',mode:0o600});
 }finally{key.fill(0)}
}catch{console.error('Temporary bundle verification failed');process.exitCode=1}
