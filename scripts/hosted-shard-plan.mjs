import {pathToFileURL} from 'node:url';
export function planShards({mode='evaluation',total,shards,seed}) {
  const min=mode==='training'?9:1,max=mode==='training'?256:100000;
  if(!['training','evaluation'].includes(mode)||!Number.isSafeInteger(total)||total<min||total>max||
     !Number.isSafeInteger(shards)||shards<1||shards>20||total<min*shards||
     !Number.isSafeInteger(seed)||seed<1||seed+total-1>0xffffffff)
    throw new Error('Invalid total, shards, seed or shard minimum');
  let offset=0;
  const include=Array.from({length:shards},(_,index)=>{
    const count=Math.floor(total/shards)+(index<total%shards?1:0);
    const row={index,offset,count,seedStart:seed+offset,seedEnd:seed+offset+count-1};
    offset+=count;
    return row;
  });
  if(offset!==total)throw new Error('Shard plan lost matches');
  return {include};
}
function parse(argv) {
  const options=new Map();
  if(argv.length%2)throw new Error('Expected option/value pairs');
  for(let i=0;i<argv.length;i+=2) {
    if(!['--mode','--total','--shards','--seed'].includes(argv[i])||options.has(argv[i]))throw new Error('Invalid option');
    options.set(argv[i],argv[i+1]);
  }
  return {mode:options.get('--mode')||'evaluation',total:Number(options.get('--total')),
    shards:Number(options.get('--shards')),seed:Number(options.get('--seed'))};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href)
  console.log(JSON.stringify(planShards(parse(process.argv.slice(2)))));
