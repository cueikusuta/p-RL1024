"""One isolated fresh-generation timing run; nothing is promoted."""
from pathlib import Path
import hashlib,os,subprocess,sys,json,threading,time,shutil,signal,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from speed_profile_receive import verify_packet,envelope,digest
from speed_prefetch_learner import train
from scripts.rnad import shared_league_learner as l
from scripts.rnad.shared_league_contract import _opponent,verify_manifest
import torch
def now():return time.time_ns()//1_000_000
def j(p):return json.loads(p.read_bytes())
def write(p,x):
    m=p.with_suffix('.part')
    with m.open('xb') as f:f.write(json.dumps(x).encode());f.flush();os.fsync(f.fileno())
    m.replace(p)
def sync_dir(p):
    f=os.open(p,os.O_DIRECTORY)
    try:os.fsync(f)
    finally:os.close(f)
def sync_file(p):
    with p.open('rb') as f:os.fsync(f.fileno())
def main():
    home=Path(__file__).resolve().parent
    here,boot=map(lambda x:Path(x).resolve(),sys.argv[1:3])
    epoch=now()
    start_setup=now()
    m=j(boot/'generation-manifest.json'); gen=m['generation']
    from scripts.rnad.shared_league_staging import staging_config
    staging_config()
    for folder in ('inbox','actors','actors/_strict_verified_matches','detailed','cloud'):(here/folder).mkdir(exist_ok=folder=='cloud')
    (here/'matches').symlink_to(here/'actors/_strict_verified_matches',target_is_directory=True)
    planp=here/'plan.json';p=j(planp)
    assert Path(p['bootstrap'])==boot and verify_manifest(m)==p['manifestSHA'] and p['generation']==gen
    for k,h in p['files'].items():
        if k.startswith(('bootstrap/actor/','bootstrap/magnet/','bootstrap/past/','bootstrap/fixed/')):
            assert digest((boot/k.removeprefix('bootstrap/')).read_bytes())==h
    by={x['task']:x for x in p['packets']}
    assert len(by)==160
    allowed={'cloud':by}
    rows=[{'match_id':r['name'].removesuffix('.json.gz'),'role':_opponent(m,x['actor'])[0]} for x in by.values() for r in x['rows']]
    ordered=[v['match_id'] for v in l._order(rows)]
    assert len(set(ordered))==1280 and p['optimizer_order']==ordered
    class Broker:
        def __init__(self):
            self.cv=threading.Condition()
            self.ready={};self.events=[];self.strict=None;self.failed=None;self.vm_actors_done=False
        def wait(self,pred):
            assert self.cv.wait_for(lambda:self.failed is not None or pred(),timeout=max(.1,(epoch+850000-now())/1000)), 'Escrow timeout'
            if self.failed:raise RuntimeError('Controller aborted') from self.failed
        def claim(self,work):
            ids=[Path(q[0]).stem for q in work];q=len(self.events)*32
            assert ids==ordered[q:q+32]
            with self.cv:
                self.wait(lambda:set(ids)<=self.ready.keys())
                self.events.append({'step':q//32+1,'released_ms':now()})
                jobs=[(i,path,role,self.ready[ids[i]][0]) for i,(path,role,_) in enumerate(work)]
            jobs.sort(key=lambda a:-Path(a[1]).stat().st_size*(1.7 if a[2]=='self' else 1))
            return jobs
        def completed(self,n):
            with self.cv:
                assert len(self.events)==n and 'updated_ms' not in self.events[-1]
                self.events[-1]['updated_ms']=now()
        def wait_for_barrier(self):
            with self.cv:self.wait(lambda:self.strict is not None)
    broker=Broker();results=[];errs=[]
    def work():
        try:results.append(train(boot,here,broker,m,rows))
        except Exception as e:errs.append(e)
    op=None;cloud=None;wave={};submitted=set();done=set();vs={};ready_at={}; queue={}
    ordered_rank={v:i for i,v in enumerate(ordered)}
    rank={k:min(ordered_rank[r['name'].removesuffix('.json.gz')] for r in by[k]['rows']) for k in by}
    try:
        vm_ready_ms=now()
        def admit(vpool,k,b,h):
            assert k not in submitted and k in allowed[h]
            submitted.add(k)
            queue[k]=b
        with ProcessPoolExecutor(3,mp_context=get_context('spawn')) as vpool:
            while not results:
                assert now()<=epoch+850000 and not (here/'inbox/abort').exists()
                if errs:raise errs[0]
                status=0
                broker.vm_actors_done=(status==0)
                available=int(next(q.split()[1] for q in Path('/proc/meminfo').read_text().splitlines() if q.startswith('MemAvailable:')))
                if op is None and (status==0 or (available>=4*1024**2 and set(ordered[:32])<=broker.ready.keys())):
                    op=threading.Thread(target=work,daemon=True);op.start()
                for k in allowed['cloud']:
                    base=here/'cloud/packets';j1=base/(k+'.json')
                    if k not in submitted and j1.exists():
                        admit(vpool,k,(j1.read_bytes(),(base/(k+'.zst')).read_bytes()),'cloud')
                while queue and len(vs)<6:
                    k=min(queue,key=rank.__getitem__);b=queue.pop(k)
                    vs[vpool.submit(verify_packet,k,by[k],m,p,*b,here)]=k
                for f in [f for f in vs if f.done()]:
                    k=vs.pop(f);proof=f.result()
                    with broker.cv:
                        assert len(proof)==8 and not (set(proof)&broker.ready.keys())
                        broker.ready.update(proof);broker.cv.notify_all()
                    done.add(k);ready_at[k]=now()
                if len(done)==160 and broker.strict is None:
                    assert set(ordered)==broker.ready.keys() 
                    for marker in wave.values():sync_file(here/'inbox'/marker['name'])
                    sync_dir(here/'inbox')
                    sync_dir(here/'cloud/packets') # packer already fsynced packet bytes
                    write(here/'strict.json',{'accepted':1280,'audits':1280,'roles':20,'all_packet_times_ms':ready_at,'pairs':broker.ready})
                    sync_dir(here)
                    with broker.cv:
                        broker.strict=now();broker.cv.notify_all()
                if vs:wait(vs,timeout=.06,return_when=FIRST_COMPLETED)
                else:time.sleep(.06)
        assert op is not None
        op.join(4);assert not op.is_alive() and not errs
        cp=here/f'output/checkpoint-g{gen+1}.pt';idx=cp.with_name(f'generation{gen+1}-index.json')
        sync_file(cp); sync_file(idx);sync_dir(cp.parent)
        index=j(idx);assert digest(cp.read_bytes())==index['full_checkpoint_sha256']
        q=torch.load(cp,map_location='cpu',weights_only=True)
        assert q['generation']==gen+1 and q['step']==0 and q['is_champion'] is False and set(q['completed_generation_match_ids'])==set(ordered)
        rx=here/'cloud/work-result.json'
        for _ in range(60):
            if rx.is_file():break
            time.sleep(.1)
        assert rx.is_file()
        end=now()
        write(here/'result.json',{'mode':'production-ten-hour-github','dispatch_ms':epoch,'vm_controller_active_setup_ms':vm_ready_ms-start_setup,
           'strict_barrier_ms':broker.strict,'verified_save_ms':end,'active_elapsed_s':(end-epoch)/1000,'cloud':j(here/'cloud/work-result.json'),
           'accepted':1280,'audits':1280,'actual_ordered_updates':len(broker.events),'optimizer_events':broker.events,
           'windows_waves':wave,'checkpoint_sha256':index['full_checkpoint_sha256'],
           'generation':gen,'production_launched':True})
    except BaseException as e:
        with broker.cv:
            broker.failed=e;broker.cv.notify_all()
        raise
    finally:
        if cloud and cloud.poll() is None:
            cloud.terminate()
            cloud.wait(timeout=3)
if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        points=' '.join(f'{Path(t.filename).name}:{t.lineno}' for t in traceback.extract_tb(exc.__traceback__)[-6:])
        print(f'Speed controller diagnostic: {type(exc).__name__} {points}',file=sys.stderr,flush=True)
        raise
