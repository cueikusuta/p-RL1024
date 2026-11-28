"""Run-scoped private relay for the isolated distributed benchmark."""
import concurrent.futures
import base64
import hashlib
import json
import os
import re
import sys
import time
import subprocess
import urllib.error
import urllib.parse
import urllib.request

REPO = 'cueikusuta/p-rl1024-handoff-private'
API = f'https://api.github.com/repos/{REPO}'
UPLOAD = f'https://uploads.github.com/repos/{REPO}'
SIZE = 96 * 1024 * 1024
SHARDS = 8
WAVES = 3
TOKEN = os.environ.pop('HANDOFF_TOKEN')
RUN = os.environ['GITHUB_RUN_ID']
assert re.fullmatch(r'[1-9][0-9]{5,14}', RUN) and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
ROUND=os.environ['LEAGUE_WARM_ROUND']
assert ROUND in ('1','2','3')
TAG=f'ghspeed-chain-{RUN}-{ROUND}'
TITLE=f'Unpublished chained benchmark {RUN}-{ROUND}'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

OPENER = urllib.request.build_opener(NoRedirect())

def request(method, url, *, auth=True, accept='application/vnd.github+json', data=None, content_type=None):
    headers = {'Accept': accept, 'User-Agent':'league-speed-relay', 'X-GitHub-Api-Version':'2022-11-28'}
    if auth:
        assert url==API or url.startswith((API+'/', UPLOAD+'/'))
        headers['Authorization'] = 'Bearer ' + TOKEN
    if content_type:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(url, headers=headers, method=method, data=data)
    return OPENER.open(req, timeout=30)

def obj(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    with request(method, API+path,data=data,content_type='application/json' if data else None) as res:
        return json.load(res) if res.status != 204 else None

def own(r):
    assert r['tag_name'] == TAG and r['name'] == TITLE and r['draft'] is True and r.get('published_at') is None
    assert type(r['id']) is int and r['id'] > 0
    return r

def lookup():
    hits = [i for i in obj('GET','/releases?per_page=100') if i['tag_name']==TAG]
    assert len(hits)<=1
    return own(hits[0]) if hits else None

def wait_for(pred, seconds=225):
    expiry = time.monotonic()+seconds
    while time.monotonic() < expiry:
        try:
            result=pred()
        except urllib.error.HTTPError as e:
            if e.code not in (403,429) or not (e.headers.get('Retry-After') or e.headers.get('X-RateLimit-Remaining')=='0'):
                raise
            retry=e.headers.get('Retry-After')
            reset=e.headers.get('X-RateLimit-Reset')
            delay=(int(retry) if retry and retry.isdecimal() else
                   max(0,int(reset)-int(time.time())) if reset and reset.isdecimal() else 10)
            if time.monotonic()+max(delay,5)>=expiry:raise
            time.sleep(max(delay,5))
            continue
        if result is not None: return result
        time.sleep(2)
    raise TimeoutError('relay deadline')

READY='readiness.json'

def anonymous_404(path):
    try:
        with request('GET',API+path,auth=False):
            raise AssertionError('private endpoint accessible without auth')
    except urllib.error.HTTPError as e:
        if e.code==404:return
        if e.code not in (403,429):raise
    # Shared Actions egress can be rate-limited for unauthenticated API calls.
    # The unauthenticated repository web URL must independently remain 404.
    try:
        with request('GET','https://github.com/'+REPO,auth=False,accept='text/html'):
            raise AssertionError('private repository is publicly reachable')
    except urllib.error.HTTPError as e:
        if e.code!=404:raise

def create():
    repo=obj('GET','')
    assert repo['private'] is True and repo['visibility']=='private' and repo['full_name']==REPO
    assert lookup() is None
    r=own(obj('POST','/releases',dict(tag_name=TAG,name=TITLE,draft=True,body='Temporary scoped development benchmark packets. Never publish.')))
    anonymous_404('/releases/'+str(r['id']))
    assert r['upload_url']==f"{UPLOAD}/releases/{r['id']}/assets"+"{?name,label}"
    data=json.dumps({'run':RUN,'draft':r['id'],'private':True}).encode()
    url=f"{UPLOAD}/releases/{r['id']}/assets?"+urllib.parse.urlencode({'name':READY})
    with request('POST',url,data=data,content_type='application/json') as res:
        asset=json.load(res)
    assert asset['name']==READY and asset['state']=='uploaded' and asset['digest']=='sha256:'+hashlib.sha256(data).hexdigest()
    print('Private draft verified; anonymous request returned 404, and readiness marker committed.',flush=True)

def wait_verified(round_no):
    assert 1<=round_no<3
    ref=f'refs/heads/ghspeed-chain/{RUN}/verified-{round_no}'
    # Use Git to read exact Git refs; rate-limited REST is reserved for the
    # actual release transport. Never put a token in arguments or logs.
    assert TOKEN
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_CONFIG_COUNT='3',
             GIT_CONFIG_KEY_0='http.extraHeader',GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+TOKEN).encode()).decode('ascii'),
             GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='',
             GIT_CONFIG_KEY_2='http.followRedirects',GIT_CONFIG_VALUE_2='false')
    deadline=time.monotonic()+330
    delay=4.0
    while time.monotonic()<deadline:
        p=subprocess.run(['git','ls-remote','--exit-code','--heads','https://github.com/'+REPO+'.git',ref],
                         env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=15,check=False)
        if p.returncode==0:
            assert re.fullmatch(rb'[0-9a-f]{40}\t'+re.escape(ref.encode())+rb'\n?',p.stdout)
            return p.stdout.split(b'\t',1)[0].decode('ascii')
        if p.returncode not in (2,128):raise RuntimeError('Git reference lookup failed')
        time.sleep(delay)
        if p.returncode==128:delay=min(delay*2,20)
    raise TimeoutError('previous verified chain checkpoint absent')

def gate():
    assert ROUND in ('2','3')
    stamp=wait_verified(int(ROUND)-1)
    wait_for(ready_draft,seconds=45)
    print(stamp,flush=True)

def await_previous():
    wait_verified(int(ROUND))

def ready_draft():
    r=lookup()
    if r is None: return None
    markers=[a for a in r.get('assets',[]) if a['name']==READY and a['state']=='uploaded']
    assert len(markers)<=1
    if not markers: return None
    a=markers[0]
    assert type(a['id']) is int and a['id']>0 and a['size']<512
    body=read_binary(f"/releases/assets/{a['id']}")
    assert len(body)==a['size'] and a['digest']=='sha256:'+hashlib.sha256(body).hexdigest()
    assert json.loads(body)=={'run':RUN,'draft':r['id'],'private':True}
    return r

def read_binary(path):
    try:
        with request('GET',API+path,accept='application/octet-stream') as res:
            return res.read(SIZE+1)
    except urllib.error.HTTPError as e:
        if e.code not in (302,303,307,308):
            raise
        url=e.headers['Location']
        p=urllib.parse.urlsplit(url)
        assert p.scheme=='https' and p.netloc in ('release-assets.githubusercontent.com','objects.githubusercontent.com')
        with request('GET',url,auth=False,accept='application/octet-stream') as res:
            return res.read(SIZE+1)

def cleanup():
    r=lookup()
    if r is None:
        print('No matching draft to clean.',flush=True)
        return
    id=r['id']
    obj('DELETE',f'/releases/{id}')
    try:
        obj('GET',f'/releases/{id}')
    except urllib.error.HTTPError as e:
        assert e.code==404 and lookup() is None
        print('Exact draft and attached assets removed.',flush=True)
        return
    raise AssertionError('draft still present')


from pathlib import Path
import io
import tarfile

def groups(plan):
    p=json.loads(plan.read_bytes())
    leases=p['prioritizedLeases']
    assert len(leases)==80 and {a for _,x,y in leases for a in (x,y)}=={x['task'] for x in p['packets']}
    first32=set(p['optimizer_order'][:32])
    packet_names={z['task']:{row['name'].removesuffix('.json.gz') for row in z['rows']} for z in p['packets']}
    hot=sum(bool(first32 & (packet_names[a] | packet_names[b])) for _,a,b in leases)
    first=(hot+SHARDS-1)//SHARDS
    assert 1<=first<=4
    mid=(first+10)//2
    out={}
    for shard in range(SHARDS):
        v=leases[shard::SHARDS]
        assert len(v)==10
        for wave in range(WAVES):
            lo,hi=((0,first),(first,mid),(mid,10))[wave]
            batch=v[lo:hi]
            out[f's{shard}-w{wave}.tar']=[key for _,x,y in batch for key in (x,y)]
    assert sorted(x for m in out.values() for x in m)==sorted(x['task'] for x in p['packets'])
    assert all(re.fullmatch(r'[A-Za-z0-9_-]{1,60}',x) for m in out.values() for x in m)
    return out

def upload(directory,plan,shard):
    assert 0<=shard<SHARDS
    r=wait_for(ready_draft)
    assert r['upload_url']==f"{UPLOAD}/releases/{r['id']}/assets"+"{?name,label}"
    start=time.monotonic()
    for name,keys in groups(plan).items():
        if not name.startswith(f's{shard}-'): continue
        files=[directory/'packets'/(k+ext) for k in keys for ext in ('.zst','.json')]
        wait_for(lambda: True if all(p.is_file() and not p.is_symlink() for p in files) else None)
        buf=io.BytesIO()
        with tarfile.open(fileobj=buf,mode='w') as tar:
            for p in files:
                b=p.read_bytes()
                assert 0<len(b)<=32*1024**2
                info=tarfile.TarInfo(p.name);info.size=len(b);info.mode=0o600
                tar.addfile(info,io.BytesIO(b))
        data=buf.getvalue()
        assert 0<len(data)<=SIZE
        url=f"{UPLOAD}/releases/{r['id']}/assets?"+urllib.parse.urlencode({'name':name})
        with request('POST',url,data=data,content_type='application/octet-stream') as res:
            asset=json.load(res)
        assert asset['name']==name and asset['state']=='uploaded' and asset['size']==len(data)
        assert asset['digest']=='sha256:'+hashlib.sha256(data).hexdigest()
    print(f'Shard published three verified waves in {time.monotonic()-start:.3f}s.',flush=True)

def receive(directory,plan):
    r=wait_for(ready_draft)
    g=groups(plan); pending=set(g)
    dst=directory/'packets';dst.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    def fetch(asset):
        name=asset['name']
        assert type(asset['id']) is int and asset['id']>0 and 0<asset['size']<=SIZE
        body=read_binary(f"/releases/assets/{asset['id']}")
        assert len(body)==asset['size'] and asset['digest']=='sha256:'+hashlib.sha256(body).hexdigest()
        names=[k+ext for k in g[name] for ext in ('.zst','.json')]
        with tarfile.open(fileobj=io.BytesIO(body),mode='r:') as t:
            members=t.getmembers()
            assert [m.name for m in members]==names
            assert all(m.isfile() and 0<m.size<=32*1024**2 for m in members)
            for m in members:
                part=dst/(m.name+'.part')
                with part.open('xb') as w:
                    w.write(t.extractfile(m).read())
                part.replace(dst/m.name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[]
        while pending:
            assert time.monotonic()-start<225
            for f in futures:
                if f.done(): f.result()
            assets=obj('GET',f"/releases/{r['id']}/assets?per_page=100")
            assert all(a['name'] in g or a['name']==READY for a in assets)
            for a in assets:
                if a['state']=='uploaded' and a['name'] in pending:
                    pending.remove(a['name']); futures.append(pool.submit(fetch,a))
            if pending: time.sleep(1)
        for f in futures: f.result()
    print(f'All {len(g)} waves authenticated and received in {time.monotonic()-start:.3f}s.',flush=True)
    part=directory/'work-result.part';part.write_text(json.dumps({'end':time.time_ns()//1_000_000,'waves':len(g)}))
    part.replace(directory/'work-result.json')

if __name__=='__main__':
    try:
        cmd=sys.argv[1]
        if cmd=='upload': upload(Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4]))
        elif cmd=='receive': receive(Path(sys.argv[2]),Path(sys.argv[3]))
        else: {'create':create,'cleanup':cleanup,'gate':gate,'await_previous':await_previous}[cmd]()
    except Exception as exc:
        loc=exc.__traceback__
        while loc.tb_next and loc.tb_next.tb_frame.f_code.co_filename.endswith('speed-prefetch-chain-relay.py'):
            loc=loc.tb_next
        print(f'Relay diagnostic: {type(exc).__name__} HTTP {getattr(exc,"code",0)} line {loc.tb_lineno}',file=sys.stderr)
        sys.exit(1)
