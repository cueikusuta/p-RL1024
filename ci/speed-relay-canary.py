"""Synthetic-only transfer trial. Never publish a release or stage real league data."""
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'cueikusuta/p-rl1024-handoff-private'
API = f'https://api.github.com/repos/{REPO}'
UPLOAD = f'https://uploads.github.com/repos/{REPO}'
SIZE = 48 * 1024 * 1024
SHARDS = 4
TOKEN = os.environ.pop('HANDOFF_TOKEN')
RUN = os.environ['GITHUB_RUN_ID']
assert re.fullmatch(r'[1-9][0-9]{5,14}', RUN) and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
TAG = f'ghspeed-relay-{RUN}'
TITLE = f'Unpublished synthetic speed relay {RUN}'

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

OPENER = urllib.request.build_opener(NoRedirect())

def request(method, url, *, auth=True, accept='application/vnd.github+json', data=None, content_type=None):
    headers = {'Accept': accept, 'User-Agent':'league-speed-relay', 'X-GitHub-Api-Version':'2022-11-28'}
    if auth:
        assert url.startswith((API+'/', UPLOAD+'/'))
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

def wait_for(pred, seconds=130):
    expiry = time.monotonic()+seconds
    while time.monotonic() < expiry:
        result=pred()
        if result is not None: return result
        time.sleep(2)
    raise TimeoutError('relay deadline')

def synthetic(i):
    assert 0<=i<SHARDS
    return hashlib.shake_256(f'public-fixture-{RUN}-{i}'.encode()).digest(SIZE)

def asset_name(i):
    return f'fixture-{i}.bin'

def anonymous_404(path):
    try:
        with request('GET',API+path,auth=False) as res:
            raise AssertionError('private endpoint accessible without auth')
    except urllib.error.HTTPError as e:
        assert e.code==404

def create():
    assert lookup() is None
    r=own(obj('POST','/releases',dict(tag_name=TAG,name=TITLE,draft=True,body='Synthetic CI transport test only. Never publish.')))
    anonymous_404('/releases/'+str(r['id']))
    print('Private draft verified; anonymous request returned 404.',flush=True)

def upload(i):
    data=synthetic(i)
    r=wait_for(lookup)
    assert r['upload_url']==f"{UPLOAD}/releases/{r['id']}/assets"+"{?name,label}"
    url=f"{UPLOAD}/releases/{r['id']}/assets?"+urllib.parse.urlencode({'name':asset_name(i)})
    start=time.monotonic()
    with request('POST',url,data=data,content_type='application/octet-stream') as res:
        asset=json.load(res)
    assert asset['name']==asset_name(i) and asset['state']=='uploaded' and asset['size']==SIZE
    if asset.get('digest'):
        assert asset['digest']=='sha256:'+hashlib.sha256(data).hexdigest()
    print(f'Synthetic shard transfer: {time.monotonic()-start:.3f}s upload.',flush=True)

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

def read():
    r=wait_for(lookup)
    def assets():
        items=obj('GET',f"/releases/{r['id']}/assets?per_page=100")
        by={i['name']:i for i in items if i['state']=='uploaded'}
        assert set(by)<=set(map(asset_name,range(SHARDS)))
        return by if len(by)==SHARDS else None
    start=time.monotonic()
    by=wait_for(assets)
    def verify(i):
        a=by[asset_name(i)]
        assert type(a['id']) is int and a['id']>0 and a['size']==SIZE
        path=f"/releases/assets/{a['id']}"
        anonymous_404(path)
        t=time.monotonic()
        data=read_binary(path)
        assert len(data)==SIZE and hashlib.sha256(data).digest()==hashlib.sha256(synthetic(i)).digest()
        return time.monotonic()-t
    with concurrent.futures.ThreadPoolExecutor(max_workers=SHARDS) as pool:
        times=list(pool.map(verify,range(SHARDS)))
    print(f'Verified {SHARDS} x 48 MiB; max download+hash {max(times):.3f}s, wait-to-verification {time.monotonic()-start:.3f}s.',flush=True)

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

if __name__=='__main__':
    try:
        cmd=sys.argv[1]
        if cmd=='upload': upload(int(sys.argv[2]))
        else: {'create':create,'read':read,'cleanup':cleanup}[cmd]()
    except Exception as exc:
        loc=exc.__traceback__
        while loc.tb_next:
            if not loc.tb_next.tb_frame.f_code.co_filename.endswith('speed-relay-canary.py'): break
            loc=loc.tb_next
        print(f'Synthetic relay failed: {type(exc).__name__} HTTP {getattr(exc,"code",0)} line {loc.tb_lineno}',file=sys.stderr)
        sys.exit(1)
