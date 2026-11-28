"""Bounded private synthetic relay diagnostic ahead of priority-wave timing on the newer standard learner; never uses real league input."""
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
SIZE = 4096
SHARDS = 8
WAVES = 3
TOKEN = os.environ.pop('HANDOFF_TOKEN')
RUN = os.environ['GITHUB_RUN_ID']
assert re.fullmatch(r'[1-9][0-9]{5,14}', RUN) and os.environ['GITHUB_RUN_ATTEMPT'] == '1'
TAG = f'ghspeed-diag-{RUN}'
TITLE = f'Unpublished synthetic relay diagnostic {RUN}'

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
    try:
        return OPENER.open(req, timeout=12)
    except urllib.error.HTTPError as e:
        if e.code in (403,429):
            path=urllib.parse.urlsplit(req.full_url).path
            host=urllib.parse.urlsplit(req.full_url).hostname
            if host=='uploads.github.com': op='asset-upload'
            elif host=='github.com': op='repository-web'
            elif host!='api.github.com': op='binary-redirect'
            elif '/releases/assets/' in path: op='asset-download'
            elif path.endswith('/assets'): op='asset-list'
            elif method=='POST' and path.endswith('/releases'): op='draft-create'
            elif method=='DELETE': op='draft-delete'
            elif '/releases' in path: op='release-lookup'
            else: op='repository-check'
            remaining=e.headers.get('X-RateLimit-Remaining','')
            remaining=int(remaining) if re.fullmatch(r'[0-9]{1,9}',remaining) else None
            retry=bool(e.headers.get('Retry-After'))
            reset=bool(e.headers.get('X-RateLimit-Reset'))
            accepted_permissions=bool(e.headers.get('X-Accepted-GitHub-Permissions'))
            try: message=str(json.loads(e.read(8192)).get('message','')).lower()
            except (OSError,ValueError,AttributeError): message=''
            if 'secondary rate limit' in message or 'abuse detection' in message: reason='secondary-limit'
            elif remaining==0 or 'api rate limit exceeded' in message: reason='primary-limit'
            elif 'resource not accessible' in message or 'insufficient permission' in message: reason='permission'
            elif 'bad credentials' in message: reason='credential'
            elif 'ip allow list' in message: reason='ip-allow-list'
            else: reason='unclassified'
            safe=dict(op=op,auth=auth,status=e.code,reason=reason,remaining=remaining,
                      retry_after_present=retry,reset_present=reset,permission_header_present=accepted_permissions)
            print('Synthetic relay HTTP diagnostic: '+json.dumps(safe,sort_keys=True),flush=True)
        raise

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
        result=pred()
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
    r=own(obj('POST','/releases',dict(tag_name=TAG,name=TITLE,draft=True,body='Synthetic communication timing only. Never publish.')))
    anonymous_404('/releases/'+str(r['id']))
    assert r['upload_url']==f"{UPLOAD}/releases/{r['id']}/assets"+"{?name,label}"
    data=json.dumps({'run':RUN,'draft':r['id'],'private':True}).encode()
    url=f"{UPLOAD}/releases/{r['id']}/assets?"+urllib.parse.urlencode({'name':READY})
    with request('POST',url,data=data,content_type='application/json') as res:
        asset=json.load(res)
    assert asset['name']==READY and asset['state']=='uploaded' and asset['digest']=='sha256:'+hashlib.sha256(data).hexdigest()
    print('Private draft verified; anonymous request returned 404, and readiness marker committed.',flush=True)

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



ROUNDS=1
REQUEST_CEILING=8

def fixture(i,kind):
    return hashlib.shake_256(f'synthetic-coordination-{RUN}-{i}-{kind}'.encode()).digest(SIZE)

def name(i,kind):
    return f'{i:02}-{kind}.bin'

def send(release,i,kind):
    data=fixture(i,kind)
    url=f"{UPLOAD}/releases/{release['id']}/assets?"+urllib.parse.urlencode({'name':name(i,kind)})
    with request('POST',url,data=data,content_type='application/octet-stream') as res:
        a=json.load(res)
    assert a['name']==name(i,kind) and a['size']==SIZE and a['state']=='uploaded'
    assert a['digest']=='sha256:'+hashlib.sha256(data).hexdigest()

def receive(release,i,kind,count,deadline):
    label=name(i,kind)
    while time.monotonic()<deadline and count[0]<REQUEST_CEILING:
        count[0]+=1
        q=obj('GET',f"/releases/{release['id']}/assets?per_page=50")
        m=[a for a in q if a['name']==label and a['state']=='uploaded']
        assert len(m)<=1
        if m:
            a=m[0]
            assert a['size']==SIZE and type(a['id']) is int and a['id']>0
            data=read_binary(f"/releases/assets/{a['id']}")
            assert len(data)==SIZE and hashlib.sha256(data).digest()==hashlib.sha256(fixture(i,kind)).digest()
            return
        time.sleep(1.5)
    raise TimeoutError('bounded synthetic lookup')

def run(lead):
    r=wait_for(ready_draft,seconds=18)
    deadline=time.monotonic()+20
    count=[0]
    times=[]
    for i in range(ROUNDS):
        t=time.monotonic()
        if lead:
            send(r,i,'weights')
            receive(r,i,'gradient',count,deadline)
        else:
            receive(r,i,'weights',count,deadline)
            send(r,i,'gradient')
        times.append(time.monotonic()-t)
    if lead:
        print('Verified synthetic round trips in seconds: '+', '.join(f'{v:.3f}' for v in times),flush=True)

if __name__=='__main__':
    try:
        {'create':create,'leader':lambda:run(True),'worker':lambda:run(False),'cleanup':cleanup}[sys.argv[1]]()
    except Exception as exc:
        tail=exc.__traceback__
        while tail.tb_next and tail.tb_next.tb_frame.f_code.co_filename.endswith('speed-relay-diagnostic.py'):
            tail=tail.tb_next
        print(f'Synthetic latency test failed: {type(exc).__name__} HTTP {getattr(exc,"code",0)} line {tail.tb_lineno}',file=sys.stderr)
        sys.exit(1)
