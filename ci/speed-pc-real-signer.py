"""Publish short one-object read capabilities privately for one isolated PC test."""
import sys,hashlib,base64,subprocess,tempfile
from pathlib import Path
import speed_priority26_import as r
import os,json,time,urllib.error,urllib.parse
TOKEN=r.TOKEN
RUN=r.RUN
REPO=r.REPO
API=r.API
REF=f'refs/heads/ghspeed-pc-actual/{RUN}'
REMOTE='https://github.com/'+REPO+'.git'
def private_git(*args):
    env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_CONFIG_COUNT='3',
             GIT_CONFIG_KEY_0='http.extraHeader',GIT_CONFIG_VALUE_0='Authorization: Basic '+base64.b64encode(('x-access-token:'+TOKEN).encode()).decode(),
             GIT_CONFIG_KEY_1='credential.helper',GIT_CONFIG_VALUE_1='',
             GIT_CONFIG_KEY_2='http.followRedirects',GIT_CONFIG_VALUE_2='false')
    return subprocess.run(['git',*args],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=25).stdout

def verified_url(asset):
    try:
        with r.request('GET',API+f"/releases/assets/{asset['id']}",accept='application/octet-stream'):
            raise RuntimeError('Asset has no isolated object URL')
    except urllib.error.HTTPError as e:
        assert e.code in (302,303,307,308)
        url=e.headers['Location']
        p=urllib.parse.urlsplit(url)
        assert p.scheme=='https' and p.netloc in ('release-assets.githubusercontent.com','objects.githubusercontent.com') and p.query and not p.fragment
        stamp=asset.get('digest','')
        assert len(stamp)==71 and stamp.startswith('sha256:') and int(stamp[7:],16)>=0
        return dict(name=asset['name'],size=asset['size'],sha256=stamp[7:],url=url)
def publish(runtime,plan):
    d=r.wait_for(r.ready_draft)
    name='pc-runtime.tar.gz'
    assert 0<runtime.stat().st_size<=320*1024*1024
    expected_sha=hashlib.file_digest(runtime.open('rb'),'sha256').hexdigest()
    with runtime.open('rb') as obj:
        uri=f"{r.UPLOAD}/releases/{d['id']}/assets?"+urllib.parse.urlencode({'name':name})
        head={'Accept':'application/vnd.github+json','Authorization':'Bearer '+TOKEN,'Content-Type':'application/gzip',
              'Content-Length':str(runtime.stat().st_size),'User-Agent':'league-speed-pc-handoff','X-GitHub-Api-Version':'2022-11-28'}
        req=r.urllib.request.Request(uri,data=obj,headers=head,method='POST')
        with r.OPENER.open(req,timeout=120) as res:declared=json.load(res)
    assert declared['size']==runtime.stat().st_size and declared['digest']=='sha256:'+expected_sha
    keys=r.groups(plan)
    allowed={name,*keys}
    def poll():
        assets=r.obj('GET',f"/releases/{d['id']}/assets?per_page=100")
        assert all(x['name'] in allowed or x['name']==r.READY for x in assets)
        by={x['name']:x for x in assets if x['state']=='uploaded'}
        return by if allowed<=by.keys() else None
    assets=r.wait_for(poll,seconds=270)
    caps=[verified_url(assets[k]) for k in (name,*sorted(keys))]
    assert sum(x['size'] for x in caps)<1024**3
    with tempfile.TemporaryDirectory() as tmp:
        private_git('init','-q',tmp)
        (Path(tmp)/'transfer.json').write_text(json.dumps(dict(run=int(RUN),generation=512,assets=caps,
            waves=keys,bytes=sum(x['size'] for x in caps),manifest=json.loads(plan.read_text())['manifestSHA'])))
        private_git('-C',tmp,'add','transfer.json')
        private_git('-C',tmp,'-c','user.name=league-runner','-c','user.email=league-runner@users.noreply.github.com','commit','-qm','Temporary isolated specific-object transfer')
        private_git('-C',tmp,'push','-q',REMOTE,'HEAD:'+REF)
        print('Isolated PC batch: 24 checked packet waves and pinned runtime ready.',flush=True)
        # Explicit bounded lease, no extending the fixture after one real cohort.
        time.sleep(360)
def wipe():
    try:
        if private_git('ls-remote','--heads',REMOTE,REF).strip():
            private_git('push','-q',REMOTE,':'+REF)
        assert not private_git('ls-remote','--heads',REMOTE,REF).strip()
        print('Exact temporary PC transfer reference removed.',flush=True)
    finally:r.cleanup()
if __name__=='__main__':
    try:
        if sys.argv[1]=='publish':publish(Path(sys.argv[2]),Path(sys.argv[3]))
        elif sys.argv[1]=='create':r.create()
        elif sys.argv[1]=='cleanup':wipe()
        else:raise ValueError('Unknown isolated command')
    except Exception as e:
        print('PC relay diagnostic: '+type(e).__name__+' HTTP '+str(getattr(e,'code',0)),file=sys.stderr)
        sys.exit(1)
