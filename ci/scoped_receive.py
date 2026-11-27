"""Private scratch packet receiver; exact frozen validators, no eval of payload."""
from pathlib import Path
import hashlib,json,struct,zipfile,re,os
import orjson,zstandard
from scripts.rnad import shared_league_contract as c
from scripts.rnad import pack_league_battle_audit as a
_LIMIT=24*1024**2
_MATCH=re.compile(r'g424-a([0-9]+)-s([0-9]+)-seat([01])\.json\.gz\Z')
def digest(b):return hashlib.sha256(b).hexdigest()
# Retain every per-Match validation in the frozen full-Actor verifier.
s=Path(c.__file__).read_text().split('def verify_actor(manifest, payload):',1)[1].split('\n\ndef expected_actor_roles',1)[0]
s='def actor_subset(manifest, payload):'+s
s=s.replace('len(matches) == MATCHES_PER_ACTOR','len(matches) == 8')
x=s.index('    expected = {(seed, seat)');y=s.index('    return {',x)
s=s[:x]+s[y:]
ns=dict(vars(c));exec(compile(s,c.__file__,'exec'),ns)
actor_subset=ns['actor_subset']
# Exact frozen audit correspondence/round-verification body.
s=Path(a.__file__).read_text().split('            pairs = ',1)[1].split('        except (OSError,',1)[0]
s='            pairs = '+s
ns={};exec(compile('def audit_match(row, match):\n'+''.join(l[8:] if l.startswith('        ') else l for l in s.splitlines(True)),a.__file__,'exec'),ns)
audit_match=ns['audit_match']
def envelope(path,plan,limit_roles):
    path=Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_size<=_LIMIT
    m=re.fullmatch(r'G424-'+re.escape(plan['runId'])+r'-windows-w([0-9]{3})\.zip',path.name)
    assert m
    with zipfile.ZipFile(path) as z:
       indexes=z.infolist();assert len(indexes)<=33 and sum(i.file_size for i in indexes)<=_LIMIT
       names=[i.filename for i in indexes];assert len(names)==len(set(names)) and 'envelope.json' in names
       a=orjson.loads(z.read('envelope.json'))
       assert a['schema']=='private-league-wave-v1' and a['run']==plan['runId'] and a['manifest_sha256']==plan['manifestSHA'] and a['host']=='windows' and a['wave']==int(m.group(1))
       by={p['task']:p for p in plan['packets'] if p['actor'] in limit_roles}
       tasks=a['tasks'];assert 0<len(tasks)<=16 and len(set(tasks))==len(tasks) and set(tasks)<=set(by)
       body={task:(z.read(task+'.json'),z.read(task+'.zst')) for task in tasks}
       assert set(names)=={'envelope.json'}|{t+ext for t in tasks for ext in ('.json','.zst')}
       for t,v in body.items():assert all(digest(b)==a['sha256'][t+k] for b,k in zip(v,('.json','.zst')))
       return a,body
def persist(path,data):
    with path.open('xb') as f:f.write(data)
def verify_packet(task,p,man,plan,receipt,packed,where):
    receipt=orjson.loads(receipt)
    assert receipt['schema']=='g423-portable-receipt-v1' and receipt['task']==task
    assert len(packed)==receipt['encodedBytes']<=32*1024**2 and digest(packed)==receipt['encodedSHA']
    n=receipt['rawBytes'];assert 8<n<=96*1024**2+32768 and zstandard.frame_content_size(packed)==n
    raw=zstandard.ZstdDecompressor().decompress(packed,max_output_size=n)
    assert len(raw)==n and digest(raw)==receipt['rawSHA'] and raw[:4]==b'GLR1'
    h=struct.unpack_from('>I',raw,4)[0];assert 0<h<=32768
    header=orjson.loads(raw[8:8+h]);assert header['task']==task
    fields=header['fields'];assert len(fields)==16
    at=h+8; decoded=[]
    for i,(name,typ,sz,expected_sha) in enumerate(fields):
        assert name==p['rows'][i//2]['name'] and typ==('accepted','audit')[i%2] and 0<sz<=16*1024**2 and at+sz<=n
        b=raw[at:at+sz];at+=sz
        assert digest(b)==expected_sha
        decoded.append((name,typ,b,orjson.loads(b)))
    assert at==n
    role,opp=c._opponent(man,p['actor'])
    matches=[r[3] for r in decoded[::2]]
    env={k:man[k] for k in ('generation','policy_id','main_sha256','magnet_sha256','source_sha','engine_sha')}
    env.update(schema=c.ACTOR_SCHEMA,actor_id=p['actor'],manifest_sha256=plan['manifestSHA'],
                 role=role,opponent_id=opp['id'],opponent_sha256=opp['sha256'],matches=matches)
    ids=actor_subset(man,env)
    proofs={}
    for (name,_,value,m),(_,_,audit_data,audit) in zip(decoded[::2],decoded[1::2],strict=True):
        assert name.removesuffix('.json.gz')==m['match_id'] in ids
        assert audit['auditSha']==man['source_sha'] and audit['generation']==man['generation'] and audit['actorId']==p['actor'] and audit['role']==role
        assert audit['runId']==plan['runId'] and str(audit['runAttempt'])==plan['runAttempt']
        assert audit['coverage']['dropped']==0
        audit_match(audit,m)
        out=where/'matches'/(name.removesuffix('.gz'))
        persist(out,value)
        audit_out=where/'detailed'/(name.removesuffix('.gz'))
        persist(audit_out,audit_data)
        proofs[m['match_id']]=(digest(value),digest(audit_data))
    return proofs
