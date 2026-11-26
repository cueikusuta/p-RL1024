"""Generate one exact f64 SIMD dot kernel, no toolchain or new dependency."""
from pathlib import Path
def leb(n):
    assert n>=0
    out=[]
    while n>=128:out.append((n&127)|128);n>>=7
    return bytes([*out,n])
def vec(xs):
    return leb(len(xs))+b''.join(xs)
def blob(x):
    return leb(len(x))+x
def section(k,data):
    return bytes([k])+blob(data)
def op(x):return bytes([x])
def get(i):return op(0x20)+leb(i)
def put(i):return op(0x21)+leb(i)
def const(v):
    assert 0<=v<64
    return bytes([0x41,v])
def plus(a,b):return a+b+op(0x6a)
def product(a,b):return a+b+op(0x6c)
def byteoff(i):return product(i,const(8))
def simd(i,imm=b''):return op(0xfd)+leb(i)+imm
load=lambda address:address+simd(0,b'\x03\x00')
store=lambda at,v:at+v+simd(11,b'\x03\x00')
def loop(idx,limit,increment,body):
    return (const(0)+put(idx)+bytes([2,0x40,3,0x40])+
            get(idx)+get(limit)+bytes([0x4f,0x0d,0x01])+body+
            plus(get(idx),const(increment))+put(idx)+bytes([0x0c,0,0x0b,0x0b]))
# (m, n, x_byte, w_byte, bias_byte, dest_byte) -> void.
# Local 6 = k, 7 = i, 8 = broadcast(x[k]).
m,n,x,w,b,z,k,i,t=range(9)
dst=plus(get(z),byteoff(get(i)))
bias=plus(get(b),byteoff(get(i)))
wk=plus(get(w),byteoff(plus(product(get(k),get(n)),get(i))))
xk=plus(get(x),byteoff(get(k)))
body=loop(i,n,2,store(dst,load(bias)))
step=xk+bytes([0x2b,3,0])+simd(20)+put(t)
step+=loop(i,n,2,store(dst,load(dst)+load(wk)+get(t)+simd(242)+simd(240)))
body+=loop(k,m,1,step)+op(0x0b)
t=section(1,vec([bytes([0x60,6]+[0x7f]*6+[0])]))
f=section(3,vec([bytes([0])]))
mem=section(5,vec([bytes([1])+leb(256)+leb(256)]))
exports=section(7,vec([blob(b'memory')+bytes([2,0]),blob(b'dot')+bytes([0,0])]))
code=section(10,vec([blob(vec([b'\x02\x7f',b'\x01\x7b'])+body)]))
Path(__file__).with_name('affine.wasm').write_bytes(b'\0asm\x01\0\0\0'+t+f+mem+exports+code)
