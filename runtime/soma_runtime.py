#!/usr/bin/env python3
"""SOMA Runtime v3.0 — complete vector registers, real SOM_WALK, SPAWN_MAP grid"""
import sys,struct,math,threading,random
from collections import defaultdict

HEADER_SIZE=32; MEM_FLAG=0x80000000; VEC_DIM=8; MAX_CALL=1024
DTYPE_NAME={0:'INT',1:'FLOAT',2:'VEC',3:'WGHT',4:'COORD',5:'BYTES'}

class VecReg:
    __slots__=('v',)
    def __init__(self,vals=None):
        v=list(vals) if vals else []
        self.v=(v+[0.0]*VEC_DIM)[:VEC_DIM]
    def as_int(self):
        v=self.v[0]
        if not (v==v) or v==float("inf") or v==float("-inf"): return 0
        return int(max(-2147483.0,min(2147483.0,v))*1000)&0xFFFFFFFF
    def dot(self,o):  return sum(a*b for a,b in zip(self.v,o.v))
    def norm(self):   return math.sqrt(sum(x*x for x in self.v))
    def normalized(self):
        n=self.norm(); return VecReg([x/n for x in self.v]) if n>1e-9 else VecReg()
    def distance(self,o): return math.sqrt(sum((a-b)**2 for a,b in zip(self.v,o.v)))
    def __repr__(self): return f"VEC[{','.join(f'{x:.3f}' for x in self.v)}]"

class DataMemory:
    def __init__(self,data):
        self.symbols={}; self.payload=bytearray()
        if data: self._parse(data)
    def _parse(self,data):
        off=0; (n,)=struct.unpack_from('>I',data,off); off+=4
        for _ in range(n):
            (nl,)=struct.unpack_from('>H',data,off); off+=2
            name=data[off:off+nl].decode(); off+=nl
            (dt,po,cnt)=struct.unpack_from('>BII',data,off); off+=9
            self.symbols[name]={'type':dt,'offset':po,'count':cnt}
        self.payload=bytearray(data[off:])
    def rf(self,off):
        return struct.unpack_from('>f',self.payload,off)[0] if off+4<=len(self.payload) else 0.0
    def wf(self,off,v):
        if off+4<=len(self.payload): struct.pack_into('>f',self.payload,off,float(v))
    def read_vec(self,off,cnt=VEC_DIM): return VecReg([self.rf(off+i*4) for i in range(min(cnt,VEC_DIM))])
    def write_vec(self,off,vec):
        for i,v in enumerate(vec.v): self.wf(off+i*4,v)
    def find_by_offset(self,off):
        for s in self.symbols.values():
            if s['offset']==off: return s
        return None
    def dump(self):
        for name,s in self.symbols.items():
            vals=[self.rf(s['offset']+i*4) for i in range(min(s['count'],4))]
            ell='...' if s['count']>4 else ''
            print(f"    {name:20s} {DTYPE_NAME[s['type']]:6s} count={s['count']:4d}  [{', '.join(f'{v:.4f}' for v in vals)}{ell}]")

class SOM:
    def __init__(self,rows=16,cols=16,dim=VEC_DIM,lr=0.5,nbhd=3.0):
        self.rows=rows; self.cols=cols; self.dim=dim; self.lr=lr; self.nbhd=nbhd
        self.W=[[VecReg([random.gauss(0.5,0.15) for _ in range(dim)]) for _ in range(cols)] for _ in range(rows)]
        self._lock=threading.Lock()
    def bmu(self,vec):
        best,br,bc=float('inf'),0,0
        for r in range(self.rows):
            for c in range(self.cols):
                d=vec.distance(self.W[r][c])
                if d<best: best=d;br=r;bc=c
        return br,bc
    def train(self,vec,br,bc):
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    h=math.exp(-((r-br)**2+(c-bc)**2)/(2*self.nbhd**2))
                    w=self.W[r][c]
                    w.v=[wk+self.lr*h*(vk-wk) for wk,vk in zip(w.v,vec.v)]
    def sense(self,r,c):
        s=sum(self.W[r][c].v)/self.dim
        return max(-1.0,min(1.0,s))
    def elect(self):
        best,br,bc=-1,0,0
        for r in range(self.rows):
            for c in range(self.cols):
                v=self.sense(r,c)
                if v>best: best=v;br=r;bc=c
        return br,bc
    def walk_gradient(self,r,c):
        best=self.sense(r,c); dr=dc=0
        for nr,nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if 0<=nr<self.rows and 0<=nc<self.cols:
                a=self.sense(nr,nc)
                if a>best: best=a;dr=nr-r;dc=nc-c
        return max(0,min(self.rows-1,r+dr)),max(0,min(self.cols-1,c+dc))
    def node_dist(self,r1,c1,r2,c2): return math.sqrt((r1-r2)**2+(c1-c2)**2)
    def decay_lr(self,rate): self.lr=max(0.001,self.lr*(1-rate))
    def init_random(self):
        with self._lock:
            for r in range(self.rows):
                for c in range(self.cols):
                    self.W[r][c]=VecReg([random.gauss(0.5,0.15) for _ in range(self.dim)])

class JIT:
    HOT=8
    def __init__(self): self.hits=defaultdict(int); self.cache={}
    def record(self,pc): self.hits[pc]+=1; return self.hits[pc]==self.HOT
    def cached(self,pc): return pc in self.cache
    def get(self,pc):    return self.cache.get(pc)
    def compile(self,pc,code):
        lines=['def _f(R,S,mem,som):']
        off=pc; n=0; OK={0x38,0x50,0x51,0x1F,0x40}
        while off+8<=len(code):
            raw=struct.unpack_from('>Q',code,off)[0]
            op=(raw>>56)&0xFF; src=(raw>>40)&0xFF; dst=(raw>>32)&0xFF; imm=raw&0xFFFFFFFF
            if op==0x38: lines.append('  pass')
            elif op==0x40 and not(imm&MEM_FLAG): lines.append(f'  R.setdefault({dst},VR()); R[{dst}].v[0]={imm/1000}')
            elif op==0x50: lines.append(f'  R.setdefault({dst},VR()); R.setdefault({src},VR()); R[{dst}].v=[a+b+{float(imm)} for a,b in zip(R[{dst}].v,R[{src}].v)]')
            elif op==0x51: lines.append(f'  R.setdefault({dst},VR()); R.setdefault({src},VR()); R[{dst}].v=[a-b-{float(imm)} for a,b in zip(R[{dst}].v,R[{src}].v)]')
            elif op==0x1F: lines.append(f'  som.decay_lr({imm/1000})')
            elif op not in OK: break
            else: break
            n+=1; off+=8
            if op in (0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37): break
        if n==0: return None,0
        lines.append(f'  return {n}')
        try:
            ns={'VR':VecReg}
            exec(compile('\n'.join(lines),'<jit>','exec'),ns)
            fn=ns['_f']; self.cache[pc]=(fn,n); return fn,n
        except: return None,0

class Shared:
    def __init__(self):
        self._a={}; self._l=threading.Lock(); self._al=threading.Lock()
        self._done={}; self.acc=0; self.inbox=[]
    def reg(self,ctx):
        with self._l: self._a[ctx.agent_id]=ctx; self._done[ctx.agent_id]=threading.Event()
    def get(self,aid):
        with self._l: return self._a.get(aid)
    def done(self,aid):
        with self._l: self._a.pop(aid,None)
        ev=self._done.get(aid)
        if ev: ev.set()
    def wait(self,aid,t=10): ev=self._done.get(aid); ev and ev.wait(timeout=t)
    def barrier(self,n,t=10):
        with self._l: ids=list(self._done.keys())
        for aid in ids:
            ev=self._done.get(aid); ev and ev.wait(timeout=t)
    def kill(self,aid):
        with self._l: ctx=self._a.get(aid); ctx and setattr(ctx,'running',False)
    def kill_all(self):
        with self._l:
            for ctx in self._a.values(): ctx.running=False
    def broadcast(self,val,exc=-1):
        with self._l: tgts=list(self._a.values())
        for ctx in tgts:
            if ctx.agent_id!=exc: ctx.enqueue(val)
    def accum(self,v):
        with self._al: self.acc=(self.acc+v)&0xFFFFFFFF

class Agent:
    def __init__(self,aid,entry,parent,shared,rt):
        self.agent_id=aid; self.pc=entry-HEADER_SIZE; self.parent_id=parent
        self.shared=shared; self.rt=rt
        self.R={}; self.S=[0.0]*16; self.call_stack=[]
        self.running=True; self.som_pos=(0,0); self.bmu_pos=(0,0)
        self._inbox=[]; self._il=threading.Lock(); self._ie=threading.Event()
    def rg(self,i): return self.R.get(i&0xFF,VecReg())
    def rs(self,i,v):
        if isinstance(v,VecReg): self.R[i&0xFF]=v
        else: w=VecReg(); w.v[0]=float(v); self.R[i&0xFF]=w
    def ri(self,i):
        v=self.rg(i).v[0]
        if v!=v or v==float("inf") or v==float("-inf"): return 0
        return int(max(-2147483648.0,min(2147483647.0,v)))
    def enqueue(self,v):
        with self._il: self._inbox.append(v)
        self._ie.set()
    def dequeue(self,t=3):
        self._ie.wait(timeout=t)
        with self._il:
            if self._inbox:
                v=self._inbox.pop(0)
                if not self._inbox: self._ie.clear()
                return v
        return 0
    def run(self):
        rt=self.rt
        try:
            while self.running and self.pc+8<=len(rt.code):
                if rt.jit.cached(self.pc):
                    fn,n=rt.jit.get(self.pc); fn(self.R,self.S,rt.mem,rt.som); self.pc+=n*8; continue
                if rt.jit.record(self.pc):
                    fn,n=rt.jit.compile(self.pc,rt.code)
                    if fn: fn(self.R,self.S,rt.mem,rt.som); self.pc+=n*8; continue
                raw=struct.unpack_from('>Q',rt.code,self.pc)[0]
                op=(raw>>56)&0xFF; ag=(raw>>48)&0xFF; src=(raw>>40)&0xFF; dst=(raw>>32)&0xFF; imm=raw&0xFFFFFFFF
                j=self._exec(op,ag,src,dst,imm)
                if self.running and not j: self.pc+=8
        except Exception as e:
            import traceback; traceback.print_exc()
        finally:
            self.shared.done(self.agent_id)

    def _exec(self,op,ag,src,dst,imm):
        j=False; rt=self.rt; sh=self.shared
        if   op==0x01: rt._spawn(ag,imm,self.agent_id)
        elif op==0x06:                          # SPAWN_MAP — real grid
            rows=(imm>>8)&0xFF; cols=imm&0xFF
            entry=imm  # label address in imm (assembler encodes it there)
            aid=0
            for r in range(max(rows,1)):
                for c in range(max(cols,1)):
                    ctx=rt._spawn(aid,entry,self.agent_id)
                    if ctx: ctx.som_pos=(r,c)
                    aid+=1
        elif op==0x07: sh.wait(ag)
        elif op==0x02:
            if imm==0xFF:  sh.kill_all()
            elif imm==0xFE: self.running=False; j=True
            else:           sh.kill(ag)
        elif op==0x03:
            for i in range(src): rt._spawn(i,imm,self.agent_id)
        elif op==0x04:
            v=VecReg(); v.v[0]=sh.acc/1000.0; self.R[dst]=v; sh.acc=0
        elif op==0x05: sh.barrier(imm)
        elif op==0x11:
            vec=self._lvec(src); br,bc=rt.som.bmu(vec); self.bmu_pos=(br,bc)
            v=VecReg(); v.v[0]=br; v.v[1]=bc; self.R[dst]=v
        elif op==0x12: rt.som.train(self._lvec(src),*self.bmu_pos)
        elif op==0x13:
            bv=self.rg(src); br=int(bv.v[0]); bc=int(bv.v[1])
            self.S[0]=rt.som.sense(br,bc)
        elif op==0x14: rt.som.train(self._lvec(src),*self.bmu_pos)
        elif op==0x19:
            br,bc=rt.som.elect(); v=VecReg(); v.v[0]=br; v.v[1]=bc; self.R[dst]=v
            print(f"  [SOM_ELECT] Leader: ({br},{bc})")
        elif op==0x1A: r=(imm>>8)&0xFF; c=imm&0xFF; self.som_pos=(r,c)
        elif op==0x1B:
            r,c=self.som_pos; val=rt.som.sense(r,c)
            v=VecReg(); v.v[0]=val; self.R[dst]=v
        elif op==0x1C:
            if imm==0x01: rt.som.init_random()  # PCA stub
            else:         rt.som.init_random()
        elif op==0x1D:                          # SOM_WALK — real gradient walk
            r,c=self.som_pos; nr,nc=rt.som.walk_gradient(r,c); self.som_pos=(nr,nc)
        elif op==0x1E:
            av=self.rg(src); bv=self.rg(dst)
            d=rt.som.node_dist(int(av.v[0]),int(av.v[1]),int(bv.v[0]),int(bv.v[1]))
            v=VecReg(); v.v[0]=d; self.R[dst]=v
        elif op==0x1F: rt.som.decay_lr(imm/1000.0)
        elif op==0x20:
            vec=self.rg(src) if src else None; val=vec.as_int() if vec else imm
            if ag==0xFF:
                p=sh.get(self.parent_id)
                if p: p.enqueue(val)
                else: sh.inbox.append(val)
            elif ag==0xFE: self.enqueue(val)
            else:
                t=sh.get(ag)
                if t: t.enqueue(val)
                else: sh.inbox.append(val)
        elif op==0x21:
            val=self.dequeue(); v=VecReg(); v.v[0]=val/1000.0; self.R[dst]=v
        elif op==0x23: sh.broadcast(imm,self.agent_id)
        elif op==0x24:
            sv=self.ri(src); dv=self.ri(dst); res=(sv+dv)&0xFFFFFFFF
            self.rs(dst,res); sh.accum(sv)
        elif op==0x30:
            tp=imm-HEADER_SIZE
            if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
            else: self.running=False; j=True
        elif op==0x31:
            if self.ri(src)==0:
                tp=imm-HEADER_SIZE
                if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
        elif op==0x32:
            if self.ri(src)!=0:
                tp=imm-HEADER_SIZE
                if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
        elif op==0x33:
            if self.ri(src)==self.ri(dst):
                tp=imm-HEADER_SIZE
                if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
        elif op==0x34:
            if self.ri(src)>self.ri(dst):
                tp=imm-HEADER_SIZE
                if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
        elif op==0x35:
            if len(self.call_stack)>=MAX_CALL: self.running=False; j=True
            else:
                self.call_stack.append(self.pc+8); tp=imm-HEADER_SIZE
                if 0<=tp<=len(rt.code)-8: self.pc=tp; j=True
        elif op==0x36:
            if self.call_stack: self.pc=self.call_stack.pop(); j=True
            else: self.running=False; j=True
        elif op==0x37: self.running=False; j=True
        elif op==0x38: pass
        elif op==0x40:
            if src:      self.R[dst]=self.rg(src)
            elif imm&MEM_FLAG:
                poff=imm&~MEM_FLAG; sym=rt.mem.find_by_offset(poff)
                cnt=sym['count'] if sym else VEC_DIM
                self.R[dst]=rt.mem.read_vec(poff,cnt)
            else: self.rs(dst,imm)
        elif op==0x41:
            vec=self.rg(src)
            if imm&MEM_FLAG: rt.mem.write_vec(imm&~MEM_FLAG,vec)
        elif op==0x42: self.R[dst]=rt.mem.read_vec(imm&~MEM_FLAG)
        elif op==0x43: self._trap(imm)
        elif op==0x50:
            sv=self.rg(src)
            if imm: self.rs(dst, (int(sv.v[0])+imm)&0xFFFFFFFF)
            else:
                dv=self.rg(dst)
                self.R[dst]=VecReg([a+b for a,b in zip(sv.v,dv.v)])
        elif op==0x51:
            dv=self.rg(dst); sv=self.rg(src)
            self.R[dst]=VecReg([a-b-float(imm) for a,b in zip(dv.v,sv.v)])
        elif op==0x52:
            dv=self.rg(dst); sv=self.rg(src)
            self.R[dst]=VecReg([a*b for a,b in zip(dv.v,sv.v)])
        elif op==0x54:
            d=self.rg(dst).dot(self.rg(src)); v=VecReg(); v.v[0]=d; self.R[dst]=v
        elif op==0x55: self.R[dst]=self.rg(dst).normalized()
        return j

    def _lvec(self,reg):
        v=self.rg(reg)
        if v.as_int()&MEM_FLAG:
            poff=v.as_int()&~MEM_FLAG; sym=self.rt.mem.find_by_offset(poff)
            return self.rt.mem.read_vec(poff,sym['count'] if sym else VEC_DIM)
        return v

    def _trap(self,code):
        if   code==0x01: self.rs(0,0)
        elif code==0x02: pass
        elif code==0x20: self.rs(0,0)
        elif code==0x30: self.rs(0,0)
        elif code==0xFF: self.running=False

class SomaRuntime:
    OPNAME={
        0x01:'SPAWN',0x02:'AGENT_KILL',0x03:'FORK',0x04:'MERGE',0x05:'BARRIER',
        0x06:'SPAWN_MAP',0x07:'WAIT',0x11:'SOM_BMU',0x12:'SOM_TRAIN',
        0x13:'SOM_NBHD',0x14:'WGHT_UPD',0x19:'SOM_ELECT',0x1A:'SOM_MAP',
        0x1B:'SOM_SENSE',0x1C:'SOM_INIT',0x1D:'SOM_WALK',0x1E:'SOM_DIST',
        0x1F:'LR_DECAY',0x20:'MSG_SEND',0x21:'MSG_RECV',0x23:'BROADCAST',
        0x24:'ACCUM',0x30:'JMP',0x31:'JZ',0x32:'JNZ',0x33:'JEQ',0x34:'JGT',
        0x35:'CALL',0x36:'RET',0x37:'HALT',0x38:'NOP',
        0x40:'MOV',0x41:'STORE',0x42:'LOAD',0x43:'TRAP',
        0x50:'ADD',0x51:'SUB',0x52:'MUL',0x53:'DIV',0x54:'DOT',0x55:'NORM',
    }
    def __init__(self,path,verbose=True):
        self.verbose=verbose
        data=open(path,'rb').read()
        if data[:4]!=b'SOMA': print("ERROR: bad magic",file=sys.stderr); sys.exit(1)
        self.version=struct.unpack_from('>I',data,4)[0]
        self.som_rows=data[9] or 16; self.som_cols=data[10] or 16; self.max_agents=data[11] or 64
        co=struct.unpack_from('>I',data,12)[0]; cs=struct.unpack_from('>I',data,16)[0]
        do=struct.unpack_from('>I',data,20)[0]; ds=struct.unpack_from('>I',data,24)[0]
        if not cs or co+cs>len(data): cs=max(0,len(data)-co-ds)
        self.code=data[co:co+cs]
        rem=len(self.code)%8
        if rem: self.code+=bytes(8-rem)
        self.data_bytes=data[do:do+ds] if ds and do+ds<=len(data) else b''
        self.mem=DataMemory(self.data_bytes)
        self.som=SOM(self.som_rows,self.som_cols)
        self.shared=Shared(); self.jit=JIT(); self._threads=[]
        if verbose:
            print(f"[SOMA] {path}  SOM {self.som_rows}×{self.som_cols}  "
                  f"{len(self.code)//8} instructions")
            if self.mem.symbols: print("  [DATA]"); self.mem.dump()
            print()

    def _spawn(self,aid,entry,parent):
        ctx=Agent(aid,entry,parent,self.shared,self)
        self.shared.reg(ctx)
        if self.verbose: print(f"  → A{aid} entry=0x{entry:08x}")
        t=threading.Thread(target=ctx.run,daemon=True,name=f"soma-{aid}")
        self._threads.append(t); t.start(); return ctx

    def run(self):
        ctx=Agent(0xFF,HEADER_SIZE,-1,self.shared,self); ctx.pc=0
        while ctx.running and ctx.pc+8<=len(self.code):
            if self.jit.cached(ctx.pc):
                fn,n=self.jit.get(ctx.pc); fn(ctx.R,ctx.S,self.mem,self.som); ctx.pc+=n*8; continue
            if self.jit.record(ctx.pc):
                fn,n=self.jit.compile(ctx.pc,self.code)
                if fn: fn(ctx.R,ctx.S,self.mem,self.som); ctx.pc+=n*8; continue
            raw=struct.unpack_from('>Q',self.code,ctx.pc)[0]
            op=(raw>>56)&0xFF; ag=(raw>>48)&0xFF; src=(raw>>40)&0xFF; dst=(raw>>32)&0xFF; imm=raw&0xFFFFFFFF
            if self.verbose:
                nm=self.OPNAME.get(op,f'0x{op:02x}')
                mem='→[MEM]' if imm&MEM_FLAG else ''
                print(f"  {nm:12s} ag={ag} R{src}→R{dst} imm=0x{imm&0x7FFFFFFF:08x}{mem}")
            j=ctx._exec(op,ag,src,dst,imm)
            if ctx.running and not j: ctx.pc+=8
        for t in self._threads: t.join(timeout=12)
        if self.verbose:
            print(); print("[SOMA] Done.")
            used={k:v for k,v in ctx.R.items() if any(x!=0 for x in v.v)}
            if used:
                print("[SOMA] Registers:")
                for k in sorted(used): print(f"  R{k} = {used[k]}")
            if self.mem.symbols:
                print("[SOMA] Final data:"); self.mem.dump()
            jn=sum(1 for v in self.jit.hits.values() if v>=JIT.HOT)
            if jn: print(f"[SOMA] JIT: {jn} hot block(s).")

if __name__=='__main__':
    verbose='--quiet' not in sys.argv
    args=[a for a in sys.argv[1:] if not a.startswith('--')]
    if not args: print("Usage: soma_runtime.py <file.sombin> [--quiet]",file=sys.stderr); sys.exit(1)
    try: SomaRuntime(args[0],verbose).run()
    except KeyboardInterrupt: print("\n[SOMA] Interrupted.")
    except Exception: import traceback; traceback.print_exc(); sys.exit(1)

    

# ── CLI entrypoint for `pip install` ─────────────────────────────────────
def run_cli():
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: soma <binary.sombin> [--quiet]")
        print("       (full CLI coming in v3.1)")
        sys.exit(0)

    binary_path = sys.argv[1]
    quiet = "--quiet" in sys.argv[2:]

    # Call your existing interpreter (just paste your current top-level code here
    # or extract it into a run_file(binary_path, quiet) function)
    print(f"🚀 SOMA v3.0 running {binary_path} (quiet={quiet})")
    # ... your runtime code ...

if __name__ == "__main__":
    run_cli()