"""Local-only demo with a persistent worker and an active-time clock."""
import io, os, threading, time, traceback
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .environment import Environment
from .policy import Learner
from .shadows import device_name, sync

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(max(1,int(os.getenv('VOXELFORM_THREADS','1'))))

class Session:
    def __init__(self):
        self.lock=threading.RLock();self.running=False;self.single=False;self.training=True
        self.active=0.;self.started=None;self.episodes=0;self.steps=0;self.revision=0
        self.ready=False;self.error=None;self.pending_seed=123;self.plan='rectangle';self.recent=[]
        self.stats=dict(updates=0,loss=0.,gradientNorm=0.);self.max_steps=16
        # CPU is the measured low-latency default on this Mac for small grids.
        # Explicit mps/cuda remains available for scale comparisons.
        self.device=device_name(os.getenv('VOXELFORM_DEVICE','cpu'));self.env=None
        self.worker=threading.Thread(target=self.loop,daemon=True);self.worker.start()
    def clock(self):return self.active+(time.perf_counter()-self.started if self.started is not None else 0.)
    def pause(self):
        if self.started is not None:self.active+=time.perf_counter()-self.started;self.started=None
        self.running=False
    def build(self):
        self.env=Environment(shape=(6,24,32),seed=self.pending_seed,device=self.device,plan=self.plan)
        self.policy=Learner(self.device);self.ready=True;self.revision+=1
    def next_episode(self):
        self.env.reset();self.steps=0;self.policy.clear();self.revision+=1
    def loop(self):
        try:
            with self.lock:self.build()
            while True:
                with self.lock:
                    if self.running or self.single:
                        if self.steps>=self.max_steps:self.next_episode()
                        stamp=time.perf_counter();a,masks=self.policy.choose(self.env,training=self.training,greedy=not self.training)
                        sync(self.device);self.action_seconds=time.perf_counter()-stamp;q=self.env.quality.detach().clone()
                        done=self.env.apply(a,masks);self.steps+=1
                        if self.training:self.policy.record(self.env.quality-q-(0. if done else .002))
                        if done or self.steps>=self.max_steps:
                            stamp=time.perf_counter()
                            if self.training:self.stats=self.policy.finish()
                            sync(self.device);self.learning_seconds=time.perf_counter()-stamp;self.episodes+=1
                            self.recent.append(dict(episode=self.episodes,**self.env.metrics()));self.recent=self.recent[-100:];self.steps=self.max_steps
                        self.revision+=1;self.single=False
                time.sleep(.01)
        except Exception:
            with self.lock:self.error=traceback.format_exc();self.pause()
    def state(self):
        with self.lock:
            if not self.ready:return dict(ready=False,error=self.error,device=self.device)
            e=self.env;result=dict(ready=True,error=self.error,running=self.running,training=self.training,
                activeSeconds=self.clock(),episodes=self.episodes,step=self.steps,maxSteps=self.max_steps,
                seed=self.pending_seed,plan=self.plan,revision=self.revision,metrics=e.metrics(),learning=self.stats,
                actionSeconds=getattr(self,'action_seconds',0.),learningSeconds=getattr(self,'learning_seconds',0.),
                shape=list(e.shape),occupied=e.occupied.to('cpu',torch.uint8).flatten().tolist(),
                allowed=e.allowed.to('cpu',torch.uint8).flatten().tolist(),
                light=e.light.detach().cpu().flatten().tolist(),noise=e.noise.detach().cpu().flatten().tolist(),
                usable=e.usable.to('cpu',torch.uint8).flatten().tolist(),boxes=e.boxes.tolist(),recent=self.recent)
            return result

session=Session();app=FastAPI(title='VoxelForm RL')
app.mount('/assets',StaticFiles(directory=ROOT/'public'),name='assets')
@app.get('/')
def home():return FileResponse(ROOT/'public/index.html')
@app.get('/api/state')
def state():return session.state()
@app.post('/api/control')
async def control(request:Request):
    data=await request.json();command=data.get('command')
    with session.lock:
        if not session.ready:raise HTTPException(409,'Environment is still preparing')
        if command=='run':
            new_training=bool(data.get('training',True))
            if new_training!=session.training:session.policy.clear()
            session.training=new_training;session.running=True
            if session.started is None:session.started=time.perf_counter()
        elif command=='pause':session.pause()
        elif command=='step':
            session.pause();session.single=True;new_training=bool(data.get('training',False))
            if new_training!=session.training:session.policy.clear()
            session.training=new_training
        elif command=='reset':
            session.pause();session.next_episode()
        elif command=='site':
            session.pause();session.pending_seed+=1;plan=data.get('plan','rectangle')
            if plan not in ('rectangle','triangle','courtyard','branch'):raise HTTPException(400,'Unknown synthetic plot')
            session.plan=plan;session.env=Environment(seed=session.pending_seed,device=session.device,plan=plan)
            session.steps=0;session.policy.clear();session.revision+=1
        else:raise HTTPException(400,'Unknown command')
    return dict(ok=True)
@app.get('/api/policy')
def save():
    with session.lock:
        if not session.ready:raise HTTPException(409,'Not ready')
        buffer=io.BytesIO();torch.save(session.policy.state(),buffer)
    return Response(buffer.getvalue(),media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="voxelform-policy.pt"'})
@app.post('/api/policy')
async def load(request:Request):
    raw=await request.body()
    if len(raw)>20_000_000:raise HTTPException(413,'Checkpoint too large')
    with session.lock:
        session.pause()
        try:
            session.policy.load(torch.load(io.BytesIO(raw),map_location=session.device,weights_only=True));session.steps=0
            session.stats=dict(updates=session.policy.updates,loss=0.,gradientNorm=0.);session.revision+=1
        except Exception as exc:raise HTTPException(400,str(exc)) from exc
    return dict(ok=True)
