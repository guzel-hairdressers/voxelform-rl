"""Surface-anchored volume edits and explicit proxy evaluation."""
import hashlib,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from .shadows import DirectShadows,BoundaryShadows,DepthShadows,SelfShadowMaps,FastVoxelShadows,ReceiverCache,MeshShadows,sun_path,sync
from .actions import ExteriorFacades
from .workload import reset_budget
from .fields import grid_points,noise_prior,sky_boundary,diffuse_access,facade_sky_access,receivers,usable_depth_proxy,facade_depth,depth_badness
from .cores import solve_cores,DEFAULT_MIN_CORE_SPACING,DEFAULT_MAX_CORE_DIST


def _scene_cache_path(cache_dir,context,shape,sun_count,angle,allowed_np,suns_np,pitch=1.,version='scene_cache_v2'):
    h=hashlib.sha256()
    for part in (str(version),context.get('site_id',''),repr(tuple(shape)),context.get('gridAlignment','world'),f'{float(angle):.9f}',str(int(sun_count)),f'{float(pitch):.6f}'):
        h.update(part.encode('utf-8'))
    h.update(np.ascontiguousarray(allowed_np,dtype=np.bool_).tobytes())
    h.update(np.ascontiguousarray(suns_np,dtype=np.float32).tobytes())
    for b in context.get('buildings',()):
        h.update(np.ascontiguousarray(b['vertices'],dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(b['faces'],dtype=np.int32).tobytes())
    return Path(cache_dir)/f"{context.get('site_id','site')}_{h.hexdigest()[:16]}.pt"


def context_boxes(nx,ny,count=16,seed=123,height=24.):
    rng=random.Random(seed);rows=[]
    for k in range(count):
        side=k%4;hx=rng.uniform(4,9);hy=rng.uniform(4,9);yaw=rng.uniform(-.6,.6);radius=math.hypot(hx,hy)+rng.uniform(3,14)
        x,y=(rng.uniform(-10,nx+10),-radius) if side==0 else (nx+radius,rng.uniform(-10,ny+10)) if side==1 else (rng.uniform(-10,nx+10),ny+radius) if side==2 else (-radius,rng.uniform(-10,ny+10))
        h=rng.uniform(7,max(8,height*1.8));rows.append((x,y,h/2,hx,hy,h/2,math.cos(yaw),math.sin(yaw)))
    return np.asarray(rows,dtype=np.float32)


class Environment:
    def __init__(self,shape=(6,24,32),seed=123,device='cpu',shadow_method='boundary',sun_count=8,pitch=1.,plan='rectangle',context=None,target_far=3.,depth_weight=0.,frontier_factory=ExteriorFacades,far_tolerance=0.,scene_cache_dir=None,fast_self_shadows=False,fast_diffuse=False,cores_enabled=False,min_core_spacing=DEFAULT_MIN_CORE_SPACING,max_core_dist=DEFAULT_MAX_CORE_DIST):
        if context is not None:shape=context['shape']
        self.shape=tuple(shape);self.device=torch.device(device);self.seed=seed;self.pitch=pitch;self.sun_count=sun_count
        self.nz,self.ny,self.nx=self.shape;self.plan=plan;self.context=context
        self.frontier_factory=frontier_factory
        self.fast_self_shadows=bool(fast_self_shadows)
        self.fast_diffuse=bool(fast_diffuse)
        self.cores_enabled=bool(cores_enabled)
        self.min_core_spacing=max(0,int(min_core_spacing))
        self.max_core_dist=max(1,int(max_core_dist))
        self._reset_core_fields()
        self.depth_weight=float(depth_weight)
        if not math.isfinite(self.depth_weight) or self.depth_weight<0:raise ValueError("depth_weight must be nonnegative and finite")
        ctx_metrics=(context or {}).get('metrics') or {}
        self.context_avg_height=float((context or {}).get('avgHeight') or ctx_metrics.get('avgHeight') or 18.)
        self.context_max_height=float((context or {}).get('maxHeight') or ctx_metrics.get('maxHeight') or float(self.nz*3))
        self.points=grid_points(shape,self.device);self.x=self.points[...,0];self.y=self.points[...,1];self.z=self.points[...,2]/3
        self.rel_z=torch.tanh(self.points[...,2]/36.)
        allowed=torch.ones(shape,dtype=torch.bool,device=self.device)
        if plan=='triangle':allowed&=self.x/self.nx+self.y/self.ny<1.
        elif plan=='courtyard':allowed&=~((self.x>.65*self.nx)&(self.y>.65*self.ny))
        elif plan=='branch':allowed&=(self.y<self.ny*.55)|((self.x>.35*self.nx)&(self.x<self.nx*.65))
        if context is not None:
            allowed=torch.as_tensor(context['allowed'],device=self.device)[None].expand(self.shape).clone()
        self.allowed=allowed;self.total_area=allowed.sum()
        self.capacity=int(self.total_area.detach().cpu())
        self.plot_area=float(context["siteArea"]) if context is not None else float(allowed[0].sum().detach().cpu())
        self.far_tolerance=self.validate_tolerance(far_tolerance)
        self.target_far,self.target_count=self.validate_far(target_far)
        self.boxes=context_boxes(self.nx,self.ny,16,seed,self.nz*3) if context is None else np.empty((0,8),np.float32)
        self.suns_np=sun_path(sun_count)
        angle=float((context or {}).get('gridAngle',0.))
        if angle:
            c,s=math.cos(angle),math.sin(angle);self.suns_np[:,:2]=self.suns_np[:,:2]@np.array([[c,-s],[s,c]])
        self.suns=torch.tensor(self.suns_np,dtype=torch.float32,device=self.device)
        self.bounds=((0,0,0),(self.nx,self.ny,self.nz*3))
        cls={'boundary':BoundaryShadows,'depth':DepthShadows,'direct':DirectShadows}[shadow_method]
        if context is not None:
            self.static=MeshShadows(context['buildings'],self.suns_np,self.device)
        else:
            self.static=cls(self.boxes,self.suns_np,self.bounds,pitch,self.device) if shadow_method!='direct' else cls(self.boxes,self.suns_np,self.device)
        self.static_receivers=ReceiverCache(self.static,6*int(np.prod(shape)))
        self._scene_cache_path=None;self._scene_cache_hit=False;self._precomputed_hit=False
        self._cache_version='scene_cache_v3_fast' if self.fast_self_shadows else 'scene_cache_v2'
        if context is not None and scene_cache_dir is not None:
            cache_path=_scene_cache_path(scene_cache_dir,context,self.shape,self.sun_count,angle,context['allowed'],self.suns_np,pitch,version=self._cache_version)
            self._scene_cache_path=cache_path
            if cache_path.is_file():
                try:
                    cached=torch.load(cache_path,map_location='cpu',weights_only=True)
                    base_ok=(isinstance(cached,dict) and cached.get('version')==self._cache_version
                        and tuple(cached['shape'])==self.shape and int(cached['sun_count'])==self.sun_count
                        and tuple(cached['sky_boundary'].shape)==tuple(n+2 for n in self.shape)
                        and cached['known'].shape==self.static_receivers.known.shape
                        and cached['values'].shape==self.static_receivers.values.shape)
                    if base_ok and self.fast_self_shadows:
                        self.sky_boundary=cached['sky_boundary'].to(self.device)
                        self.static_receivers.known=cached['known'].to(self.device)
                        self.static_receivers.values=cached['values'].to(self.device)
                        self.dynamic=FastVoxelShadows(shape,self.suns_np,device=self.device)
                        self._scene_cache_hit=True
                        self._precomputed_hit=bool(cached.get('precomputed',True))
                    elif base_ok and isinstance(cached.get('dynamic_maps'),list) and len(cached['dynamic_maps'])==self.sun_count:
                        self.sky_boundary=cached['sky_boundary'].to(self.device)
                        self.static_receivers.known=cached['known'].to(self.device)
                        self.static_receivers.values=cached['values'].to(self.device)
                        dyn=SelfShadowMaps.__new__(SelfShadowMaps)
                        dyn.suns=self.suns
                        dyn.anchor=cached['dynamic_anchor'].to(self.device)
                        dyn.maps=[
                            dict(
                                parameters=(m['u'].to(self.device),m['v'].to(self.device),m['low'].to(self.device),float(m['pitch']),m['sizes'].cpu().numpy(),None),
                                ids=m['ids'].to(self.device),
                                entry=m['entry'].to(self.device),
                                image=None,
                            )
                            for m in cached['dynamic_maps']
                        ]
                        self.dynamic=dyn
                        self._scene_cache_hit=True
                        self._precomputed_hit=bool(cached.get('precomputed',True))
                except Exception:
                    self._scene_cache_hit=False;self._precomputed_hit=False
        if not self._scene_cache_hit:
            self.dynamic=FastVoxelShadows(shape,self.suns_np,device=self.device) if self.fast_self_shadows else SelfShadowMaps(shape,self.suns_np,pitch,device=self.device)
            self.sky_boundary=sky_boundary(shape,self.boxes,self.device,buildings=None if context is None else context['buildings'],grid_angle=angle)
        road=((-15.,-8.,0.),(self.nx+15.,-8.,0.),1.)
        rail=((-11.,-15.,0.),(-11.,self.ny+15.,0.),.6)
        self.noise_raw=noise_prior(self.points,[road,rail] if context is None else []);self.noise=self.noise_raw/(self.noise_raw+.12)
        self.history=[];self.reset()

    def validate_far(self,far):
        if isinstance(far,bool):raise ValueError('Target FAR must be a positive number')
        far=float(far)
        if not math.isfinite(far) or far<=0:raise ValueError('Target FAR must be a positive finite number')
        target=int(math.floor(far*self.plot_area+.5))
        if target<1 or target>self.capacity or far>self.capacity/self.plot_area:
            raise ValueError(f'Target FAR must fit this envelope (maximum {self.capacity/self.plot_area:.4f})')
        return far,target

    def reset(self):
        self.restore_occupancy(self.allowed)
        return self.observation()

    @staticmethod
    def validate_tolerance(value):
        if isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<0:raise ValueError('FAR tolerance must be finite and nonnegative')
        return float(value)

    @property
    def acceptable_counts(self):
        if not self.far_tolerance:return self.target_count,self.target_count
        low=max(1,math.ceil((self.target_far-self.far_tolerance)*self.plot_area-1e-9))
        high=min(self.capacity,math.floor((self.target_far+self.far_tolerance)*self.plot_area+1e-9))
        return (low,high) if low<=high else (self.target_count,self.target_count)
    @property
    def goal_count(self):
        low,high=self.acceptable_counts
        return min(high,max(low,self.count))
    @property
    def far_satisfied(self):return self.count==self.goal_count

    def restore_occupancy(self,occupied):
        raw=torch.as_tensor(occupied,dtype=torch.bool,device=self.device).reshape(self.shape)
        if bool((raw&~self.allowed).any()) or not bool(raw.any()):raise ValueError('Occupied voxels must lie within the permitted envelope')
        self.occupied=raw.clone();self.count=int(raw.sum().detach().cpu());self.edits=0;self.voxels_edited=0;self.last_changed_voxels=0;self.last_operation=None
        self.frontier=self.frontier_factory(raw.detach().cpu().numpy(),self.allowed.detach().cpu().numpy(),self.seed)
        self.update_depth();self.start_count=self.count;self.termination='far_reached' if self.far_satisfied else None
        reset_budget(self)
        self.evaluate()

    def configure_far(self,far,tolerance=None):
        target,count=self.validate_far(far)
        tolerance=self.far_tolerance if tolerance is None else self.validate_tolerance(tolerance)
        fields_current=self.last_audit_edit==self.edits
        self.target_far=target;self.target_count=count;self.far_tolerance=tolerance;self.start_count=self.count;self.edits=0;self.voxels_edited=0;self.last_changed_voxels=0;self.last_operation=None
        self.termination='far_reached' if self.far_satisfied else None
        reset_budget(self)
        self.depth_edit=0  # Retargeting changes neither occupancy nor depth.
        if fields_current:self.last_audit_edit=0
        else:self.evaluate()

    @property
    def far(self):return self.count/self.plot_area
    @property
    def far_error(self):return abs(self.count-self.goal_count)/self.plot_area
    @property
    def terminal(self):return self.termination is not None
    @property
    def goal_features(self):
        maximum=self.capacity/self.plot_area
        return [self.far/maximum,self.target_far/maximum,self.far_error/maximum]

    def update_depth(self):
        stamp=time.perf_counter()
        tensor=getattr(self.frontier,'depth_tensor',None)
        if tensor is not None and tensor.device.type==self.device.type and (self.device.index is None or tensor.device.index==self.device.index):self.depth=tensor
        else:
            distance=getattr(self.frontier,'depth_numpy',None)
            if distance is None:distance=facade_depth(self.frontier.occupied,self.frontier.exterior_air)
            self.depth=torch.as_tensor(distance,device=self.device)
        self.depth_badness=depth_badness(self.depth)
        self.depth_edit=self.edits;sync(self.device);self.depth_seconds=time.perf_counter()-stamp

    def precompute_external(self,chunk=8192):
        """Exactly cache fixed surrounding visibility for every possible receiver.

        Optional amortized preparation for repeated training on this scene. Self
        shadows, sky diffusion and occupied receivers still update after edits.
        Cache validity is scoped to this environment's immutable scene/sun setup.
        """
        if isinstance(chunk,bool) or int(chunk)!=chunk or chunk<1:raise ValueError('chunk must be a positive integer')
        stamp=time.perf_counter()
        if self._scene_cache_hit and self._precomputed_hit:
            return dict(seconds=time.perf_counter()-stamp,newReceivers=0,knownReceivers=int(self.static_receivers.known.sum().cpu()),
                cacheBytes=self.static_receivers.known.numel()+self.static_receivers.values.numel())
        cells=self.allowed.flatten().nonzero().flatten();new=0
        normals=self.points.new_tensor(((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)))
        spacing=self.points.new_tensor((1.,1.,3.));volume=self.occupied.numel()
        if isinstance(self.static,MeshShadows) and self.static._all_suns_up:
            slice_area=self.nx*self.ny
            max_z=self.static.max_z-1e-4
            for face in range(6):
                nz_off=normals[face][2].item()*1.5
                for z in range(0 if face!=5 else 1,self.nz):
                    z_ids=self.allowed[z].flatten().nonzero().flatten()
                    if not len(z_ids):continue
                    r_z=z*slice_area+z_ids
                    pz=(z+.5)*3.+nz_off
                    if face!=5 and pz>=max_z:
                        r_keys=face*volume+r_z
                        fresh_mask=~self.static_receivers.known[r_keys]
                        if bool(fresh_mask.any()):
                            fk=r_keys[fresh_mask]
                            self.static_receivers.values[:,fk]=True
                            self.static_receivers.known[fk]=True
                            new+=int(fk.numel())
                    else:
                        for begin in range(0,len(r_z),int(chunk)):
                            ids=r_z[begin:begin+int(chunk)]
                            xyz=torch.stack((ids%self.nx+.5,ids//self.nx%self.ny+.5,ids//slice_area+.5),dim=1)
                            points=xyz*spacing+normals[face]*spacing/2
                            _,fresh=self.static_receivers.query(face*volume+ids,points);new+=fresh
        else:
            for face in range(6):
                eligible=cells if face!=5 else cells[cells>=self.nx*self.ny]
                for begin in range(0,len(eligible),int(chunk)):
                    ids=eligible[begin:begin+int(chunk)]
                    xyz=torch.stack((ids%self.nx+.5,ids//self.nx%self.ny+.5,ids//(self.nx*self.ny)+.5),dim=1)
                    points=xyz*spacing+normals[face]*spacing/2
                    _,fresh=self.static_receivers.query(face*volume+ids,points);new+=fresh
        sync(self.device)
        if self._scene_cache_path is not None and not self._scene_cache_hit:
            self._scene_cache_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=self._scene_cache_path.with_name(f'{self._scene_cache_path.name}.{os.getpid()}.tmp')
            payload=dict(version=self._cache_version,shape=self.shape,sun_count=self.sun_count,precomputed=True,
                sky_boundary=self.sky_boundary.detach().cpu(),
                known=self.static_receivers.known.detach().cpu(),
                values=self.static_receivers.values.detach().cpu())
            if not self.fast_self_shadows:
                payload['dynamic_anchor']=self.dynamic.anchor.detach().cpu()
                payload['dynamic_maps']=[
                    dict(
                        u=m['parameters'][0].detach().cpu(),
                        v=m['parameters'][1].detach().cpu(),
                        low=m['parameters'][2].detach().cpu(),
                        pitch=float(m['parameters'][3]),
                        sizes=torch.as_tensor(m['parameters'][4],dtype=torch.long).cpu(),
                        ids=m['ids'].detach().cpu(),
                        entry=m['entry'].detach().cpu(),
                    )
                    for m in self.dynamic.maps
                ]
            torch.save(payload,tmp)
            os.replace(tmp,self._scene_cache_path)
            self._scene_cache_hit=True;self._precomputed_hit=True
        return dict(seconds=time.perf_counter()-stamp,newReceivers=new,knownReceivers=int(self.static_receivers.known.sum().cpu()),
            cacheBytes=self.static_receivers.known.numel()+self.static_receivers.values.numel())

    def daylight_proxies(self, core_mask, lateral=None):
        """Score occupied faces outside the reserved cores using audited fields."""
        if lateral is None:
            lateral = torch.as_tensor(self.frontier.remove_faces[[1, 0, 3, 2]], device=self.device)
        _, _, normals, cells, faces = self.receiver_data
        areas = torch.where(faces < 4, 3., 1.)
        weights = (self.suns @ normals.T).clamp_min(0) * areas[None]
        if core_mask is None:
            diffuse_faces = lateral
        else:
            weights = weights * (~core_mask.flatten()[cells])[None]
            diffuse_faces = lateral & ~core_mask[None]
        direct = (self.visible * weights).sum() / weights.sum().clamp_min(1)
        diffuse = (self.facade_light_faces * diffuse_faces).sum() / diffuse_faces.sum().clamp_min(1)
        return direct, diffuse

    def evaluate(self):
        stamp=time.perf_counter()
        if self.fast_diffuse and self.edits>0 and getattr(self,'last_audit_edit',-1)!=self.edits and getattr(self,'light',None) is not None and self.light.shape==self.occupied.shape:
            z_occ=self.occupied.any(dim=(1,2)).nonzero(as_tuple=False)
            z_top=int(z_occ[-1].item())+1 if len(z_occ) else 1
            z_hi=min(self.nz,z_top+4)
            if z_hi<self.nz:
                bnd=self.sky_boundary[:z_hi+2].clone()
                bnd[-1,1:-1,1:-1]=self.light[z_hi]
                sub=diffuse_access(self.occupied[:z_hi],bnd,iterations=20,init=self.light[:z_hi])
                light_new=self.light.clone();light_new[:z_hi]=sub;self.light=light_new
            else:
                self.light=diffuse_access(self.occupied,self.sky_boundary,iterations=20,init=self.light)
        else:
            self.light=diffuse_access(self.occupied,self.sky_boundary,iterations=48)
        lateral=torch.as_tensor(self.frontier.remove_faces[[1,0,3,2]],device=self.device)
        self.facade_light_faces,self.facade_light=facade_sky_access(self.occupied,self.light,self.sky_boundary,lateral)
        self.usable=usable_depth_proxy(self.occupied,self.light,boundary=self.sky_boundary)
        ids,p,n,cells,faces=receivers(self.occupied)
        outside,new=self.static_receivers.query(ids,p);self.dynamic.update(self.occupied)
        if isinstance(self.dynamic,FastVoxelShadows):
            own=self.dynamic.query(p,normals=n,active_mask=outside,faces=faces)
        else:
            own=self.dynamic.query(p,normals=n)
        front=(self.suns@n.T)>1e-6
        self.visible=outside&own&front;self.receiver_data=(ids,p,n,cells,faces)
        self.update_cores()
        core_mask = self.core_mask if self.cores_enabled else None
        sunlight, diffuse = self.daylight_proxies(core_mask, lateral)
        side=faces<4;roof=faces==4
        roof_sun=self.visible[:,roof].float().mean(0) if bool(roof.any()) else torch.zeros(0,device=self.device)
        roof_opportunity=(roof_sun>.4).float().sum()/(self.nx*self.ny)
        retention=self.occupied.sum()/self.total_area;usable=self.usable.sum()/self.total_area
        exposure=self.noise.flatten()[cells[side]].mean() if bool(side.any()) else torch.zeros((),device=self.device)
        quality=.70*usable+.18*sunlight+.12*roof_opportunity-.08*exposure
        quality=quality-self.depth_weight*self.depth_badness.sum()/self.capacity
        self.quality=quality;self.values=dict(quality=quality,retention=retention,potentialUsableFraction=usable,
            directSunProxy=sunlight,roofOpportunity=roof_opportunity,noisePrior=exposure,
            diffuseFacadeProxy=diffuse,
            occupiedFloorArea=self.occupied.sum(),potentialUsableArea=self.usable.sum())
        sync(self.device)
        self.last_audit_edit=self.edits;self.last_new_receivers=new;self.last_receiver_count=len(ids);self.last_eval_seconds=time.perf_counter()-stamp
        return quality

    def observation(self,include_depth=False,z_slice=None):
        # All channels stay on the chosen device; no per-voxel Python work.
        h_feat=self.rel_z if getattr(self,'organic_3d_growth',False) else (self.z/self.nz)
        channels=(self.occupied.float(),self.light,self.noise,self.allowed.float(),h_feat)
        if include_depth:channels=(*channels,self.depth_badness)
        if z_slice is not None:
            channels=tuple(c[:z_slice] for c in channels)
        return torch.stack(channels,dim=1)

    def apply(self,action,audit=True):
        if self.terminal:raise ValueError('Episode has already finished')
        expected=-1 if self.count>self.target_count else 1
        if action.operation!=expected:raise ValueError('Neural mask must move toward the FAR target')
        if action.count>abs(self.count-self.goal_count):raise ValueError('Neural mask would overshoot the FAR acceptance boundary')
        changed=self.frontier.apply(action)
        self.occupied.masked_fill_(action.tensor(self.device),expected==1)
        self.count+=expected*changed;self.edits+=1;self.voxels_edited+=changed;self.last_changed_voxels=changed
        self.last_generated_voxels=action.generated_voxels
        self.last_operation='subtract' if expected==-1 else 'add'
        if self.workload is not None:self.workload.observe(changed,action.generated_voxels)
        self.update_depth()
        if self.far_satisfied:self.termination='far_reached'
        elif self.workload is not None and self.workload.left(self.edits)==0:self.termination='step_budget_exhausted'
        if audit:self.evaluate()
        return self.terminal

    def metrics(self):
        sync(self.device)
        result={k:float(v.detach().cpu()) for k,v in self.values.items()}
        result.update(edits=self.edits,receivers=self.last_receiver_count,newReceivers=self.last_new_receivers,
            evaluationSeconds=self.last_eval_seconds,device=str(self.device),fieldStatus='uncalibrated_massing_proxies')
        result.update(far=self.far,targetFar=self.target_far,effectiveTargetFar=self.target_count/self.plot_area,
            farError=abs(self.far-self.target_far),plotArea=self.plot_area,
            maximumFar=self.capacity/self.plot_area,targetCells=self.target_count,occupiedCells=self.count,
            targetReached=self.far_satisfied,termination=self.termination,remainingEdits=abs(self.count-self.goal_count),remainingVoxels=abs(self.count-self.goal_count),
            requiredEdits=abs(self.start_count-min(self.acceptable_counts[1],max(self.acceptable_counts[0],self.start_count))),lastOperation=self.last_operation,
            actionSpace=getattr(self.frontier,"action_space","direct_facade_mask_far_v1"),voxelsEdited=self.voxels_edited,lastChangedVoxels=self.last_changed_voxels,lastGeneratedVoxels=getattr(self,"last_generated_voxels",0),
            facadeEvaluationSeconds=self.frontier.facade_seconds,maskProjectionSeconds=self.frontier.projection_seconds,maskCommitSeconds=self.frontier.commit_seconds,
            occupiedFloorArea=float(self.count),retention=self.count/self.capacity,
            environmentalAuditEdit=self.last_audit_edit,environmentalFieldsCurrent=self.last_audit_edit==self.edits,
            farTolerance=max(self.far_tolerance,.5/self.plot_area),farMin=self.acceptable_counts[0]/self.plot_area,farMax=self.acceptable_counts[1]/self.plot_area)
        result.update(depthDevice=str(getattr(self.frontier,'depth_device',None) or 'cpu'),depthFieldEdit=self.depth_edit,depthFieldCurrent=self.depth_edit==self.edits,depthEvaluationSeconds=self.depth_seconds,
            deepFloorArea=float(((self.depth>8)&self.occupied).sum().cpu()),
            depthPenalty=float(self.depth_badness.sum().cpu())/self.capacity,depthWeight=self.depth_weight,
            maxFacadeDepth=float(self.depth.max().cpu()))
        result['contextSource']='cached_osm' if self.context is not None else 'synthetic'
        result['noiseAvailable']=self.context is None
        result.update(stepBudget=self.workload.limit if self.workload else None,stepsRemaining=self.workload.left(self.edits) if self.workload else None,
            initialEditCapacity=self.workload.initial_capacity if self.workload else None,
            suggestedEditVoxels=self.workload.quota(abs(self.count-self.goal_count),self.edits) if self.workload else None)
        if self.context is not None:result['noisePrior']=None
        result.update(coresEnabled=self.cores_enabled,minCoreSpacing=self.min_core_spacing,maxCoreDist=self.max_core_dist,
            coreEvaluationSeconds=self.core_seconds,**self.core_metrics)
        return result

    def _reset_core_fields(self):
        self.cores=()
        self.core_mask=torch.zeros(self.shape,dtype=torch.bool,device=self.device)
        self.core_distance=torch.zeros(self.shape,dtype=torch.float32,device=self.device)
        self.core_dist_badness=torch.zeros(self.shape,dtype=torch.float32,device=self.device)
        self.core_seconds=0.
        self.core_metrics={
            'coreCount':0,'coreVoxels':0,'coreFootprintArea':0,'coreRatio':0.0,
            'coreDistanceMean':0.0,'coreDistanceMax':0.0,
            'coreUncoveredFraction':0.0,'coreUnreachableFraction':0.0,
            'coreDistBadness':0.0,'coreMinSpacingActual':0.0,
        }

    def configure_cores(self,*,cores_enabled=None,min_core_spacing=None,max_core_dist=None):
        if cores_enabled is not None:self.cores_enabled=bool(cores_enabled)
        if min_core_spacing is not None:self.min_core_spacing=max(0,int(min_core_spacing))
        if max_core_dist is not None:self.max_core_dist=max(1,int(max_core_dist))
        self.evaluate()

    def update_cores(self):
        if not self.cores_enabled:
            if not hasattr(self,'core_mask') or self.cores:
                self._reset_core_fields()
            return
        stamp=time.perf_counter()
        res=solve_cores(self.occupied,min_core_spacing=self.min_core_spacing,max_core_dist=self.max_core_dist)
        self.cores=res.cores
        self.core_mask=res.core_mask
        self.core_distance=res.core_distance
        self.core_dist_badness=res.core_dist_badness
        self.core_metrics=res.metrics
        self.core_seconds=time.perf_counter()-stamp
