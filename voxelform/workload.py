"""A fixed episode deadline estimated from initial edit capacity, not site labels."""
from dataclasses import dataclass,asdict
import math


@dataclass
class WorkloadBudget:
    required:int
    initial_capacity:int
    limit:int
    origin:int=0
    margin:int=2
    retention:float=.65

    @classmethod
    def create(cls,env,config):
        required=abs(env.count-env.goal_count)
        operation=-1 if env.count>env.goal_count else 1
        # Deep eligibility is permission, not guaranteed coherent cut capacity.
        # Use the current lateral facade for the deadline; wells may beat it.
        capacity=int((env.frontier.remove_skin if operation==-1 else env.frontier.add_skin).sum())
        usable=max(1,capacity*config.budget_fraction)
        if config.maximum_edit_voxels:usable=min(usable,config.maximum_edit_voxels)
        productive=math.ceil(required/usable)
        margin=config.budget_margin+math.ceil(productive*config.budget_reserve_fraction)
        limit=productive+margin if required else 0
        return cls(required,capacity,limit,env.edits,margin)

    def left(self,edits):return max(0,self.limit-(edits-self.origin))
    def quota(self,remaining,edits):
        # Reserve the margin for projection losses and final adjustments.
        return math.ceil(remaining/max(1,self.left(edits)-self.margin))
    def state(self):return asdict(self)
    def observe(self,applied,generated):
        self.retention=.7*self.retention+.3*max(.15,min(1.,applied/max(1,generated)))


def reset_budget(env):
    config=getattr(env.frontier,'config',None)
    env.workload=WorkloadBudget.create(env,config) if getattr(config,'allocation','probability')!='probability' else None


def action_reward(env,before):
    """Normalized progress, small action cost, success and continuous timeout error."""
    remaining=abs(env.count-env.goal_count)
    if env.workload is None:return before-env.far_error+(1. if env.termination=='far_reached' else 0.)
    initial=max(1,env.workload.required)
    progress=(before*env.plot_area-remaining)/initial
    reward=progress-.05/max(1,env.workload.limit)
    if env.termination=='far_reached':reward+=1.
    elif env.terminal:reward-=1.+remaining/initial
    return reward
