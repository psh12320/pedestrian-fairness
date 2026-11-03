"""Matched PPO ablation with raw-unit GAE and boundary-safe rollout batches.

The repaired critic uses cumulative training-target moments and a PopArt-style
output-layer rescaling. This preserves raw predictions as moments change.
It is an independent adaptation, not an exact reproduction of the MAPPO repo.
"""
import copy
import torch
from torch import nn
from ..fairness.learning import advantages
from ..fairness.networks import JunctionNetworks, mlp
from . import SCHEMA

VARIANTS = ('original', 'normalized-clock', 'normalized-clock-batch4')


class RunningValueScale(nn.Module):
    def __init__(self):
        super().__init__()
        for name, initial in [('count',0.),('mean',0.),('m2',0.),('scale',1.)]:
            self.register_buffer(name,torch.tensor(initial,dtype=torch.float64))

    def normalize(self, values):
        return (values-self.mean.to(values))/self.scale.to(values)

    def denormalize(self, values):
        return values*self.scale.to(values)+self.mean.to(values)

    @torch.no_grad()
    def update(self, targets, output_layer):
        values=targets.detach().to(dtype=torch.float64)
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError('Expected finite training targets')
        old_mean,old_scale=self.mean.clone(),self.scale.clone()
        n=values.numel(); mean=values.mean(); m2=((values-mean)**2).sum()
        count=self.count+n; delta=mean-self.mean
        self.m2.add_(m2+delta.square()*self.count*n/count)
        self.mean.add_(delta*n/count); self.count.copy_(count)
        self.scale.copy_((self.m2/count).clamp_min(1.).sqrt())
        # In-place parameter changes keep optimizer parameter references valid.
        output_layer.weight.mul_((old_scale/self.scale).to(output_layer.weight))
        output_layer.bias.copy_(((old_scale*output_layer.bias.double()+old_mean-self.mean)/self.scale).to(output_layer.bias))


class RepairNetworks(JunctionNetworks):
    def __init__(self, method, features, variant='original', seed=72001):
        if variant not in VARIANTS: raise ValueError(variant)
        super().__init__(method,features,seed)
        self.variant=variant
        self.repaired=variant!='original'
        self.scales=nn.ModuleDict({tls:RunningValueScale() for tls in self.features})
        if self.repaired:
            # Equal actor parameters and equal initial critic function across arms.
            for tls,old in list(self.critics.items()):
                with torch.random.fork_rng(devices=[]):
                    new=mlp(old[0].in_features+1,1)
                with torch.no_grad():
                    new[0].weight[:,:-1].copy_(old[0].weight)
                    new[0].weight[:,-1].zero_(); new[0].bias.copy_(old[0].bias)
                    for index in (2,4):new[index].load_state_dict(old[index].state_dict())
                self.critics[tls]=new

    def critic_inputs(self, observation, state, times, horizon):
        value=observation if self.method=='ippo' else state
        x=torch.as_tensor(value,dtype=torch.float32,device=next(self.parameters()).device)
        if self.repaired:
            if horizon<=0:raise ValueError('Invalid horizon')
            clock=torch.as_tensor(times,dtype=x.dtype,device=x.device)
            if torch.any(clock<0) or torch.any(clock>horizon):raise ValueError('Time outside finite horizon')
            x=torch.cat((x,((horizon-clock)/horizon).unsqueeze(-1)),dim=-1)
        return x

    def prediction(self,tls,inputs):
        value=self.critics[tls](inputs).squeeze(-1)
        return self.scales[tls].denormalize(value) if self.repaired else value

    def snapshot(self):
        return {'schema':SCHEMA,'method':self.method,'variant':self.variant,
                'features':self.features,'weights':copy.deepcopy(self.state_dict())}

    @classmethod
    def from_snapshot(cls,saved):
        if saved['schema']!=SCHEMA:raise ValueError('Wrong checkpoint schema')
        model=cls(saved['method'],saved['features'],saved['variant'])
        model.load_state_dict(saved['weights']); return model


def raw_return(rewards,gamma=.99):
    result=[]; following=0.
    for reward in reversed(rewards):
        following=reward+gamma*following;result.append(following)
    return list(reversed(result))


def prediction_stats(target, prediction):
    error=prediction-target; variance=target.var(unbiased=False)
    return {'rmse':float(error.square().mean().sqrt()),'bias':float(error.mean()),
            'target_mean':float(target.mean()),'prediction_mean':float(prediction.mean()),
            'explained_variance':float(1-error.var(unbiased=False)/variance) if variance>1e-12 else None}


def prepare_batch(model,tls,rollouts,weight=3):
    """GAE stops at each episode's terminal state; concatenation happens last."""
    device=next(model.parameters()).device
    def tensor(v,dtype=torch.float32):return torch.as_tensor(v,dtype=dtype,device=device)
    episodes=[]
    for rollout in rollouts:
        rows=[r for r in rollout['trace'] if r['junction']==tls]
        if len(rows)<2:raise ValueError('Rollout too short')
        obs=tensor([r['observation'] for r in rows[:-1]])
        states=tensor([r['global_state'] for r in rows[:-1]])
        inputs=model.critic_inputs(obs,states,[r['time'] for r in rows[:-1]],rollout['horizon'])
        actions=tensor([r['action'] if r['action'] is not None else 0 for r in rows[:-1]],torch.long)
        valid=tensor([r['action'] is not None and sum(r['mask'])>1 for r in rows[:-1]],torch.bool)
        masks=tensor([r['mask'] if any(r['mask']) else [True,False,False] for r in rows[:-1]],torch.bool)
        if any(r['action'] is not None and not r['mask'][r['action']] for r in rows[:-1]):
            raise ValueError('Action violates the collection mask')
        rewards=[-(r['vehicle_queue']+r['uninserted_due']+weight*r['pending_count'])/10 for r in rows[1:]]
        with torch.no_grad():
            values=model.prediction(tls,inputs)
            old_logp=model.distribution(tls,obs,masks).log_prob(actions)
        # Detect stale-policy collection before doing any optimizer update.
        for i,row in enumerate(rows[:-1]):
            if 'behavior_logp' in row and abs(float(old_logp[i])-row['behavior_logp'])>1e-5:
                raise ValueError('Rollout is not on-policy or action mask changed')
        adv,returns=advantages(rewards,values.tolist())
        episodes.append({'observations':obs,'inputs':inputs,'actions':actions,'valid':valid,'masks':masks,
                         'old_logp':old_logp,'old_values':values,'advantages':tensor(adv),
                         'returns':tensor(returns),'monte_carlo':tensor(raw_return(rewards))})
    return {key:torch.cat([e[key] for e in episodes]) for key in episodes[0]}


class RepairLearner:
    def __init__(self,model,weight=3):
        self.model=model;self.weight=weight
        self.optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,eps=1e-5)

    def update(self,rollouts):
        if not rollouts:raise ValueError('Empty rollout batch')
        # Prepare all junctions before changing any parameters.
        batches={tls:prepare_batch(self.model,tls,rollouts,self.weight) for tls in self.model.features}
        results={}
        for tls,b in batches.items():
            valid=b['valid']; adv=b['advantages']
            if valid.any():adv=(adv-adv[valid].mean())/adv[valid].std(unbiased=False).clamp_min(1e-6)
            value_scale=b['returns'].std(unbiased=False).clamp_min(1.)
            if self.model.repaired:
                with torch.no_grad():before=self.model.prediction(tls,b['inputs']).clone()
                self.model.scales[tls].update(b['returns'],self.model.critics[tls][-1])
                with torch.no_grad():after=self.model.prediction(tls,b['inputs'])
                if not torch.allclose(before,after,atol=1e-4,rtol=1e-5):
                    raise AssertionError('Normalisation changed raw value predictions')
            for _ in range(4):
                distribution=self.model.distribution(tls,b['observations'],b['masks'])
                logp=distribution.log_prob(b['actions']); ratio=(logp-b['old_logp']).exp()
                surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
                actor_loss=-surrogate[valid].mean() if valid.any() else logp.sum()*0
                value=self.model.prediction(tls,b['inputs'])
                if self.model.repaired:
                    normalized=self.model.critics[tls](b['inputs']).squeeze(-1)
                    value_loss=(normalized-self.model.scales[tls].normalize(b['returns'])).square().mean()
                else:value_loss=((value-b['returns'])/value_scale).square().mean()
                entropy=distribution.entropy()[valid].mean() if valid.any() else actor_loss*0
                loss=actor_loss+.5*value_loss-.01*entropy
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                if not valid.any():
                    # Adam momentum must not move an actor with no free decisions.
                    for parameter in self.model.actors[tls].parameters():parameter.grad=None
                gradient=torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5)
                if not torch.isfinite(gradient):raise FloatingPointError('Nonfinite gradient')
                self.optimizer.step()
            with torch.no_grad():
                value=self.model.prediction(tls,b['inputs'])
                dist=self.model.distribution(tls,b['observations'],b['masks'])
                delta=dist.log_prob(b['actions'])-b['old_logp']
                kl=((delta.exp()-1)-delta)[valid].mean() if valid.any() else torch.tensor(0.)
            results[tls]={'loss':float(loss.detach()),'actor_loss':float(actor_loss.detach()),
                'value_loss':float(value_loss.detach()),'entropy':float(entropy.detach()),
                'gradient_norm':float(gradient),'approx_kl_after':float(kl),'decisions':int(valid.sum()),
                'states':len(valid),'episodes_in_update':len(rollouts),
                'normalizer_count':float(self.model.scales[tls].count),
                'normalizer_mean':float(self.model.scales[tls].mean),'normalizer_scale':float(self.model.scales[tls].scale),
                'before_fit_mc':prediction_stats(b['monte_carlo'],b['old_values']),
                'after_fit_mc':prediction_stats(b['monte_carlo'],value),
                'after_fit_gae':prediction_stats(b['returns'],value)}
        return results
