import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from src.traffic_control.ppo_repair.learning import (
    RepairNetworks,RepairLearner,RunningValueScale,prepare_batch,VARIANTS,
)
from src.traffic_control.fairness.networks import JunctionNetworks
from src.traffic_control.fairness.learning import Learner
from src.traffic_control.ppo_repair.reporting import comparison,bootstrap_estimator,estimates


FEATURES={'C0':['queue','phase'],'C1':['queue','phase']}


def trace(queue=2,forced=False):
    return [{'junction':tls,'time':i+1,'observation':[float(queue),float(i%2)],
             'global_state':[float(queue),float(i%2),float(queue),float(i%2)],
             'mask':[True,False,False] if forced else [True,True,False],
             'action':0 if forced else i%2,'vehicle_queue':queue+i,
             'uninserted_due':0,'pending_count':1} for i in range(6) for tls in FEATURES]


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_equal_initial_actors_and_critic_functions(self):
        for method in ('ippo','mappo'):
            old=RepairNetworks(method,FEATURES,'original',1)
            new=RepairNetworks(method,FEATURES,'normalized-clock',1)
            for tls in FEATURES:
                x=torch.tensor([[2.,1.],[1.,0.]])
                a=old.critic_inputs(x,torch.cat([x,x],1),[1.,5.],6)
                b=new.critic_inputs(x,torch.cat([x,x],1),[1.,5.],6)
                self.assertTrue(torch.equal(old.prediction(tls,a),new.prediction(tls,b)))
                self.assertTrue(torch.equal(old.actors[tls](x),new.actors[tls](x)))
                # Actor signatures accept local observations and masks only.
                self.assertEqual(old.actors[tls][0].in_features,2)
                self.assertEqual(new.critics[tls][0].in_features,(2 if method=='ippo' else 4)+1)

    def test_normalizer_preserves_raw_predictions_and_counts_once(self):
        scale=RunningValueScale();head=torch.nn.Linear(3,1)
        x=torch.tensor([[1.,2.,3.],[2.,1.,0.]])
        for targets in (torch.tensor([-500.,-100.,-300.]),torch.tensor([-20.,-40.])):
            before=scale.denormalize(head(x)).detach()
            scale.update(targets,head)
            self.assertTrue(torch.allclose(before,scale.denormalize(head(x)),atol=5e-5))
        pooled=torch.tensor([-500.,-100.,-300.,-20.,-40.],dtype=torch.float64)
        self.assertEqual(float(scale.count),5.)
        self.assertAlmostEqual(float(scale.mean),float(pooled.mean()))
        self.assertAlmostEqual(float(scale.scale),float(pooled.std(unbiased=False)))

    def test_original_update_matches_existing_learner(self):
        for method in ('ippo','mappo'):
            original=JunctionNetworks(method,FEATURES,7)
            new=RepairNetworks(method,FEATURES,'original',7)
            old_learner=Learner(original);new_learner=RepairLearner(new)
            for _ in range(2):
                old_learner.update(trace(),[])
                new_learner.update([{'trace':trace(),'horizon':6}])
            for key,value in original.state_dict().items():
                self.assertTrue(torch.allclose(value,new.state_dict()[key],atol=1e-7,rtol=1e-6),key)

    def test_gae_does_not_leak_between_episodes(self):
        model=RepairNetworks('ippo',FEATURES)
        with torch.no_grad():
            for critic in model.critics.values():
                for p in critic.parameters():p.zero_()
        first=trace(0);second=trace(100)
        b=prepare_batch(model,'C0',[{'trace':first,'horizon':6},{'trace':second,'horizon':6}])
        # At each true terminal, target is just that episode's final reward.
        self.assertAlmostEqual(float(b['returns'][4]),-(5+3)/10,places=6)
        self.assertAlmostEqual(float(b['returns'][9]),-(105+3)/10,places=5)
        self.assertEqual(len(b['returns']),10)

    def test_stale_actor_log_probability_rejected(self):
        model=RepairNetworks('ippo',FEATURES)
        rows=trace();rows[0]['behavior_logp']=-100.
        with self.assertRaises(ValueError):prepare_batch(model,'C0',[{'trace':rows,'horizon':6}])

    def test_forced_decisions_do_not_move_actor_even_with_adam_momentum(self):
        model=RepairNetworks('ippo',FEATURES,'normalized-clock',7);learner=RepairLearner(model)
        learner.update([{'trace':trace(),'horizon':6}])
        before=copy.deepcopy(model.actors.state_dict())
        stats=learner.update([{'trace':trace(forced=True),'horizon':6}])
        self.assertTrue(all(torch.equal(v,model.actors.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(s['decisions']==0 for s in stats.values()))

    def test_model_optimizer_and_normalizer_resume_exactly(self):
        model=RepairNetworks('mappo',FEATURES,'normalized-clock-batch4',7);learner=RepairLearner(model)
        batch=[{'trace':trace(i+1),'horizon':6} for i in range(4)]
        learner.update(batch)
        stream=io.BytesIO();torch.save({'model':model.snapshot(),'optimizer':learner.optimizer.state_dict()},stream)
        stream.seek(0);saved=torch.load(stream,weights_only=True)
        restored=RepairNetworks.from_snapshot(saved['model']);other=RepairLearner(restored)
        other.optimizer.load_state_dict(saved['optimizer'])
        learner.update(batch);other.update(batch)
        self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
        self.assertEqual(float(model.scales['C0'].count),40.)

    def test_actor_changes_after_real_free_decisions_all_variants(self):
        for variant in VARIANTS:
            model=RepairNetworks('ippo',FEATURES,variant);before=copy.deepcopy(model.actors.state_dict())
            learner=RepairLearner(model)
            stats=learner.update([{'trace':trace(),'horizon':6}])
            self.assertTrue(any(not torch.equal(v,model.actors.state_dict()[k]) for k,v in before.items()))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertTrue(all(v['states']==5 for v in stats.values()))

    def test_atomic_checkpoint_recovers_missing_final_exports(self):
        from src.traffic_control.ppo_repair.pilot import train_job,atomic_torch
        scratch=Path(__file__).resolve().parents[1]/'runs';scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='repair-unit-',dir=scratch) as folder:
            root=Path(folder);(root/'baselines').mkdir()
            (root/'baselines/one.json').write_text(json.dumps({'records':[{'features':FEATURES}]}))
            label='mappo-normalized-clock-batch4-r0';directory=root/'models'/label;directory.mkdir(parents=True)
            model=RepairNetworks('mappo',FEATURES,'normalized-clock-batch4',7);learner=RepairLearner(model)
            batch=[{'trace':trace(i+1),'horizon':6} for i in range(4)];learner.update(batch)
            history=[{'episodes':4,'rollouts':[]}]
            atomic_torch(directory/'latest.pt',{'fingerprint':'unit','label':label,'episodes':4,
                'model':model.snapshot(),'optimizer':learner.optimizer.state_dict(),'rng':torch.get_rng_state(),
                'history':history,'validations':[]})
            p={'fingerprint':'unit','initial_seeds':[7],'episodes':4}
            with patch('src.traffic_control.ppo_repair.pilot.episode',side_effect=AssertionError('Must not rerun completed batch')):
                train_job((str(root),p,'mappo','normalized-clock-batch4',0,label))
            self.assertEqual(json.loads((directory/'learning.json').read_text())['history'],history)
            self.assertTrue((directory/'complete.json').is_file())
            final=torch.load(directory/'final.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,final['model']['weights'][k]) for k,v in model.state_dict().items()))


class ComparisonTests(unittest.TestCase):
    def result(self,loss=10,completed=10,tail=20):
        return {'demand_sha256':'same','vehicles':{'completed_mean_time_loss_seconds':loss,
            'completed_count':completed,'unfinished_count':10-completed,'uninserted_count':0},
            'crossings':{f':C{i//4}_c{i%4}':{'waits':[tail]*100,'p95':tail,'entries':100,
                'pending_at_cutoff':0,'unresolved_after_followup':0,'unknown_cohort_entries':0,'lost':0} for i in range(12)}}

    def test_vehicle_gain_cannot_hide_unserved_or_bad_crossing(self):
        reference=[[self.result()]*2 for _ in range(3)]
        models=[[self.result(loss=8)]*2 for _ in range(3)]
        c=comparison(models,reference,draws=40)
        self.assertTrue(c['descriptive_pass'])
        self.assertEqual(c['exploratory_intervals']['vehicle_delta_seconds'],[-2,-2])
        models[0][0]=self.result(loss=8,completed=9)
        self.assertFalse(comparison(models,reference,40)['descriptive_pass'])
        models[0][0]=self.result(loss=8,tail=25)
        self.assertFalse(comparison(models,reference,40)['descriptive_pass'])

    def test_missing_crossing_is_not_zero_wait_and_demand_must_match(self):
        ref=self.result();a=self.result(loss=8)
        a['crossings'][':C0_c0'].update(waits=[],p95=None,entries=0)
        result=comparison([[a]],[[ref]])
        self.assertEqual(result['point']['vehicle_delta_seconds'],-2)
        self.assertEqual(result['point']['missing_crossings'],[':C0_c0'])
        self.assertIsNone(result['point']['worst_crossing_delta_seconds'])
        self.assertFalse(result['descriptive_pass']);self.assertTrue(result['sparse'])
        a['demand_sha256']='different'
        with self.assertRaises(ValueError):comparison([[a]],[[ref]])

    def test_fast_weighted_tails_match_explicit_resampling(self):
        import random
        rng=random.Random(42)
        data=[[[self.result(loss=7+r+s+k) for s in range(4)] for r in range(3)] for k in range(2)]
        for group in data:
            for row in group:
                for result in row:
                    for crossing in result['crossings'].values():
                        crossing['waits']=[rng.randrange(100)/5 for _ in range(rng.randrange(1,30))]
        fast=bootstrap_estimator(*data)
        for _ in range(25):
            reps=rng.choices(range(3),k=3);seeds=rng.choices(range(4),k=4)
            a,b=fast(reps,seeds),estimates(*data,reps,seeds)
            self.assertAlmostEqual(a['vehicle_delta_seconds'],b['vehicle_delta_seconds'])
            for c in a['crossing_p95_delta_seconds']:
                self.assertAlmostEqual(a['crossing_p95_delta_seconds'][c],b['crossing_p95_delta_seconds'][c])


if __name__=='__main__':unittest.main()
