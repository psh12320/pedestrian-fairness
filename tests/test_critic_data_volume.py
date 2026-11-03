import unittest
import json
from pathlib import Path
import tempfile
import torch
from scripts.critic_data_volume_check import check_split, value_model, paired_gate, bootstrap_ratio, fit_job, validate_fit, TRAIN, CHECK


class DataVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_frozen_seed_split(self):
        check_split(TRAIN,CHECK)
        self.assertEqual(TRAIN[:4],[49101,49102,49103,49104])
        self.assertEqual((len(TRAIN),len(CHECK)),(32,8))
        for training,checks in [([1,1],[2]),([1],[1]),([1],[58001])]:
            with self.assertRaises(ValueError):check_split(training,checks)

    def test_shared_initial_predictions_and_transform(self):
        y=torch.tensor([-200.,-100.,-50.]);x=torch.ones((3,4))
        small=value_model(4,1,y);large=value_model(4,1,y)
        self.assertTrue(torch.equal(small(x),large(x)))
        self.assertTrue(torch.equal(small.transform.mean,large.transform.mean))
        self.assertFalse(small.transform.mean.requires_grad)
        self.assertTrue(torch.equal(small.network[0].weight[:,-1],torch.zeros(64)))

    def test_different_denominator_cannot_create_win(self):
        a={'rmse':10.,'constant_rmse':20.,'own_constant_rmse':20.}
        b={'rmse':11.,'constant_rmse':20.,'own_constant_rmse':50.}
        self.assertFalse(paired_gate([a],[b])['passed'])
        b['rmse']=8.;self.assertTrue(paired_gate([a],[b])['passed'])
        b['own_constant_rmse']=7.;self.assertFalse(paired_gate([a],[b])['passed'])

    def test_paired_bootstrap_known_ratio_and_repeatability(self):
        cells={(i,s):(float(i+s),float(i+s)*.5) for i in [1,2,3] for s in [10,20,30]}
        first=bootstrap_ratio(cells,[1,2,3],[10,20,30],200)
        self.assertEqual(first,[.5,.5])
        self.assertEqual(first,bootstrap_ratio(cells,[1,2,3],[10,20,30],200))

    def test_fitting_cache_and_provenance_end_to_end(self):
        # Synthetic tensors test the pipeline only; never save them as SUMO results.
        protocol={'fingerprint':'unit-test-only','training_demand_seeds':TRAIN,'check_demand_seeds':CHECK}
        scratch=Path(__file__).resolve().parents[1]/'runs'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='critic-unit-',dir=scratch) as temporary:
            folder=Path(temporary)
            for seed in TRAIN+CHECK:
                x=torch.tensor([[0.,0.],[.2,.4],[.4,.2],[.8,.8]]) + (seed % 7)*.01
                record={'fingerprint':protocol['fingerprint'],'seed':seed,'junctions':{
                    tls:{'observations':x, 'returns':-100+20*x[:,0]+j,
                         'times':torch.tensor([1.,150.,300.,599.])} for j,tls in enumerate(('C0','C1','C2'))}}
                torch.save(record,folder/f'trajectory-{seed}.pt')
            args=(str(folder),protocol,'mappo',49301)
            path=Path(fit_job(args));original=path.read_bytes()
            row=json.loads(original)
            self.assertEqual(len(row['results']),18)
            self.assertEqual({r['training_states'] for r in row['results']},{16,128})
            validate_fit(row,protocol,'mappo',49301)
            self.assertEqual(fit_job(args),str(path))
            self.assertEqual(path.read_bytes(),original)
            row['results'].append(row['results'][0])
            with self.assertRaises(ValueError):validate_fit(row,protocol,'mappo',49301)
            row=json.loads(original);row['fingerprint']='wrong-source'
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):fit_job(args)


if __name__=='__main__':unittest.main()
