import copy
from pathlib import Path
import tempfile
import unittest
from scripts.run_local_timing_study import compare, validate, phase_checks, exclusive_lock, configurations


def result(loss=20, tail=30):
    return {'demand_sha256':'same','vehicles':{'scheduled_count':100,'completed_count':90,
        'unfinished_count':10,'uninserted_count':0,'completed_mean_time_loss_seconds':loss},
        'persons':{'scheduled':100,'completed':100,'unfinished':0,'uninserted':0},
        'crossings':{f':C{j}_c{c}':{'waits':[tail]*100,'p95':tail,'entries':100,
            'pending_at_cutoff':0,'unresolved_after_followup':0,'unknown_cohort_entries':0,'lost':0}
            for j in range(3) for c in range(4)}}


class LocalTimingTests(unittest.TestCase):
    def test_paired_improvement_and_seed_bootstrap(self):
        a=[result(18,29) for _ in range(4)];b=[result() for _ in range(4)]
        r=compare(a,b,draws=40)
        self.assertTrue(r['descriptive_joint_pass'])
        self.assertEqual(r['paired_seed_intervals']['vehicle_delta'],[-2,-2])
        self.assertEqual(r['point']['crossings_no_worse'],12)

    def test_service_loss_and_bad_crossing_fail(self):
        a=[result(18,29) for _ in range(4)];b=[result() for _ in range(4)]
        a[0]['vehicles'].update(completed_count=89,unfinished_count=11)
        self.assertFalse(compare(a,b,0)['descriptive_joint_pass'])
        a=[result(18,29) for _ in range(4)]
        for r in a:r['crossings'][':C0_c0'].update(waits=[35]*100,p95=35)
        self.assertFalse(compare(a,b,0)['descriptive_joint_pass'])

    def test_no_missing_wait_imputation_or_unmatched_demand(self):
        a=result(18);b=result()
        a['crossings'][':C0_c0'].update(waits=[],p95=None,entries=0)
        r=compare([a],[b],0)
        self.assertIsNone(r['point']['worst_crossing_delta'])
        self.assertFalse(r['descriptive_joint_pass']);self.assertTrue(r['sparse'])
        a['demand_sha256']='other'
        with self.assertRaises(ValueError):compare([a],[b],0)

    def test_person_and_vehicle_conservation(self):
        r=result();validate(r)
        r['persons']['completed']=99
        with self.assertRaises(ValueError):validate(r)
        r=result();r['vehicles']['completed_count']=89
        with self.assertRaises(ValueError):validate(r)

    def test_admission_and_minimum_phase_checks(self):
        m={'config':{'pedestrian_green_seconds':18,'vehicle_green_min_seconds':12,'yellow_seconds':3,'all_red_seconds':2},
           'phase_history':{'C0':[(0,'EW'),(12,'EW_YELLOW'),(15,'ALL_RED'),(17,'PED'),(35,'PED_CLEARANCE')]}}
        self.assertEqual(phase_checks(m)['PED'],1)
        m['phase_history']['C0'][-1]=(34,'PED_CLEARANCE')
        with self.assertRaises(AssertionError):phase_checks(m)

    def test_lock_excludes_second_writer_and_releases(self):
        root=Path(__file__).resolve().parents[1]/'runs'
        with tempfile.TemporaryDirectory(dir=root) as folder:
            with exclusive_lock(Path(folder)):
                with self.assertRaises(RuntimeError):
                    with exclusive_lock(Path(folder)):pass
            with exclusive_lock(Path(folder)):pass

    def test_balanced_144_run_design(self):
        from scripts.run_local_timing_study import CONTROLLERS,DURATIONS,SETTINGS
        p={'controllers':CONTROLLERS,'admission_seconds':DURATIONS,'settings':SETTINGS,'seeds':list(range(910001,910005))}
        self.assertEqual(len(configurations(p)),144)
        self.assertEqual(len(set(configurations(p))),144)


if __name__=='__main__':unittest.main()
