"""Small fresh-checkout source bundle or completed summary-only result bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT/'runs/ppo-repair-pilot')
    args=parser.parse_args();(ROOT/'runs').mkdir(exist_ok=True)
    if args.results:
        root=args.root.resolve();protocol=json.loads((root/'protocol.json').read_text())
        # Validate completion without repeating the already-finished bootstrap.
        summary=json.loads((root/'summary.json').read_text())
        fp=protocol['fingerprint']
        if summary['protocol']!=protocol or summary['fingerprint']!=fp or not (root/'REPORT.md').is_file():
            raise ValueError('Missing or mismatched completed report')
        expected_models=len(protocol['methods'])*len(protocol['variants'])*protocol['replicas']
        if summary['completed_models']!=expected_models:raise ValueError('Incomplete model coverage')
        for method in protocol['methods']:
            for variant in protocol['variants']:
                for r in range(protocol['replicas']):
                    label=f'{method}-{variant}-r{r}';folder=root/'models'/label
                    complete=json.loads((folder/'complete.json').read_text())
                    if complete['fingerprint']!=fp or complete['episodes']!=protocol['episodes']:
                        raise ValueError('Incomplete training: '+label)
                    sha=hashlib.sha256((folder/'final.pt').read_bytes()).hexdigest()
                    if sha!=complete['checkpoint_sha256']:raise ValueError('Changed model: '+label)
                    for scenario in protocol['evaluation_scenarios']:
                        for seed in protocol['evaluation_seeds']:
                            record=json.loads((root/'evaluation'/f'{label}-{scenario}-{seed}.json').read_text())
                            if record['fingerprint']!=fp or record['checkpoint_sha256']!=sha:
                                raise ValueError('Changed or mismatched evaluation')
        path=ROOT/'runs/ppo-repair-results.zip'
        if protocol['quick']:path=ROOT/'runs/ppo-repair-smoke-results.zip'
        files=[*root.rglob('*.json'),root/'REPORT.md']
        for f in files:
            if f.suffix=='.json' and json.loads(f.read_text()).get('fingerprint')!=fp:
                raise ValueError('Mixed protocols in output: '+str(f))
        manifest={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT).as_posix())
            z.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
    else:
        path=ROOT/'runs/ppo-repair-source.tar.gz'
        files=[]
        for folder,pattern in [('src','*.py'),('config','*.json'),('scenario','single.*.xml')]:
            files.extend((ROOT/folder).rglob(pattern))
        files.extend(ROOT/'tests'/n for n in ('__init__.py','test_ppo_repair.py','test_fairness_learning.py',
            'test_fairness_analysis.py','test_fairness_v2.py','test_fairness_stress.py'))
        files.extend(ROOT/n for n in ['scripts/run_ppo_repair_cluster.sh','scripts/package_ppo_repair.py',
            'scripts/collect_ppo_repair_status.py','docs/ppo_repair_pilot.md','requirements.txt'])
        for f in files:
            if not f.is_file():raise FileNotFoundError(f)
        temporary=path.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as archive:
            for f in sorted(files):archive.add(f,arcname='trafficlightcontrol/'+f.relative_to(ROOT).as_posix())
        with tarfile.open(temporary,'r:gz') as archive:
            if len(archive.getnames())!=len(set(archive.getnames())):raise ValueError('Duplicate source entries')
            for f in files:
                name='trafficlightcontrol/'+f.relative_to(ROOT).as_posix()
                if archive.extractfile(name).read()!=f.read_bytes():raise ValueError('Source archive mismatch')
        temporary.replace(path)
    print(path)
    print(f'{path.stat().st_size/1024:.1f} KiB')


if __name__=='__main__':main()
