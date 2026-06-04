import subprocess, sys, random, os

def run(task='PickCube-v1', n=25):
    base_seed = random.randint(0, 99999)
    print('base_seed:', base_seed)
    proc = subprocess.Popen(
        [sys.executable, 'rollout_subprocess.py',
         '--task', task, '--n', str(n), '--base-seed', str(base_seed)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd='/content/rdt-igtesting'
    )
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    print('Exit code:', proc.returncode)
