import os
from subprocess import PIPE, run

try:
    os.mkdir('results')
except:
    pass

def run_cmd(command, name):
    print("Render {} ..".format(name))
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    log_str = result.stdout
    with open('results/{}_log.txt'.format(name), 'w') as file:
        file.write(log_str)

# 5 min for all methods that we want to compare
timeout = 1*60*30
# timeout = 5
# timeout = 30

block_size = 1024
# block_size = 512

name = "ring_sms_reference"
cmd = "mitsuba "
cmd += "sms_visnet_ring.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
# cmd += "-Dspp=5 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "
# cmd += "-Dcaustics_biased=true "
# cmd += "-Dcaustics_max_trials={} ".format(8)
cmd += "-Dvisnet_enable=false "
# print(cmd)
run_cmd(cmd, name)

