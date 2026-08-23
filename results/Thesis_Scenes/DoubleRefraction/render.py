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
timeout = 5*60
# timeout = 5
# timeout = 30

block_size = 1024
# block_size = 512

name = "slab_sms_test"
cmd = "mitsuba "
cmd += "slab_sms.xml "
cmd += "-o results/{}.exr ".format(name)
# cmd += "-Dspp=999999999 "
cmd += "-Dspp=5 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "
# cmd += "-Dcaustics_max_trials={} ".format(8)
# run_cmd(cmd, name)

name = "slab_visnet_sms_vis_test"
cmd = "mitsuba "
cmd += "slab_visnet_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
# cmd += "-Dspp=5 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "

# cmd += "-Dcaustics_biased=true "
# cmd += "-Dcaustics_max_trials=8 "

cmd += "-Dmodel_device=gpu "
cmd += "-Dvisnet_enable=true "
cmd += "-Dvisnet_rr_threshold=0.1 "
cmd += "-Dvisnet_threshold=0.4 "

cmd += "-Dblock_size={} ".format(block_size)
# print(cmd)
# run_cmd(cmd, name)

name = "slab_visnet_sms_test"
cmd = "mitsuba "
cmd += "slab_visnet_sms.xml "
cmd += "-o results/{}.exr ".format(name)
cmd += "-Dspp=999999999 "
# cmd += "-Dspp=5 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "

# cmd += "-Dcaustics_biased=true "
# cmd += "-Dcaustics_max_trials=8 "

cmd += "-Dvisnet_enable=false "

cmd += "-Dblock_size={} ".format(block_size)
# print(cmd)
# run_cmd(cmd, name)

name = "slab_flow_test"
cmd = "mitsuba "
cmd += "slab_flow.xml "
cmd += "-o results/{}.exr ".format(name)
# cmd += "-Dspp=999999999 "
cmd += "-Dspp=1 "
cmd += "-Dsamples_per_pass=1 "
cmd += "-Dtimeout={} ".format(timeout)
cmd += "-Dcaustics_biased=false "

# cmd += "-Dcaustics_biased=true "
# cmd += "-Dcaustics_max_trials=8 "

cmd += "-Dmodel_device=gpu "
cmd += "-Dvisnet_enable=true "
cmd += "-Dvisnet_threshold=0.4 "

cmd += "-Dblock_size={} ".format(256)
print(cmd)
run_cmd(cmd, name)

# for M in [1, 2, 4, 8]:
#     name = "slab_sms_b{:02d}".format(M)
#     cmd = "mitsuba "
#     cmd += "slab_sms.xml "
#     cmd += "-o results/{}.exr ".format(name)
#     cmd += "-Dspp=999999999 "
#     cmd += "-Dsamples_per_pass=1 "
#     cmd += "-Dtimeout={} ".format(timeout)
#     cmd += "-Dcrop_offset_x={} ".format(crop_x)
#     cmd += "-Dcrop_offset_y={} ".format(crop_y)
#     cmd += "-Dcrop_width={} ".format(crop_s)
#     cmd += "-Dcrop_height={} ".format(crop_s)
#     cmd += "-Dcaustics_biased=true "
#     cmd += "-Dcaustics_max_trials={} ".format(M)
#     # run_cmd(cmd, name)

print("done.")
