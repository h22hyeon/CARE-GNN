import paramiko
import time
import os
import itertools
import json

sshs = {}
channels = {}
#사용할 서버 리스트입니다. 맞추어 변경하시면 됩니다.
server_list = ["grsv-5","grsv-6"]

gpu_bus2idx = {}
for server_name in server_list:
	print(f"establishing connection to {server_name}...")
	new_ssh = paramiko.SSHClient()
	new_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
	new_ssh.connect(server_name)
	#환경 세팅에 필요한 부분을 new_channel에 전송하여 생성하시면 됩니다.
	new_channel = new_ssh.invoke_shell()
	new_channel.send("screen -S PCGNN_screen\n")
	time.sleep(1)
	new_channel.send("conda activate FDS\n")
	time.sleep(1)
	new_channel.send("cd ~/CARE-GNN\n")
	time.sleep(1)
	new_channel.send("module load cuda/cuda-10.2\n")
	time.sleep(1)
	new_channel.recv(9999)
	stdin, stdout, stderr = new_ssh.exec_command("nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader")
	gpu_bus2idx[server_name] = {}
	for line in stdout.readlines():
		idx, busid = line.strip().split(", ")
		gpu_bus2idx[server_name][busid] = idx
	sshs[server_name] = new_ssh
	channels[server_name] = new_channel

print("Established all connections!")

server_idx = 0
num_server = len(server_list)

prime_l = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
dir_path_l = [f'./exp_config_CARE_grid_{prime}' for prime in prime_l]

json_list = []
for dir_path in dir_path_l:
        json_list += sorted([os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)])

cnt = 0
for json_path in json_list:
    with open(json_path, 'r') as file:
        args = json.load(file)
    avail_gpu = None
    while avail_gpu is None:
        stdin, stdout, stderr = sshs[server_list[server_idx]].exec_command("nvidia-smi --query-gpu=index,pci.bus_id,memory.total,memory.used --format=csv,noheader")
        for line in stdout.readlines():
            gpu_idx, busid, mem_total, mem_used = line.strip().split(", ")
            # GPU 할당 조건 설정하는 부분입니다. 할당 여부만을 확인하여 진행하고 싶으시면 mem_used == 0 으로 조건 변경하여 사용하시면 됩니다.
            # 혹은 위의 exec_command의 command를 변경하는 방법도 있습니다.
            # "nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader"으로 하시면 결과값이 현재 사용중인 (process가 할당된) GPU의 busid가 나옵니다.
            mem_total = int(mem_total.split()[0])
            mem_used = int(mem_used.split()[0])
            if mem_used < 10 :
                avail_gpu = busid
                avail_gpu_idx = gpu_idx
                break
            
        if avail_gpu is None:
            #print(f"server {server_list[server_idx]} is not available")
            server_idx += 1
            server_idx %= num_server
            if server_idx == 0:
                #GPU 확인 주기입니다. 기본은 10초인데, 편하신대로 변경하시면 됩니다.
                time.sleep(10)
        else:
            #여기에 각 서버/gpu 에서 실행할 명령어를 적으시면 됩니다. CUDA_VISIBLE_DEVICES={avail_gpu_idx} 이후를 변경하시면 됩니다.
            my_command = f"\nCUDA_VISIBLE_DEVICES={avail_gpu_idx} python main.py --exp_config_path={json_path} >& /dev/null &\n"
            while not channels[server_list[server_idx]].send_ready():
                pass
            channels[server_list[server_idx]].send("\n")
            time.sleep(1)
            while channels[server_list[server_idx]].recv_ready():
                channels[server_list[server_idx]].recv(9999)
            sent_bytes = channels[server_list[server_idx]].send(my_command)
            assert sent_bytes != 0
            time.sleep(1)
            received = (channels[server_list[server_idx]].recv(9999)).decode('utf-8')
            received = received.split("\r\n")
            try:
                target_pid = received[-2].split()[1]
            except:
                print(server_list[server_idx], avail_gpu_idx)
                print(sent_bytes)
                print(my_command)
                print(received)
                target_pid = received[-2].split()[1]
            print(server_list[server_idx], avail_gpu_idx, target_pid, json_path)
            #print(received)
            while True:
                stdin, stdout, stderr = sshs[server_list[server_idx]].exec_command("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
                gpu_pids = []
                for line in stdout.readlines():
                    pid = line.strip()
                    gpu_pids.append(pid)
                if target_pid in gpu_pids:
                    break
                time.sleep(1)
print(cnt)
for server_name in server_list:
    channels[server_name].send("exit\n")
    sshs[server_name].close()