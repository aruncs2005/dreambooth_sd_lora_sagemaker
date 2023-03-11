import argparse
import json
import os
import socket
import subprocess as sb
from datetime import datetime
import yaml

SM_CONFIG_PATH = '/opt/ml/input/config/resourceconfig.json'

if __name__ == "__main__":
    
    if os.path.exists(SM_CONFIG_PATH):
        with open(SM_CONFIG_PATH) as file:
            cluster_config = json.load(file)

        hosts = cluster_config['hosts']
        default_nodes = len(hosts)
        default_node_rank = hosts.index(os.environ.get("SM_CURRENT_HOST"))
        
        # elect a leader for accelerate
        for host in cluster_config['hosts']:
            print(f'host: {host}, IP: {socket.gethostbyname(host)}')
        leader = cluster_config['hosts'][0]  # take first machine in the host list
        
        # Set the network interface for inter node communication
        os.environ['NCCL_SOCKET_IFNAME'] = cluster_config['network_interface_name']
        leader = socket.gethostbyname(hosts[0])
        
    else:
        # if not on SageMaker, default to single-machine (eg test on Notebook/IDE)
        default_nodes = 1
        default_node_rank = 0
        leader = '127.0.0.1'




    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--gpus", type=int, default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--nodes", type=int, default=default_nodes)
    parser.add_argument("--node_rank", type=int, default=default_node_rank)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--instance_data_dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--pretrained_model_name_or_path", type=str,required=True)
    parser.add_argument("--instance_prompt", type=str,default="a photo of sks dog")
    parser.add_argument("--resolution", type=int,default=512)
    parser.add_argument("--train_batch_size", type=int,default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,default=1)
    parser.add_argument("--learning_rate", type=float,default=5e-6)
    parser.add_argument("--max_train_steps", type=int,default=400)
    
    
    date_string = datetime.now().strftime("%Y%m-%d%H-%M%S")
    
    args, _ = parser.parse_known_args()

    # Dynamically add number of GPUs, number of nodes and leader node information to the config file
    file = open('accelerate_config.yaml', 'r')

    accelerate_config = yaml.safe_load(file)

    file.close()

    accelerate_config['num_processes'] = int(args.gpus)*int(args.nodes)
    accelerate_config['num_machines'] = args.nodes
    accelerate_config['main_process_ip'] = leader
    accelerate_config['main_process_port'] = 7777
        

    print("launching accelerate with the following config params")
    print(accelerate_config)
    file = open('accelerate_config.yaml', 'w')
    yaml.dump(accelerate_config, file)
    file.close()
    
    print("***********printing the args passed to accelerate function **************")
    print(args)
    
    sb.call([
             # torch cluster config
             'accelerate', 
             'launch',
             '--config_file', 'accelerate_config.yaml',                     
             # training config
             'train_dreambooth_lora.py',
            '--pretrained_model_name_or_path',args.pretrained_model_name_or_path,
            '--instance_data_dir',args.instance_data_dir,
            '--output_dir',args.model_dir,
            '--instance_prompt',"a photo of sks dog",
            '--resolution',str(args.resolution),
            '--train_batch_size',str(args.train_batch_size),
            '--gradient_accumulation_steps',str(args.gradient_accumulation_steps),
            '--learning_rate',str(args.learning_rate),
            '--max_train_steps',str(args.max_train_steps)
       
            ])