import argparse
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil
import logging
import collections
import time
import uuid
from pprint import pformat

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(dirname(dirname(abspath(__file__))))
from envs import registration
from envs.wrappers import StackFrame
from td3.net import *
from td3.rl import Actor, Critic, TD3, ReplayBuffer, DynaTD3, Model, SMCPTD3
from td3.collector import CondorCollector, LocalCollector

from td3.sac import GaussianActor, SAC

# 加载配置文件yaml
def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config
# initialize_logging函数用于初始化日志记录器。它根据配置文件中的设置,创建一个唯一的保存路径,并在该路径下创建一个SummaryWriter对象,用于将训练指标写入TensorBoard日志。此外,它还将配置文件复制到保存路径中。
def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    if training_config["safe_rl"]:
        mode = training_config["safe_mode"]
        string = f"safe_rl_{mode}_"
        if mode == "lagr":
            string = string + "lagr"+str(training_config["safe_lagr"]) + "_"
    else:
        string = ""

    string = string + dt_string

    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        string,
        uuid.uuid4().hex[:4]
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer
# initialize_envs函数根据配置文件中的设置创建强化学习环境。如果使用Condor进行分布式训练,则将init_sim设置为False。然后,使用gym.make创建指定的环境,并使用StackFrame封装器对观测值进行堆叠。
def initialize_envs(config):
    env_config = config["env_config"]
    if env_config["use_condor"]:
        env_config["kwargs"]["init_sim"] = False
    
    # if not env_config["use_condor"]:
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    # else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
    #    print("    >>>> Using actors on Condor")
    #    env = InfoEnv(config)
    return env

def seed(config):
    env_config = config["env_config"]
    
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])
# 该函数根据指定的编码器类型(如MLP、RNN、CNN或Transformer)和参数创建编码器对象。这些编码器可能在td3.net模块中定义。
def get_encoder(encoder_type, args):
    if encoder_type == "mlp":
        encoder=MLPEncoder(**args)
    elif encoder_type == 'rnn':
        encoder=RNNEncoder(**args)
    elif encoder_type == 'cnn':
        encoder=CNNEncoder(**args)
    elif encoder_type == 'transformer':
        encoder=TransformerEncoder(**args)
    else:
        raise Exception(f"[error] Unknown encoder type {encoder_type}!")
    return encoder

# 该函数根据配置文件初始化强化学习策略(policy)和经验回放缓冲区(replay buffer)。
'''
首先,它从环境中获取状态和动作的维度,以及动作空间的上下界。然后,根据配置文件中指定的编码器类型和参数,创建Actor和Critic网络,并将它们移动到指定的设备上(CPU或GPU)。Actor和Critic的优化器也被创建。
接下来,根据配置文件中的设置,创建不同类型的策略:
如果dyna_style为True,则创建一个Model网络和DynaTD3策略。
如果MPC为True,则创建一个Model网络和SMCPTD3策略。
如果safe_rl为True,则创建一个安全Critic网络和带有安全约束的TD3策略。
否则,创建一个标准的TD3策略。
最后,如果init_buffer为True,则创建一个ReplayBuffer对象。ReplayBuffer的大小和其他参数从配置文件中获取。
'''

def initialize_policy(config, env, init_buffer=True):
    training_config = config["training_config"]
    
    # 获取状态和动作的维度,以及动作空间的上下界
    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    device = "cpu" # "cuda:%d" %(devices[0]) if len(devices) > 0 else "cpu"
    print("    >>>> Running on device %s" %(device))

    # 创建状态编码器(encoder):
    # 首先从配置字典中获取编码器类型(encoder_type)和编码器参数(encoder_args),然后调用get_encoder函数创建对应类型的编码器实例。编码器将被用于处理原始的状态观测。
    encoder_type = training_config["encoder"]
    encoder_args = {
        'input_dim': state_dim[-1],  # np.prod(state_dim),
        'num_layers': training_config['encoder_num_layers'],
        'hidden_size': training_config['encoder_hidden_layer_size'],
        'history_length': config["env_config"]["stack_frame"],
    }

    '''
    创建了一个Actor网络,它由状态编码器(state_preprocess)和多层感知机(head)组成。action_dim指定了动作空间的维度。
    创建Actor网络后,函数会创建一个Adam优化器(actor_optim)来优化Actor网络的参数,学习率由actor_lr指定。最后,函数会打印出Actor网络的总参数数量。
    '''
    input_dim = training_config['hidden_layer_size']
    
    actor_class = GaussianActor if "SAC" in training_config["algorithm"] else Actor
    
    # actor = Actor(
    # actor = actor_class(
    #     #state_preprocess=state_preprocess,
    #     state_preprocess=get_encoder(encoder_type, encoder_args),
    #     head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
    #     #head=nn.Identity(),
    #     action_dim=action_dim
    # ).to(device)
    # actor_optim = torch.optim.Adam(
    #     actor.parameters(), 
    #     lr=training_config['actor_lr']
    # )
    # print("Total number of parameters: %d" %sum(p.numel() for p in actor.parameters()))
    
    # '''
    # 创建了一个Critic网络,它的结构与Actor网络类似,但输入维度需要加上动作空间的维度,因为Critic网络需要同时接收状态和动作作为输入。
    # 同样,函数会创建一个Adam优化器(critic_optim)来优化Critic网络的参数,学习率由critic_lr指定。
    # '''
    # input_dim += np.prod(action_dim)
    # critic = Critic(
    #     state_preprocess=get_encoder(encoder_type, encoder_args),
    #     head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
    #     #head=nn.Identity(),
    # ).to(device)
    # critic_optim = torch.optim.Adam(
    #     critic.parameters(), 
    #     lr=training_config['critic_lr']
    # )
    actor = actor_class(
        encoder=get_encoder(encoder_type, encoder_args),
        head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
        action_dim=action_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(), 
        lr=training_config['actor_lr']
    )
    # print("Total number of parameters: %d" %sum(p.numel() for p in actor.parameters()))

    # initialize critic
    input_dim += np.prod(action_dim)
    critic = Critic(
        encoder=get_encoder(encoder_type, encoder_args),
        head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
    ).to(device)
    critic_optim = torch.optim.Adam(
        critic.parameters(), 
        lr=training_config['critic_lr']
    )
    
    '''
    创建(policy):
    如果dyna_style为True,函数会创建一个Model网络和一个DynaTD3策略;如果MPC为True,函数会创建一个Model网络和一个SMCPTD3策略;
    如果safe_rl为True,函数会创建一个安全Critic网络(safe_critic)和一个带有安全约束的TD3策略;
    否则,函数会创建一个标准的TD3策略。策略的各种超参数和设置由配置字典指定。
    '''
    if training_config["dyna_style"]:
        model = Model(
            state_preprocess=get_encoder(encoder_type, encoder_args),
            head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
            state_dim=state_dim,
            deterministic=training_config['deterministic']
        ).to(device)
        model_optim = torch.optim.Adam(
            model.parameters(), 
            lr=training_config['model_lr']
        )
        policy = DynaTD3(
            model, model_optim,
            training_config["model_update_per_step"],
            training_config["n_simulated_update"],
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    elif training_config["MPC"]:
        model = Model(
            state_preprocess=get_encoder(encoder_type, encoder_args),
            head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
            state_dim=state_dim,
            deterministic=training_config['deterministic']
        ).to(device)
        model_optim = torch.optim.Adam(
            model.parameters(), 
            lr=training_config['model_lr']
        )
        policy = SMCPTD3(
            model, model_optim,
            training_config["horizon"],
            training_config["num_particle"],
            training_config["model_update_per_step"],
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    elif training_config["safe_rl"]:
        safe_critic = Critic(
            state_preprocess=get_encoder(encoder_type, encoder_args),
            head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
        ).to(device)
        safe_critic_optim = torch.optim.Adam(
            safe_critic.parameters(), 
            lr=training_config['critic_lr']
        )
        policy = TD3(
            actor, actor_optim, 
            critic, critic_optim, 
            action_range=[action_space_low, action_space_high],
            safe_critic=safe_critic, safe_critic_optim=safe_critic_optim,
            device=device,
            safe_lagr=training_config['safe_lagr'],
            safe_mode=training_config['safe_mode'],
            **training_config["policy_args"]
        )
    else:
        # policy = TD3(
        policy = SAC(
            actor, actor_optim, 
            critic, critic_optim, 
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    # initialize agents  
      
      
        
    '''
    如果init_buffer为True,函数会创建一个ReplayBuffer实例。缓冲区的大小由buffer_size指定,奖励归一化的设置由reward_norm指定。
    如果init_buffer为False,函数不会创建缓冲区,而是返回None。
    '''
    if init_buffer:
        try:
            config['env_config']["reward_norm"]
        except KeyError:
            config['env_config']["reward_norm"] = False
        buffer = ReplayBuffer(state_dim, action_dim, training_config['buffer_size'],
                            device=device, safe_rl=training_config["safe_rl"],
                            reward_norm=config['env_config']["reward_norm"])
    else:
        buffer = None

    return policy, buffer

'''
函数首先调用initialize_logging函数初始化日志记录器,并根据配置文件中的设置创建数据收集器(collector)。如果使用Condor进行分布式训练,则创建CondorCollector;否则,创建LocalCollector。
接下来,函数从配置文件中获取训练参数(training_args),并使用collector预先收集一定数量的经验数据(由pre_collect参数指定)。
然后,函数进入主训练循环,直到达到最大步数(max_step)。在每个循环中:
根据当前步数和配置文件中的参数,线性衰减探索噪声(exploration_noise)。
使用collector收集指定数量的经验数据(由collect_per_step参数指定),并更新步数(n_steps)、迭代次数(n_iter)和总episodes数(n_ep)。
将收集到的episodes信息(epinfo)存储到epinfo_buf中,并按照环境(world)进行分类存储到world_ep_buf中。
根据update_per_step参数指定的次数,从缓冲区中采样数据并更新策略,并将每次更新的损失信息存储到loss_infos列表中。
计算平均损失信息。
记录当前的各种指标,如平均回报、平均episode长度、成功率、碰撞率、fps等,并将这些指标存储到log字典中。
如果当前迭代次数是log_intervals的整数倍,则将log字典中的指标写入TensorBoard日志,并保存当前的策略参数。此外,还按照环境(world)分别记录平均回报等指标。
最后,如果使用Condor进行分布式训练,函数会删除缓冲区目录,以强制所有Actor任务停止;否则,函数会关闭环境。
'''

def train(env, policy, buffer, config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    print("    >>>> initialized logging")
    
    if env_config["use_condor"]:
        collector = CondorCollector(policy, env, buffer, config)
    else:
        collector = LocalCollector(policy, env, buffer)

    training_args = training_config["training_args"]
    print("    >>>> Pre-collect experience")
    collector.collect(n_steps=training_config['pre_collect'])
    print("    >>>> Start training")

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()
    
    while n_steps < training_args["max_step"]:
        # Linear decaying exploration noise from "start" -> "end"
        policy.exploration_noise = \
            - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
            *  n_steps / training_args["max_step"] + training_config["exploration_noise_start"]
        steps, epinfo = collector.collect(n_steps=training_args["collect_per_step"])
        
        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)
        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        # log字典还会包含当前迭代的策略损失信息。这些损失信息在训练循环中通过以下代码计算和记录
        loss_infos = []
        for _ in range(training_args["update_per_step"]):
            loss_info = policy.train(buffer, training_args["batch_size"])
            loss_infos.append(loss_info)

        loss_info = {}
        for k in loss_infos[0].keys():
            loss_info[k] = np.mean([li[k] for li in loss_infos if li[k] is not None])

        t1 = time.time()
        
        '''
        train函数会输出一个log,这个log包含了训练过程中的各种指标和信息。在每个训练循环中,这些指标会被记录下来,并定期地写入到TensorBoard日志和控制台中。
        log字典包含了以下内容:
        "Episode_return": 平均每个episode的总回报。
        "Episode_length": 平均每个episode的长度(步数)。
        "Success": 平均每个episode的成功率。
        "Time": 平均每个episode的运行时间。
        "Collision": 平均每个episode的碰撞次数。
        "fps": 训练的帧率,即每秒处理的环境步数。
        "n_episode": 到目前为止总共运行的episodes数量。
        "Steps": 到目前为止总共运行的环境步数。
        "Exploration_noise": 当前的探索噪声大小。
        '''
        
        log = {
            "Episode_return": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Success": np.mean([epinfo["success"] for epinfo in epinfo_buf]),
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise,
        }
        log.update(loss_info)
        print(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar('train/' + k, log[k], global_step=n_steps)
            policy.save(save_path, "last_policy")
            print("Logging to %s" %save_path)

            for k in world_ep_buf.keys():
                writer.add_scalar(k + "/Episode_return", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Success", np.mean([epinfo["success"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]), global_step=n_steps)


    if env_config["use_condor"]:
        BASE_PATH = os.getenv('BUFFER_PATH')
        shutil.rmtree(BASE_PATH, ignore_errors=True)  # a way to force all the actors to stop
    else:
        env.close()

if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description = 'Start condor training')
    parser.add_argument('--config_path', dest='config_path', default="../configs/config.ymal")
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    # CONFIG_PATH = args.config_path
    CONFIG_PATH = "/home/jidan/AVWorkSpace/e2e_jackal_ws/src/the-barn-challenge-e2e/end_to_end/td3/configs/config.yaml"
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    seed(config)
    print(">>>>>>>> Creating the environments")
    train_envs = initialize_envs(config)
    env = train_envs if config["env_config"]["use_condor"] else train_envs
    
    print(">>>>>>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env)
    print(">>>>>>>> Start training")
    train(train_envs, policy, buffer, config)
