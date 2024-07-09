"""
    © 2023 This work is licensed under a CC-BY-NC-SA license.
    Title: Towards biologically plausible Dreaming and Planning in recurrent spiking networks
    Authors: Anonymus
"""

import os
import gym
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import trange
from argparse import ArgumentParser
from src.functions import plot_rewards, plot_dram, plot_dynamics
from src.functions import plot_planning
from src.optimizer import Adam
from src.agent import AGEMO
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from examples.control.hitting_agent import build_agent
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from examples.control.defending_agent import DefendingAgent
#if_dream = 0
import threading
from time import sleep, time

parser = ArgumentParser()
parser.add_argument('-if_dream', required = False,  type = int, dest = 'if_dream', help = 'to dream or not to dream', default = 0)
parser.add_argument('--env', required=False,  type = str, dest = 'env', help = 'Environment to use. Options: \'pong\', \'airhockey\'. Default: \'airhockey\'', default = 'airhockey')
par_inp = vars(parser.parse_args())

if_dream = par_inp['if_dream']
current_time = datetime.now()
folder_name = current_time.strftime("RUN %d-%m_%H-%M-%S")
folder = folder_name

start_learn = 1*50

# Check whether the specified path exists or not
isExist = os.path.exists(folder)
if not isExist:
  # Create a new directory because it does not exist
  os.makedirs(folder)
  print("The new directory is created!")

act_factor = 5.


def agent_spike_printer(spike_train, folder, iteration):
    
    indices = np.arange(len(spike_train))
    values = spike_train.astype(int)
    plt.figure(figsize=(10, 4))
    plt.scatter(indices, values, color='blue')
    plt.xlabel('Indice')
    plt.ylabel('Valore')
    plt.title('Agent spikes')
    plt.ylim(-0.1, 1.1)

    filename = f"agent_{iteration}.png"
    filepath = os.path.join(folder, filename)
    

    plt.savefig(filepath)
    plt.close()
    
def planner_spike_printer(spike_train, folder, iteration):
    
    indices = np.arange(len(spike_train))
    values = spike_train.astype(int)
    plt.figure(figsize=(10, 4))
    plt.scatter(indices, values, color='red')
    plt.xlabel('Indice')
    plt.ylabel('Valore')
    plt.title('Planner spikes')
    plt.ylim(-0.1, 1.1)
    
    filename = f"planner_{iteration}.png"
    filepath = os.path.join(folder, filename)
    

    plt.savefig(filepath)
    plt.close()

def env_step(env, action):
    try:
        ram_all, r, done, _, _ = env.step (action) 
        
    except ValueError:
        ram_all, r, done, _ = env.step (action) 
    # print(env.base_env.get_joints(ram_all))      
    ram = import_ram(ram_all)
    return ram, r, done

        

for repetitions in range(10):

    N_ITER =   10000                                #50*40
    TIMETOT = 100

    if par_inp['env'] == 'pong':
        env = gym.make('Pong-ramDeterministic-v4', difficulty = 0, render_mode="human")
        from src.functions import import_ram_pong as import_ram
        from src.functions import get_dummy_action_pong as get_dummy_action
        from src.functions import cat2act_pong as cat2act
        from src.config import PONG_V4_PAR_I4 as par
    elif par_inp['env'] == 'airhockey':
        env = AirHockeyChallengeWrapper(env="3dof-defend", interpolation_order=3, debug=True)
        from src.functions import import_ram_airhockey as import_ram
        from src.functions import get_dummy_action_airhockey as get_dummy_action
        from src.functions import cat2act_airhockey as cat2act
        from src.config import AIRHOCKEY as par
    else:
        raise AttributeError(f'The value {par_inp["env"]} is not a valid option for the environment')
        
    
    rendering = False

    def render_thread():
        while True:
            if rendering:
                now = time()
                env.render()
                sleep(0.03 % now)
            else:
                sleep(0.5)
                
    threading.Thread(target=render_thread).start()

    # print (f'Pong: Observation space: {env.observation_space}')
    # print (f'Pong: Action Meaning: {env.unwrapped.get_action_meanings()}')


    plt.rcParams.update({'font.size': 14})

    train_par = {'epochs'    : par['epochs'],
                 'epochs_out' : par['epochs_out'],
                 'clump'   : par['clump'], 'feedback'  : par['feedback'],
                 'verbose' : par['verbose'], 'rank' : par['rank']}
    
    par["I"] = 6

    agent = AGEMO(par)
    
    par["I"] = 8    #par['O'] + par['I'] #TODO: input del planner, il modello prende come input stato (4) e azione (3) e da' come output next state e rew
    
    par["tau_ro"] = 2.*par["dt"]
    planner = AGEMO(par) #TODO:planner=rete modello, agent=interazione con environment

    alpha_rout = agent.par['alpha_rout']

    plt.figure() 

    # Erase both the Jrec and the Jout
    agent.forget()
    # Reset agent internal variables
    agent.reset()
    
    robot=DefendingAgent(env.base_env.env_info)
    # robot.reset()
    
    count = -1
    agent.Jout = np.random.normal(0,.1,size=(agent.O,agent.N)) 
    agent.J = np.random.normal(0,1./np.sqrt(agent.N),size=(agent.N,agent.N))#*=0    #TODO: J pesi ricorrenti (se messi a 0 diventa un layer tradizionale di una rete feedforward), Jout output

    agent.adam_rec = Adam (alpha = 0.001, drop = .99, drop_time = 10000) #TODO: Adam viene creato per ogni set di parametri, serve per ottenere i gradienti e fare gli step
    agent.adam_out = Adam (alpha = 0.001, drop = .99, drop_time = 10000)

    eta_factor_r = 0.2
    planner.adam_out_s = Adam (alpha = 0.002, drop = .99, drop_time = 10000) #0.01
    planner.adam_out_r = Adam (alpha = 0.002*eta_factor_r, drop = .99, drop_time = 10000) #0.005
    planner.adam_rec = Adam (alpha = 0.004, drop = .99, drop_time = 10000)

    planner.Jout = np.random.normal(0,.1,size=(agent.O,agent.N))
    planner.J = np.random.normal(0,1./np.sqrt(agent.N),size=(agent.N,agent.N))#*=0
    planner.Jout_s_pred = np.zeros((agent.I,agent.N)) #readout del planner planner.I= stato, planner.N = taglia della rete
    planner.Jout_r_pred = np.zeros((1,agent.N)) 

    planner.dJ_aggregate = 0
    planner.dJout_s_aggregate = 0 
    planner.dJout_r_aggregate = 0 

    REWARDS = []
    REWARDS_MEAN = []
    REWARDS_STANDARD_MEAN = []
    ENTROPY = []

    ERROR_RAM = []
    ERROR_R = []

    MEAN_ERROR_RAM = []
    MEAN_ERROR_R = []


    S = []

    agent.dJ_aggregate=0
    agent.dJout_aggregate=0
    planner.state = 0

    for iteration in trange(N_ITER): #TODO: N_ITER=2000, numero di partite che durano 100 step 
        rendering = False

        initial_obs=env.reset() #initial obs is a 12 dimensional obj with puck and EE initial pos and vel 
        agent.reset()
        planner.reset()
        

        S_planner = []
        S_agent = []

        R = []
        R_PRED = []

        RAM = []
        RAM_PRED = []
        DRAM_PRED = []
        DRAM = []

        PLANNER_STATES = []

        agent.dH = np.zeros (agent.N) #TODO: traccia pre sinaptica per il calcolo del gradiente
        planner.dH = np.zeros (agent.N)

        RTOT = 0
        vx_old = 0

        agent.dJfilt =0
        agent.dJfilt_out = 0
        
        home_ee = robot.get_ee_pose(initial_obs)
        home_joint_pos = robot.get_joint_pos(initial_obs)
        
        ram, r, done = env_step(env, get_dummy_action(robot, initial_obs))
    
        ram_old = ram
    



        ######### AWAKE PHASE ########## during the awale phase the TWO networks interact with the environment

        for skip in range(1): 
            act_vec = np.zeros((par['O'],))
            act_vec = act_vec*0  #TODO: forcing the agent not to act for 20 steps?
            act_vec[0]=1 #TODO: why having a vector of 0s and then add 1?
            
            _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )  #TODO: act_vec*act_factor ? why performing this product?
            
            
            #2.2 Learning the world model
            ds_pred,r_pred = planner.prediction() #TODO: predicted state variation and the reward? prediction:paddles (y, yopponent) and ball coordinates?
            planner.state = ds_pred + ram 

            S_planner.append(planner.S[:]) #.S spikes train
            S_agent.append(agent.S[:])

            ram_old = ram
            
            ram, r, done = env_step(env, get_dummy_action(robot, initial_obs))

            PLANNER_STATES.append( planner.state_out )
            RAM_PRED.append( planner.state )
            RAM.append( ram )

            R += [r]
            R_PRED += [r_pred]

            dram = ram - ram_old
            dram[np.abs(dram)>30]=0.

            planner.learn_model(ds_pred,r_pred,dram,r)  #TODO: dreaming network learning the model 
            planner.model_update() #TODO: agent.py @550

            DRAM.append(dram)
            DRAM_PRED.append(ds_pred)

        frame = 0
        ifplot = 1
        entropy=0

        OUT = []

        r_learn = 0

        while not done and frame<TIMETOT:
            rendering = False
            # if iteration % 100 == 0 and iteration > 0:
            #     rendering = True
                
            frame += 1
            ram_old = ram
            action, out = agent.step_det(ram/255) 
            act_vec = np.copy(out)

            act_vec = act_vec*0
            act_vec[action]=1

            _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) ) #TODO: why does the step of the planner needs two inputs ?
            ds_pred,r_pred = planner.prediction()

            PLANNER_STATES.append( planner.state_out )

            S_planner.append(planner.S[:])
            S_agent.append(agent.S[:])
            
            planner_spikes = planner.S[:]
            agent_spikes = agent.S[:]
            
            # if iteration % 200 == 0:
            #     planner_spike_printer(planner_spikes, folder, iteration)
            #     agent_spike_printer(agent_spikes, folder, iteration)

            
            ram, r, done = env_step(env, cat2act(action, initial_obs, robot, frame))
            
            if_learn=0
            if iteration > start_learn: #TODO: why do I wait for learning?
                if_learn=1

            agent.learn_error(r*if_learn) 


            entropy+=agent.entropy

            dram = ram-ram_old
            dram[np.abs(dram)>30]=0.

            r_learn = r_learn*.5 + r

            planner.learn_model(ds_pred,r_pred,dram,r_learn)
            planner.model_update()

            planner.state = ram_old+ds_pred

            RAM_PRED.append(planner.state)
            RAM.append(ram)

            OUT.append(out)

            RTOT +=r
                
            R += [r]
            R_PRED += [r_pred]
            DRAM_PRED.append(ds_pred)
            DRAM.append(dram)

        try:
            env.close()
        except AttributeError:
            pass
        REWARDS.append(RTOT)
        ENTROPY.append(entropy)
        ERROR_RAM.append(np.std(np.array(DRAM)-np.array(DRAM_PRED),axis=0))
        ERROR_R.append( np.std( np.array(R)-np.array(R_PRED) ) )


        if (iteration%1==0)&(iteration>0):
            agent.update_J(r)

        if (iteration%10==0)&(iteration>0):

            REWARDS_MEAN.append(np.mean(REWARDS[-50:]))
            plot_rewards(REWARDS,REWARDS_MEAN,S_agent,OUT,RAM,RAM_PRED,R,R_PRED,ENTROPY,filename = os.path.join(folder, 'rewards_dynamics_r0_initrand_aggr_ifdream_' + str(if_dream) + '.png') )
            np.save(os.path.join(folder,"rewards_" + str(repetitions) + "if_dream_" + str(if_dream) + ".npy"), REWARDS_MEAN)
            
            plot_dynamics(REWARDS,REWARDS_MEAN, S_agent,OUT,RAM,RAM_PRED,R,R_PRED,ENTROPY,filename = os.path.join(folder, 'NetworkDynamics_ifdream_' + str(if_dream) + '_' + str(repetitions) +'.png') )
            np.save(os.path.join(folder,"dynamics_" + str(repetitions) + "if_dream_" + str(if_dream) + ".npy"), REWARDS_MEAN)
            

            MEAN_ERROR_RAM.append(np.mean(np.array(ERROR_RAM)[-50:,:],axis=0))
            MEAN_ERROR_R.append(np.mean(np.array(ERROR_R)[-50:]))

            plot_dram(DRAM,DRAM_PRED,R,R_PRED,MEAN_ERROR_RAM,MEAN_ERROR_R,filename= os.path.join(folder, 'planning_dram_fit.png'))

        ######### DREAMING PHASE ##########
        rendering = False
        plot_dream_every = 50

        for dream_times in range(if_dream):

            RAM_PLAN = []
            REWS_PLAN = []
            S_agent = []
            S_planner = []

            env.reset()
            agent.reset()
            planner.reset()

            ram_all, r, done, _, _ = env.step (0)
            ram = import_ram(ram_all)
            t_skip = 20

            for skip in range(t_skip):

                ram_all, r, done, _, _ = env.step (0)
                RAM_PLAN.append(ram_all[[49, 50, 51, 54]])

                act_vec = np.copy(out)
                act_vec = act_vec*0
                act_vec[0]=1

                _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
                ds_pred,r_pred = planner.prediction()
                ram = import_ram(ram_all)
                REWS_PLAN.append(r_pred)

                S_planner.append(planner.S[:])
                S_agent.append(agent.S[:])

            time_dream = 50

            for plannng_steps in range(time_dream):
                agent.dH = np.zeros (agent.N)
                planner.dH = np.zeros (agent.N)

                action, out = agent.step_det(ram/255)

                act_vec = np.copy(out)
                act_vec = act_vec*0
                act_vec[action]=1

                _, _ =  planner.step_det( np.concatenate((act_vec*act_factor, ram/255), axis=0) )
                ds_pred,r_pred = planner.prediction()

                S_planner.append(planner.S[:])
                S_agent.append(agent.S[:])

                ram = ram + ds_pred

                if_learn=0
                if iteration > start_learn:
                    if_learn=1

                agent.learn_error(r_pred*if_learn)

                RAM_PLAN.append(ram)
                REWS_PLAN.append(r_pred)
            agent.update_J(r_pred)

            if (iteration%50==0)&(dream_times==0):
                plot_planning(REWS_PLAN,R,RAM_PLAN,RAM,S_agent,S_planner,t_skip,filename = os.path.join(folder, 'planning.png'))

    agent.save (os.path.join(folder,'model_PG_out_r0_initrand_aggr' + str(if_dream) + '_60''.py') )
