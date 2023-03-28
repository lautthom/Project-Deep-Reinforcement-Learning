# Project Deep Reinforcement Learning: DQN, DDQN, and Dueling Networks

The code is written in and tested for `Python 3.9`, the required packages are listed in `requirements.txt`.

Clone the repository to the current directory of your machine to run the code:
```
git clone GITHUB LINK
```

It is recommended to set up and use a virtual environment to run the code: 
```
# pip install virtualenv
cd <path of DQN directory>
python -m virtualenv env
.\env\Scripts\activate
```

After that the required packages can be installed:
```
pip install -r requirements.txt
```

Then the code can be run.  
`atari_dueling_ddqn.py` is the main implementation containing the DQN, DDQN, and Dueling Networks implementation,  
`make_statistics.py` is the code used for the statistical evaluation  
```
python atari_dueling_ddqn.py
python make_statistics.py
```

For the `atari_dueling_ddqn.py` different options in the command line can be used:  

`-t` or `--training` if argument is given, model will be trained, otherwise an already trained model will be loaded  
`-dqn`    if argument is given, DQN will be used as algorithm, otherwise DDQN will be used
`-sn` or `--single_network`                   if argument is given, single stream network is used, otherwise dueling network is used  
`-g` or `--game` choose game that is played; game must be given as a string, it is case-sensitive; an overview of the available games is given at: https://www.gymlibrary.dev/environments/atari/#; default game is "Pong"  
`-a` or `--adam` if argument is given, Adam is used as an optimizer
`-tf` or `--training_frames` choose the number of frames, that is used for training; default is 5,000,000 frames  
`-b` or `--batch_size` choose the batch size, that is used for training; default is 32  
`-r` or `--replay_size` choose the replay memory size, that is used for training; default is 100,000  
`-u` or `--update_frequency` choose the frequency, with which the policy network is updated; default is every 4 frames  
`-tu` or `--target_update` choose the frequency, with which the target network is updated; default is every 1,000 frames  
`-lr` or `--learning_rate`  choose the learning rate, that is used for training; default is 0.001  
`-ie` or `--initial_exploration` choose the initial exploration rate; default is 1  
`-fe` or `--final_exploration` choose the final exploration rate; default is 0.02  
`-ff` or `--final_exploration_frame` choose the final exploration frame, after which the final exploration rate is used; default is a 10th of total training frames  

This means, that Deep SAD using MNIST, the `standard` mode, with the weight `3`, the `0` category as normal class, the `1` category as anomaly class, a labeled 
anomaly ratio of `0.05`, no labeled normal data, a pollution of `0.1` in the unlabeled data, and no pollution in the labeled anomalies, can for example be run by using:
```
python DeepSAD.py
```
Deep SAD using CIFAR-10, the `extended` mode, with the weight `2` for labeled normal data, and a secondary weight `4` for labeled anomalies, 
the `1` category as normal class, the `2` category as anomaly class, a labeled anomaly ratio of `0.01`, a labeled normal data ratio of `0.1`, 
a pollution of `0.1` in the unlabeled data, and a pollution of `0.01` in the labeled anomalies, can for example be run by using:
```
python DeepSAD.py -d "cifar10" -m "extended" -w 2 -sw 4 -cn 1 -ca 2 -ra 0.01 -rn 0.1 -rpu 0.1 -rpl 0.01
```

