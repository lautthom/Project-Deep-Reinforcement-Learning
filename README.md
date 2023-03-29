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
  
The trained models used for testing are provided in the `models` folder, the results of the test runs can be found in the `results` folder, the figures that are created by `make_statistics.py` are saved in the `figures` folder.
Note, that training a new model, running an evaluation or running `make_statistics.py` for one of the already provided combinations of game, algorithm (dqn or ddqn), and network structure (single or dueling) will overwrite the corresponding files.
  
There are different options for `atari_dueling_ddqn.py` that can be used in the command line:  

`-t` or `--training` if argument is given, a new model will be trained, otherwise an already trained model will be loaded from the `model` folder, if one is available for the chosen combination of game, algorithm, and network structure  
`-dqn`    if argument is given, DQN will be used as algorithm, otherwise DDQN will be used
`-sn` or `--single_network`                   if argument is given, single stream network is used, otherwise dueling network is used  
`-g` or `--game` choose game that is played; game must be given as a string, it is case-sensitive; an overview of the available games is given at: https://www.gymlibrary.dev/environments/atari/#; default game is "Pong"  
`-a` or `--adam` if argument is given, Adam is used as an optimizer; otherwise RMSprop is used
`-tf` or `--training_frames` choose the number of frames, that is used for training; default is 5,000,000 frames  
`-b` or `--batch_size` choose the batch size, that is used for training; default is 32  
`-r` or `--replay_size` choose the replay memory size, that is used for training; default is 100,000  
`-u` or `--update_frequency` choose the frequency, with which the policy network is updated; default is every 4 frames  
`-tu` or `--target_update` choose the frequency, with which the target network is updated; default is every 1,000 frames  
`-lr` or `--learning_rate`  choose the learning rate, that is used for training; default is 0.001  
`-ie` or `--initial_exploration` choose the initial exploration rate; default is 1  
`-fe` or `--final_exploration` choose the final exploration rate; default is 0.02  
`-ff` or `--final_exploration_frame` choose the final exploration frame, after which the final exploration rate is used; default is a 10th of total training frames  

An evaluation run using an already trained model for "Pong", which was trained using DDQN and a dueling network, can for example be started using:
```
python atari_dueling_ddqn.py
```
If a new model should be trained, playing "Breakout" and trained using DQN and a single network, using Adam as an optimizer, with 10,000,000 training frames, a batch size of 64, a replay memory size of 250,000, an update frequency of every 8 frames, target network updates every 5,000 frames, using a learning rate of 0.0005, an initial exploration of 0.5, a final exploration of 0.01, and the final exploration frame should be the 10,000,000th frame, this can be done using:
```
python atari_dueling_ddqn.py -t -dqn -sn -g "Breakout" -a -tf 10000000 -b 64 -r 250000 -u 8 -tu 5000 -lr 0.0005 -ie 0.5 -fe 0.01 -ff 10000000
```

