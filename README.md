# Approximate Q-Learning for the Atari Breakout Game
## Kimberley Yu, Lily Zhang, Frances Ding
## CS182 Final Project

First, set up the OpenAI gym code by following the instructions in their documentation: https://gym.openai.com/docs.

Next, clone our repo at https://github.com/lhz1029/breakout-rl. You'll see several files, of which the following are important:

1. game.py. This file sets up the game infrastructure and runs the other two files below.

2. qlearningagent.py. This file contains our QLearner, QLearnerPlus, and QLearnerPlusLambda classes.

3. featureExtractor.py. This file contains our getFeature and getFeaturePlus methods, to extract features from our RGB image states.

Run our code: (python game.py) with the following optional arguments:

1. --version.
    Input either 1, 2, or 3. Defaults to 1.
    Version 1 uses our QLearner class with only the basic features of the ball's x and y positions and the paddle's x position. Version 2 uses QLearnerPlus with the added features of predicted trajectory and relative position of ball to paddle. Version 3 uses the features of QLearnerPlus, but uses the Watkins Q(lambda) algorithm.
2. --epsilon.
    Input a float between 0.0 to 1.0 to adjust the exploration rate of the epsilon-greedy algorithm. Defaults to 0.05.
3. --gamma.
    Input a float between 0.0 to 1.0 to adjust the discount factor of the Q-learning algorithm. Defaults to 0.99.
4. --alpha.
    Input a float between 0.0 to 1.0 to adjust the learning rate of the Q-learning algorithm. Defaults to 0.02.
5. --numtrain.
    Input an integer $>$ 1 to specify the number of training episodes. Defaults to 10000.
6. --lambd.
    If using version 3, input a float between 0.0 to 1.0 to adjust the decay rate of the eligibility traces. If not version 3, any input for this argument is ignored. Defaults to 0.9.
The order of arguments does not matter.

An example command is as follows:
```
python game.py --version 3 --epsilon 0.9 --numtrain 100000 --lambd 0.8
```

As it runs, the file will output the number of timesteps and rewards for each episode. An example line of output is:
```
Episode finished after 218 timesteps with 1.0 reward
```

At the end, our code will output 3 graphs:

1. Number of time steps per episode vs. Episode number.
This shows how the length of each episode changes with further training. The file will be named similar to "lengths-v1-e0.05-g0.99-a0.02-n10000-l0.9.png", where each numerical value corresponds to the given arguments (where v=version, e=epsilon, g=gamma, a=alpha, n=numtrain, and l=lambda).
2. Rewards received vs. Episode number.
This shows how the amount of reward received changes with further training. The file will be named similar to "rewards-v1-e0.05-g0.99-a0.02-n10000-l0.9.png".
3. Feature weights vs. Episode number.
This shows how feature weights change with further training. The file will be named similar to "weights-v1-e0.05-g0.99-a0.02-n10000-l0.9.png".
