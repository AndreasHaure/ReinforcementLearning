# Lunar Lander with keras-rl

I have trained an agent that runs the [Deep Q-Learning](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47) algorithm (DQNAgent from `keras-rl`) to learn the Lunar Lander reinforcement environment from Open AI Gym.

The brains of the agent is a deep neural network with three fully-connected hidden layers. The neural net is implemented in Keras, which is
the logical choice in combination with the `keras-rl` library for reinforcement learning.

Dependencies:

- Python3
- OpenAI Gym: `pip install gym`
- Tensorflow: `pip install tensorflow`
- Keras: `pip install keras`
- Keras RL: `pip install keras-rl`
- Box2D (I think): `pip install box2d-py`
- h5py (I think): `pip install h5py`

I'm not 100% sure of all the dependencies, but it's in the ballpark.

## Run the pretrained agent

You can test the pretrained agent (and your own agent if you ran the trained one) with:

```
python lunar-dqn-test.py
```

This script first runs an untrained agent for 5 episodes and then the pretrained agent for 5 episodes. The purpose is that you can see the difference before and after training.

Here is a video that shows the expected output (first sequence untrained, second sequence after training).

<iframe width="560" height="315" src="https://www.youtube.com/embed/Nc1EsqCuUsE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The test scripts automatically tries to load a weight weights file called `dqn_lunarlander-v2_weights.m5f`, which will be produced by the training script (see below). If that fails, the script loads a weights file called `dqn_lunarlander-v2_weights_20190321.m5f` that should already exist in this repository. The weights file contains weights for the deep neural network mentioned above.

## Train the agent from scratch

You can also train the agent from scratch. You tell the script how many decision you will train the agent on, for example 10000 decisions.

```
python lunar-dqn-train.py 10000
```

The output of the script is a file called `dqn_lunarlander-v2_weights.m5f`. If this file exists, then the test script (see above) will use those weights instead of the default.

If you repeatedly run the training script, the script will first load the previous weights (if they exist) and then resume training from that point.
