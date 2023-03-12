Atari Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
========================================
.. note::
	I have not had any luck with installing this on Windows, unfortunately. It had some CMake issues, which I resolved, but then I ran into some dependency issues with libtorrent. I will add the steps for when I actually get around to figuring that mess out. Until then Linux it is!

All you need to do is run the following:

- pip install gym[atari,accept-rom-license]


The reason for choosing this environment is because we are going to start looking at some seminal papers of RL. Starting with what I like to call



`The Experience replay paper  <https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf>`_ 
================================================================================================================================

I focus only on the `pong environment <https://gymnasium.farama.org/environments/atari/pong/>`_ for now.

Important takeaways from the paper are the following:

1. All frames must be preprocessed before handling them. The preprocessing steps involved are:
	- To handle Atari environment's flickering problem, store the max of every pixel between current & previous frame across RGB channels.
	- Convert to luma only. 
	- Downscale to (84, 84)
	- The input to the agent will a batch of 4 frames of size (84, 84) resulting in a tensor of shape (84, 84, 4)

2. Actions taken by the agent will be replicated for 4 frames.

3. An experience replay buffer must be implemented to account for the highly correlated samples, if taken from (say) one episode. This way, you could just
randomly sample a batch of (say) 32 from this buffer. This will result in uncorrelated samples.

4. Two networks will be maintained, one for the agent & one for the target. The agent network's weights will be updated every step, but the target network will 
be updated every 'C' steps. In my implementation, C was kept at 1000. This is done to avoid chasing a moving target.

5. Epsilon annealing is implemented to decay smoothly from 0.9 to 0.01.

6. Additionally, I made sure to create environment with *full_action_space*=False, which restricts the number of actions to 6.

::

   env = gym.make("PongNoFrameskip-v4", full_action_space=False)


To avoid any unforseen crashes during training, I dump the learned model every 5000-10000 steps along with the replay buffer.

- FYI, *gym.Wrapper* exists. So, you could inherit from this class & override implementations of important gym functions like *step*.

- I modified the reward slightly: gym, by default, gives 0 reward for every step taken and +1 if the player wins & -1 if player loses. 
- I didn't like this very much, since there is a chance of the agent learning to do nothing since all it receives is 0 until it wins. To encourage some exploration, I give a small negative reward(-0.2) for every action taken & +5 if agent wins, while -5 if agent loses.

