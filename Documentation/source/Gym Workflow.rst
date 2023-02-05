Gym Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We should first try to understand the gym workflow first, so let's focus on the `cartpole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ .

Understand the action space, observation space, the end goal, the terminating conditions for an episode. 
To get our hands dirty with the gym workflow, let's run one episode of CartPole. 
Please try to understand what each line of the following code means:
::

   env = gym.make('CartPole-v1')
   done = False
   obs = env.reset()
   while not done:
      a = env.action_space.sample()
      obs, r, done, _ = env.step(a)
      env.render()