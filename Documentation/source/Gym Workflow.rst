Gym Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's focus on the `cartpole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ .

Understand the action space, observation space, the end goal, the terminating conditions for an episode is important. 

This is how you would run one episode of CartPole, wherein the agent just takes random actions

::

   env = gym.make('CartPole-v1')
   done = False
   obs = env.reset()
   while not done:
      a = env.action_space.sample()
      obs, r, done, _ = env.step(a)
      env.render()