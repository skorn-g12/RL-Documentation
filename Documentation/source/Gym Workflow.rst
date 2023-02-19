Gym Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I'll focus on the `Cartpole environment <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ first.

.. image:: ../resources/cart_pole.gif
   :width: 300px
   :alt: Mountain Car
   :align: center

It only has two actions: *go left* & *go right*. The goal is to balance the pole on the cart with just these actions. 

Here's the code for running one episode of cart pole, where the agent takes random actions(No RL yet)
::

   env = gym.make('CartPole-v1')
   done = False
   obs = env.reset()
   while not done:
      a = env.action_space.sample()
      obs, r, done, _ = env.step(a)
      env.render()

Let's apply basic RL on this
========================================
It is not wise to maintain a Q table for every state-action pair. The number of states is practically infinite in most cases. 
So we approximate the Q function with a model. 
I'm going to begin with modeling the Q function with a linear model:

.. math::
   Q(s, a)  = W^T x

where x is the concatenated array of s & a.

We could create our own model class that does this & implements the weight update equation:

.. math::
   W = W - \alpha\frac{\partial J}{\partial W} X

   and, J = \frac{1}2 (y- \hat{y})^{2}

This is indeed a straightforward approach & I encourage you to try it out on your own. I've done it & it works just fine. 
But let's try to use sklearn's SGDRegressor instead. The important thing to note here is that we want to use *partial_fit()* instead *fit()*, because we only want to do a single-step update as this is an online learning technique. 

.. note::
   Q function is non-linear, by nature. Therefore, we cannot directly apply a linear model like this. We will use some feature transformer like RBFSampler() or Nystroem() from sklearn.
   Initiially, we will play a bunch of episodes wherein the agent takes random actions and store the state-action pairs encountered in a list. This list will be used to fit the featurizer. 

- Here's the pseudocode for setting up the featurizer & model:

::

   # Featurizer
   featurizer = RBFSampler()
   samples = runRandomEpisodes()
   featurizer.fit(samples)

   # Model
   model = SGDRegressor(max_iter=1, tol=None, learning_rate="constant", eta0=1e-2, fit_intercept=True)
   model.coef_ = np.zeros(self.nWeights)
   model.intercept_ = np.atleast_1d(np.random.rand())

.. note::
   SGBRegressor has max_iter=1, because I want to do a single step update. You can experiment with different learning rates & eta0 to get better results.
   sklearn's fit() wants an array for the intercept as well. Hence you see np.atleast_1d()

- The process of computing Q for any state-action pair involves the following process:

::

   sa = getConcatStateActionPair(s, a)
   # Change of basis
   phi_sa = featurizer.transform(sa) 
   # Compute Q[sa]
   model.predict(phi_sa)

- Finally, putting all the pieces together, the convergence loop is as follows:

::

   s = env.reset()
   while not gameOver:
      # Choose an action & move
      a = epsilonGreedy(s)
      s2, r, terminated, truncated, info = env.step(a)
      gameOver = terminated or truncated
      Q_sa = computeQsa[s, a]

      if gameOver:
         target = r
      else:
         # Get best action from s2: a_max
         a_max = getBestAction(s2)
         Q_s2amax= computeQsa[s2, a_max]
         target = r + GAMMA*Q_s2amax

      # Weight update for the model
      target = np.atleast_1d(target)
      model.partial_fit(phi_sa, target)

      # Update s
      s = s2

