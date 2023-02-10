The basics-Gridworld
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the example that everybody uses to start RL with. It is mandatory : )
Consider a 3x4 grid, the goal of the agent is to start from a position on the grid, and navigate its way to **end: +5**(and not at **end: -5**)

.. image:: ../resources/GridWorld.jpeg
   :width: 200px
   :alt: 3x4 Gridworld
   :align: center
   

In my implementation, I'm giving the agent a reward of -1 if it lands on any other position. 
People usually have a wall at (1,1) (0-indexed notation : ) ), that's up to you. You can put up a wall there too. 


Dynamic Programming
===================

1. **Iterative Policy evaluation:**

In common terms, given a policy, tell me how good it is. 
A state in grid world is the position on the grid. Let's say the policy we want to evaluate has only action that can be taken from a position. This is how the pseudocode would like: 

::

   delta = 0
   while True:
      for s in allStates: # Policy update loop
         cached_V = V[s]
         a = policy[s]
         s2, r = agent.move(s, a)
         V = r + gamma*V[s2]
         delta = max(delta, np.abs(V - cached_V))
      if delta < epsilon:
         break

We break as soon as the max change in one update loop is less than a small value, epsilon

2. **Policy Improvement:**

Great! So we now know how to evalue a policy. But the main goal of RL is to find out the best policy. 
This is pseudocode for policy improvement:

::

   while True:
      evaluatePolicy()
      isPolicyStable = True
      for s in allStates:
         actionAsPerCurrentPolicy = policy[s]

         # The next few lines will try to find the best action to take from current state. 
         values_list = {"L": float(-inf), "R": float(-inf), "D": float(-inf), "U": float(-inf)}
         for a in allPossibleActionsInState[s]:
            s2, r = agent.move(s, a)
            values_list[a] = r + gamma*V[s2]

         newAction = max(zip(values_list.values(), values_list.keys())) # Essentially argmax : )
         if newAction != actionAsPerCurrentPolicy:
            isPolicyStable = False
            currentPolicy[s] = newAction

      if isPolicyStable: break


2. **Monte Carlo**:

"*Sample mean is an estimate of true mean*"

.. math::
   V_\pi(s) = E[G_t|S_t=s] \approx \frac{1}N \sum_{i=1}^{N} G_i,s

Remember the recursive relationship to obtain G?

.. math::
   
   G(t) = r + \gamma*G(t+1)

This is what we're going to use to get the expected value of a state, s.

.. note::
   In an episode, if we only average over the rewards obtained after state s was visited for the first time then it is called First-visit Monte Carlo, and if in an episode, we average over the rewards obtained every time state s is visited we call it Every-visit Monte Carlo.


You play one episode of the game, collect states & rewards. 
Now work backwards i.e. from t = T-1 to 0. 
Using first-visit MC, we just average the returns like so:

::

   for iter in range(maxEpisodes):
      states, rewards = playEpisode(policy)

      T = len(states)
      G = 0
      for t in range(T - 1, -1, -1):
         s = states[t]
         r = rewards[t]
         G = r + GAMMA * G

         # First-visit MC
         if s not in states[:t]:
             returns[s].append(G)
             V[s] = np.mean(returns[s])

Instead of computing V[s], if you computed Q[s, a], you can actually complete Policy Improvement! 
Besides the states & rewards at every step like in policy evaluation, we also need actions taken at each stage.

Therefore, returns[s] becomes returns[s, a], and Q[s, a] can be found out by mean(returns[s, a]).
Finally, choosing the optimal action is a matter of taking argmax(Q[s, allPossibleActions])


::
   
   policy = # Initialize to a random policy
   for iter in range(maxEpisodes):
      states, actions, rewards = playEpisode(policy)

      T = len(states)
      G = 0
      for t in range(T - 1, -1, -1):
         s = states[t]
         r = rewards[t]
         a = actions[t]
         G = r + GAMMA * G

         # First-visit MC
         if s, a not in states[:t] & actions[:t:
             # Compute returns
             returns[s, a].append(G)
             # Compute Q[s, a]
             Q[s, a] = np.mean(returns[s, a])
             #Find optimal action
             best_a = argmax(Q[s, :])
             policy[s] = best_a

For action selection, I've tried `Reward based epsilon decay <https://aakash94.github.io/Reward-Based-Epsilon-Decay/>`_ & standard epsilon greedy. 
I encourage you to try whichever technique works for you. If you have something better, go for it! 

3. **TD learning**

DP & Monte Carlo = TD Learning.

DP:

.. math::
   
   V_\pi(s) = E_\pi[R_{t+1} + \gamma V_\pi(S_{t+1})|S_t = s]

MC:

.. math::
   
   V(s) = V_{N-1}(s) + \frac{1}{N}(G_N(s) + V_{N-1}(s))

TD:

.. math::
   
   V(s) = V(s) + \alpha(r + \gamma V(s') - V(s))

I hope you can see parallels. In TD learning, you have a constant step size instead of a varying step size based on N. 
G(s) of MC equation is replaced with:

.. math::
   r + \gamma V(s')


Let's talk about the most popular TD learning technique: **Q-learning**

The equation for V can be modified to Q and it'll look like so:

.. math::
   
   Q(s, a) = Q(s, a) + \alpha(r + \gamma Q(s', a') - Q(s, a))

The most important term in the above equation is *Q(s', a')*, specifically *a'*. 
Q-learning is known to be an off-policy technique i.e. the policy that the agent is trying to learn is different from the action agent takes.
So when agent reaches state, s', it will perform action, *a'*, based on epsilon-greedy, but the equation will still have a' = amax:

.. math::
   
   a'_{best} = argmax(Q(s', :))

Therefore, what the agent does: 

.. math::
   
   Q(s, a) = Q(s, a) + \alpha(r + \gamma Q(s', a') - Q(s, a))

What the agent is trying to learn:

.. math::
   
   Q(s, a) = Q(s, a) + \alpha(r + \gamma Q(s', a_{max}) - Q(s, a))

The pseudocode for this goes something like:


::

   env = # Grid world
   policy = # Initialize to a random policy
   Q = # Initialize Q
   for epochs in range(maxEpochs):
   s = # Random start state
      while episode != over:
         a = epsilon-greedy(s)
         s2, r = env.move(a)
         a2 = argmax(Q(s2, :))
         Q[(s, a)] = Q[(s, a)] + alpha * (r + gamma * Q[(s2, a2)] - Q[(s, a)])
         s = s2

   # Best policy learned by agent
   for s in nonTerminalStates:
      policy[s] = argmax(Q(s, :))