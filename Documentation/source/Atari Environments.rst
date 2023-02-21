Atari Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
========================================
.. note::
	I have not had any luck with installing this on Windows, unfortunately. It had some CMake issues, which I resolved, but then I ran into some dependency issues with libtorrent. I will add the steps for when I actually get around to figuring that mess out. Until then Linux it is!

All you need to do is run the following:
- pip install gym[atari,accept-rom-license]


The reason for choosing this environment is because we are going to start looking at some seminal papers of RL. Starting with what I like to call


`The Experience replay paper  <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ 
========================================================================================================================
