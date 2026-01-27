## Observation space VS Human understanding
We need to understand how to define the observation space (the way the model sees our benchmark) and make it flexible enough to accommodate training of multiple types of agent architectures (not multi-agent environment). This is probably not trivial and need some consideration. Perhaps it means we still can code the game as we would for a human — but need to layer the observation space on top, it must be rich enough for learning, but also structured to accommodate different agent architectures.

Check out `.venv/lib/python3.10/site-packages/gymnasium/envs/classic_control` to start with for getting a grasp on how this is done, later we can check out `.venv/lib/python3.10/site-packages/gymnasium/envs/box2d/car_racing.py` because I think it uses a pixel-based observation space, because I had to set the policy for PPO to CnnPolicy. The Gymnasium bench that is most similar to our SynergyGrid should be `.venv/lib/python3.10/site-packages/gymnasium/envs/toy_text/cliffwalking.py`. It has -1 for every move and -100 for falling off the cliff. So it has different rewards compared to frozen lake that only has 1 for reaching the goal and 0 for everything else.

So...benches to check out:
- classic control:
    - cartpole: the most basic one in terms of discrete actions
    - mountain car: has short code, might be easy to understand
- box2d:
    - car racing: it has a pixel-based observation spoace, good for understanding how we can have multiple observation spaces (if possible)
- toy_text:
    - cliffwalking: because it the one that resembles out bench the most