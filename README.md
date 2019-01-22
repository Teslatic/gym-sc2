# gym-sc2
Our gym starcraft 2 environment/wrapper for the pysc2 library

Setup:
1. clone repo to desired location
2. ``` cd gym-sc2/ ```
3. ``` pip install -e .```

Import in python script:

```
import gym
import gym_sc2
```

References:
General: https://github.com/openai/gym/tree/master/gym/envs
Example repo: https://github.com/openai/gym-soccer

Deinstallation:
```
pip uninstall gym-sc2
```

Also remove egg info directory in cloned repository (was created during installation)
```
rm -rf gym_sc2.egg-info/
```
