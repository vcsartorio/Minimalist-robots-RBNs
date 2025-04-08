# Description
This file contains the configuration scripts for ARGoS using the Kilobot plugin.

## Usage
If you want to make any change, do it in __kilobot_generic_controller.argos__:   
when the project is built, python scripts will create all other config scripts based on this one. Some values as alpha and rho will be changed by the scripts in the __generated_configs__, but the rest will be copied (like for instance a change of length or argos tick per second). the __kilobot_generic_controller_viz.argos__ is a copy of the other one, just adding the vizualisation for visual checking.   

## Generation
To see how they are generated, look at:  
- src/behavior/CMakeLists.txt --> This is where the set of parameters for generation is defined 
- scr/generate.configs.py
- generate_config_visual.py
