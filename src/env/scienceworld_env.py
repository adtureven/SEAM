"""ScienceWorld environment import path.

The implementation lives in sciworld_env.py for backward compatibility with
older local scripts; new code should import ScienceWorldEnv from this module.
"""

from env.sciworld_env import SciWorldEnv, SciWorldTaskInfo


ScienceWorldEnv = SciWorldEnv
ScienceWorldTaskInfo = SciWorldTaskInfo

