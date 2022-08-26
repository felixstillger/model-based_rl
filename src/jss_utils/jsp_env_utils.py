import random

import numpy as np

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
import jss_utils.jsp_custom_instance_generator as jsp_gen

from typing import Union

from stable_baselines3.common.vec_env import VecEnv

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jsp_instance_downloader import download_instances
from jss_utils.jss_logger import log


def get_benchmark_instance_and_lower_bound(name: str) -> (np.array, float):
    jsp_instance_details = details.get_jps_instance_details(name)
    jps_instance = parser.get_instance_by_name(name)
    return jps_instance, jsp_instance_details["lower_bound"]


def get_benchmark_instance_and_details(name: str) -> (np.array, float):
    jsp_instance_details = details.get_jps_instance_details(name)
    jps_instance = parser.get_instance_by_name(name)
    return jps_instance, jsp_instance_details


def get_pre_configured_example_env(name="ft06", **kwargs) -> DisjunctiveGraphJssEnv:
    env = DisjunctiveGraphJssEnv(**kwargs)
    return load_benchmark_instance_to_environment(env=env, name=name)


def load_benchmark_instance_to_environment(env: Union[DisjunctiveGraphJssEnv, VecEnv, None] = None, name: str = None) \
        -> Union[DisjunctiveGraphJssEnv, VecEnv]:
    if name is None:
        name = "ft06"
        log.info(f"no benchmark is specified. Usinig '{name}' as a fallback.")

    all_instance_details_dict = details.parse_instance_details()

    if name not in all_instance_details_dict.keys():
        error_msg = f"the instance {name} is not present in the details dict. " \
                    f"you might need to download the all benchmark instance and download benchmark details first. " \
                    f"try to run the 'jss_utils.jsp_instance_downloader' script. " \
                    f"And then the 'jss_utils.jsp_instance_details' script. "
        log.error(error_msg)
        raise RuntimeError(error_msg)

    jsp_instance_details = all_instance_details_dict[name]
    jps_instance = parser.get_instance_by_name(name)

    if env is None:
        log.info("no environment is specified. Creating a blank environment with default parameters")
        env = DisjunctiveGraphJssEnv()

    if isinstance(env, DisjunctiveGraphJssEnv):
        log.info(f"loading instance '{name}' into the environment.")
        env.load_instance(
            jsp_instance=jps_instance,
            scaling_divisor=jsp_instance_details["lower_bound"]
        )
        return env
    elif isinstance(env, VecEnv):
        raise NotImplementedError()
    else:
        error_msg = f"the specified environment type ({type(env)}) is not supported."
        log.error(error_msg)
        raise ValueError(error_msg)


def get_random_custom_instance_and_details_and_name(n_jobs: int, n_machines: int) -> (np.ndarray, dict, str):
    dir = PATHS.JSP_INSTANCES_CUSTOM_PATH.joinpath(f"{n_jobs}x{n_machines}")

    if not dir.exists():
        log.info(f"there are no custom instances of size ({n_jobs},{n_machines}) in the resource directory. "
                 f"Therefore some will be generated.")
        jsp_gen.generate_jsp_instances(n_jobs=n_jobs, n_machines=n_machines)

    instance = random.choice([*dir.glob('*.txt')])

    jsp, _ = parser.parse_jps_taillard_specification(instance_path=instance)
    name = instance.stem
    custom_jsp_details = details.get_custom_instance_details(name)

    return jsp, custom_jsp_details, name


if __name__ == '__main__':
    #get_random_custom_instance_and_details_and_name(n_jobs=4, n_machines=4)

    download_instances(start_id=1, end_id=1)

    abz5_ta_path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath("abz5.txt")

    jsp_instance_from_ta, _ = parser.parse_jps_taillard_specification(abz5_ta_path)

    instance, lb = get_benchmark_instance_and_lower_bound(name="abz5")
