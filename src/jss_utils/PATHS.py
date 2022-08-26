import pathlib as pl

import os
import rich

PATHS_FILE_PATH = pl.Path(os.path.abspath(__file__))
JSS_UTILS_PATH = PATHS_FILE_PATH.parent
SRC_PATH = JSS_UTILS_PATH.parent

PROJECT_ROOT_PATH = SRC_PATH.parent

WANDB_API_KEY_FILE_PATH = PROJECT_ROOT_PATH.joinpath("wandb_api_key_file")

RESOURCES_ROOT_PATH = PROJECT_ROOT_PATH.joinpath("resources")

LOGS_FILE_PATH = RESOURCES_ROOT_PATH.joinpath("jss_logger.txt")

JSP_INSTANCES_PATH = RESOURCES_ROOT_PATH.joinpath("jsp_instances")
RL_OUTPUT_PATH = RESOURCES_ROOT_PATH.joinpath("rl_output")

TENSORBOARD_LOGS_PATH = RL_OUTPUT_PATH.joinpath("tensorboard_logs")
WANDB_PATH = RL_OUTPUT_PATH.joinpath("wandb")
SB3_EXAMPLES = RL_OUTPUT_PATH.joinpath("sb3_examples")

SB3_EXAMPLES_GIF = SB3_EXAMPLES.joinpath("gif")
SB3_EXAMPLES_VIDEO = SB3_EXAMPLES.joinpath("video")

JSP_INSTANCES_DETAILS_PATH = RESOURCES_ROOT_PATH.joinpath("jps_instance_details")

JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH = JSP_INSTANCES_DETAILS_PATH.joinpath("benchmark_details.json")
JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH = JSP_INSTANCES_DETAILS_PATH.joinpath("custom_instance_details.json")

JSP_INSTANCES_STANDARD_PATH = JSP_INSTANCES_PATH.joinpath("standard")
JSP_INSTANCES_TAILLARD_PATH = JSP_INSTANCES_PATH.joinpath("taillard")
JSP_INSTANCES_CUSTOM_PATH = JSP_INSTANCES_PATH.joinpath("custom")

_deep_paths = [
    JSP_INSTANCES_DETAILS_PATH,
    JSP_INSTANCES_STANDARD_PATH,
    JSP_INSTANCES_TAILLARD_PATH,
    JSP_INSTANCES_CUSTOM_PATH,
    WANDB_PATH,
    SB3_EXAMPLES,
    SB3_EXAMPLES_GIF
]


def setup_paths() -> None:
    for path in _deep_paths:
        path.mkdir(parents=True, exist_ok=True)


setup_paths()


def main() -> None:
    rich.print(f"""
        these should be the absolute paths [bold]on your machine[/bold]:

        Project_Root:   {PROJECT_ROOT_PATH}
        ┣━━ resources: {RESOURCES_ROOT_PATH}
        ┃   ┣━━ jps_instance_details: {JSP_INSTANCES_DETAILS_PATH}
        ┃   ┃   ┗━━ benchmark_details.json: {JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH}
        ┃   ┣━━ jsp_instances: {JSP_INSTANCES_PATH}
        ┃   ┃   ┣━━ custom: {JSP_INSTANCES_CUSTOM_PATH}
        ┃   ┃   ┣━━ standard: {JSP_INSTANCES_STANDARD_PATH}
        ┃   ┃   ┗━━ taillard: {JSP_INSTANCES_TAILLARD_PATH}
        ┃   ┗━━ rl_output: {RL_OUTPUT_PATH}
        ┃       ┗━━ wandb: {WANDB_PATH}
        ┗━━ src: {SRC_PATH}
            ┗━━ JssUtils: {JSS_UTILS_PATH}
                ┣━━ sb3_examples: {SB3_EXAMPLES}
                ┃   ┗━━ gif: {JSP_INSTANCES_TAILLARD_PATH}
                ┗━━ pathy.py: {PATHS_FILE_PATH}

        """)


if __name__ == '__main__':
    main()
