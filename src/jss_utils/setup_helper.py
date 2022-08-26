from pathlib import Path

from jss_utils.jsp_instance_details import download_benchmark_instances_details, update_custom_instance_details
from jss_utils.jsp_instance_downloader import download_instances
from jss_utils.jss_logger import log

from jss_utils import PATHS


def rm_tree(pth):
    """
    deletes a folder and every subfolder and files inside that folder.

    :param pth: a path (to a folder)
    :return: None
    """
    pth = Path(pth)
    if not pth.exists():
        return

    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)

    if not pth.is_dir():
        pth.rmdir()


def clear_resources_dir(sure: bool = False) -> None:
    if not sure:
        log.warn(f"running 'clear_resources_dir' will clear all download files and RL experiments. "
                 f"These files cannot be recovered afterwards. "
                 f"If you are aware of that call this function with the parameter 'sure' set to 'Ture'. "
                 f"I hope you know what you're doning.")
        return
    # Code that will run before the tests
    resources_pth = PATHS.RESOURCES_ROOT_PATH
    for child in resources_pth.glob('*'):
        if child.name == "readme_images":
            continue
        rm_tree(child)

    PATHS.setup_paths()


def prepare_for_testing() -> None:
    resources_pth = PATHS.RESOURCES_ROOT_PATH
    for child in resources_pth.glob('*'):
        if child.name == "readme_images":
            continue
        if child.name == "jsp_instances":
            rm_tree(PATHS.JSP_INSTANCES_TAILLARD_PATH)
            rm_tree(PATHS.JSP_INSTANCES_STANDARD_PATH)
            # only 3x3 instances are used for testing
            rm_tree(PATHS.JSP_INSTANCES_CUSTOM_PATH.joinpath("3x3"))
            continue
        log.info(f"deleting '{child}'")
        rm_tree(child)

    PATHS.setup_paths()


def post_process_testing() -> None:
    # only 3x3 instances are used for testing
    rm_tree(PATHS.JSP_INSTANCES_CUSTOM_PATH.joinpath("3x3"))
    download_instances()
    download_benchmark_instances_details()
    update_custom_instance_details()


def default_project_setup(sure: bool = False) -> None:
    clear_resources_dir(sure=True)
    download_instances()
    download_benchmark_instances_details()
    update_custom_instance_details()


if __name__ == '__main__':
    default_project_setup(sure=True)
