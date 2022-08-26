import requests
import bs4
import json

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_or_tools_solver as or_tools_solver

from typing import Dict

from jss_utils.jss_logger import log


def download_benchmark_instances_details() -> None:
    """
    scrapes additional information of benchmark instances from

        http://jobshop.jjvh.nl

    and save them in as in .json-format in the 'resources' directory.

    :return: None
    """
    url = 'http://jobshop.jjvh.nl/'
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, features="html.parser")
    jsp_instance_details = {}
    for row in soup.findAll("tr", attrs={"class": "hideRow"}):
        instance_name = row.find('a', attrs={"class": "instance"}).getText()
        log.info(f"parsing instance '{instance_name}'")
        lower_bound = int(row.find('div', attrs={"class": "lb"}).getText())
        upper_bound = int(row.find('div', attrs={"class": "ub"}).getText())

        _, jobs, machines, *_ = row.find_all('td')

        jobs = int(jobs.getText())
        machines = int(machines.getText())

        # the number of solutions is the last div in a row
        no_solutions = None
        for no_solutions in row.find_all('div'): pass
        if no_solutions:
            no_solutions = int(no_solutions.getText())
        else:
            log.warning(f"could not parse the number of solutions for instance {instance_name}.")

        data = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "jobs": jobs,
            "machines": machines,
            "n_solutions": no_solutions,
            "lb_optimal": no_solutions > 0
        }
        jsp_instance_details[instance_name] = data
        log.info(f"adding {data} with key '{instance_name}' to details")

    PATHS.JSP_INSTANCES_DETAILS_PATH.mkdir(parents=True, exist_ok=True)
    log.info(f"saving details to .json file ('{PATHS.JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH}')")

    with open(PATHS.JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH, 'w') as fp:
        json.dump(jsp_instance_details, fp, indent=4)

    log.info(f"successfully saved details to .json file ('{PATHS.JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH}')")


def parse_instance_details() -> Dict:
    """
    reads 'jps_instance_details/benchmark_details.json' file and returns it as a dictionary.

    make sure to download the instance details beforehand (run 'download_benchmark_instances_details' once)

    :return: the 'jps_instance_details/benchmark_details.json'-file as a python dictionary
    """
    with open(PATHS.JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH) as f:
        details_dict = json.load(f)
    return details_dict


def parse_custom_instance_details() -> Dict:
    """
    reads 'jps_instance_details/custom_instance_details.json' file and returns it as a dictionary.

    make sure to download the instance details beforehand (run 'download_benchmark_instances_details' once)

    :return: the 'jps_instance_details/custom_instance_details.json'-file as a python dictionary
    """
    with open(PATHS.JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH) as f:
        details_dict = json.load(f)
    return details_dict


def get_jps_instance_details(instance: str) -> Dict:
    """
    looks up the details-entry that corresponds to the specified instance in the
    'jps_instance_details/benchmark_details.json'-file and returns them as a python dictionary.

    :param instance: the name of instance (example: 'ft06'). see: http://jobshop.jjvh.nl/index.php
    :return:
    """
    return parse_instance_details()[instance]


def update_custom_instance_details() -> None:
    # check if custom instance details file exists
    if not PATHS.JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH.is_file():
        log.info("there is no custom instance details file jet. creating one..")
        with open(PATHS.JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH, 'w') as fp:
            json.dump({}, fp, indent=4)

    with open(PATHS.JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH) as f:
        details_dict = json.load(f)

    # all instance files
    custom_instances_generator = PATHS.JSP_INSTANCES_CUSTOM_PATH.glob('**/*.txt')

    details_keys = details_dict.keys()
    for custom_instance in custom_instances_generator:
        name = custom_instance.stem
        if name in details_keys:
            continue

        log.info("")
        log.info(f"handling custom instance '{name}'")

        jsp, _ = parser.parse_jps_taillard_specification(instance_path=custom_instance)
        _, n_jobs, n_machines = jsp.shape

        makespan, status, df, info = or_tools_solver.solve_jsp(jsp_instance=jsp, plot_results=False)

        details_dict[name] = {
            "lower_bound": makespan,
            "upper_bound": makespan,
            "jobs": n_jobs,
            "machines": n_machines,
            "n_solutions": 1,
            "lb_optimal": status == "OPTIMAL",
            "path": str(custom_instance),
            "gantt_df": df.to_dict(),
        }

    log.info(f"saving details to .json file ('{PATHS.JPS_BENCHMARK_INSTANCES_DETAILS_FILE_PATH}')")
    with open(PATHS.JPS_CUSTOM_INSTANCES_DETAILS_FILE_PATH, 'w') as fp:
        json.dump(details_dict, fp, indent=4)


def get_custom_instance_details(name: str) -> dict:
    all_details = parse_custom_instance_details()

    if name not in all_details.keys():
        log.info(f"there are no details for the custom instance '{name}' in the details file. "
                 f"updating details file and retrying...")
        update_custom_instance_details()
        if name not in all_details.keys():
            raise ValueError(f"the requested custom instance '{name}' seems not to exists in the resource folder.")
    return all_details[name]


if __name__ == '__main__':
    download_benchmark_instances_details()
    # details = parse_instance_details()

    # log.info(f"EXAMPLE: details for ft06: {details['ft06']}")

    # update_custom_instance_details()
    # get_jps_instance_details("abz6")
