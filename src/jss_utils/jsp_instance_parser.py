import random
import json

import numpy as np
import pathlib as pl

import jss_utils.PATHS as PATHS

from typing import Dict

from jss_utils.jss_logger import log


def parse_jps_standard_specification(instance_path: pl.Path) -> (np.ndarray, np.ndarray):
    """
    Note:   the description of the specification indexes machines at 1. The instance .txt file indexes machines
            with index 0.
            So 'machine 1' in the description is denoted as '0' in the instance .txt file.

    Description of the standard specification (src: http://jobshop.jjvh.nl/explanation.php)

            On the first line are two numbers, the first is the number of jobs and the second the number of machines.
            Following there is a line for each job. The order for visiting the machines is presented together with the
            corresponding processing time. The numbering of the machines starts at 0.

            For example an instance with only a single job on three machines where the processing time is 5 on
            machine 1, 6 on machine 2 and 7 on machine 3 and the order that the machines are to be visited by that
            job is 2,3,1. The instance would be presented as:

            1	3
            1	6	2	7	0	5

    Explanation of the provided example

            Notation:
            mi      ->  machine i
            t(mi)   ->  procession time of the corresponding operation on machine i

            Annotated example:

            1(jobs: j1)	3(machines: m1,m2,m3)
    [j1:]   1(m2)	    6(t(m2))	            2(m3)	    7(t(m3))	    0(m1)	    5(t(m1))

            Order of j1:
            m2, m3, m4

    Notation:
            m_i      ->  machine i
            t(m_i)   ->  procession time of the corresponding operation on machine i


    output format

        [
            [(m_0,t(m_0)), (m0,t(m0)), ...], # job0
            [(m_0,t(m_0)), (m0,t(m0)), ...], # job1
            ...
        ]

    :param instance_path:   path to the jsp instance
    :return:                jsp as numpy array, standard style jsp numpy array
    """

    log.info(f"parsing file '{instance_path}'")

    with open(instance_path) as f:
        lines = f.readlines()

    first_line = lines.pop(0)
    split_data = first_line.split()

    if len(split_data) != 2:
        raise ValueError(f"file does not follow specification. "
                         f"Expected 2 values in the first line but got {len(split_data)}")
    # number of jobs and machines (n = number)
    n_jobs, n_machines = (int(n) for n in split_data)

    log.info(f"instance appears to contain {n_jobs} jobs, {n_machines} machines")

    # lines str array to numpy array
    std_matrix = np.array([[int(entry) for entry in strg_line.split()] for strg_line in lines])

    # infer instance size from data and check validity
    n, m = std_matrix.shape

    assert n == n_jobs
    assert m == 2 * n_machines  # in standard specification there are two cols per machine

    job_machine_order = std_matrix[:, ::2]
    job_processing_times = std_matrix[:, 1::2]

    res = np.array([
        job_machine_order,
        job_processing_times
    ])

    return res, std_matrix


def parse_jps_taillard_specification(instance_path: pl.Path) -> (np.ndarray, np.ndarray):
    """

    Description of the Taillard specification (src: http://jobshop.jjvh.nl/explanation.php)

            On the first line are two numbers, the first is the number of jobs and the second the number of machines.
            Following there are two matrices the first with a line for each job containing the processor times for each
            operation the second with the order for visiting the machines. The numbering of the machines starts at 1.

            For example the same instance as above would be presented as:

            1	3
            6	7	5
            2	3	1

    Explanation of the provided example

            Notation:
            mi      ->  machine i
            t(mi)   ->  procession time of the corresponding operation on machine i


            [JPS size]
            1(jobs: j1)   3(machines: m1,m2,m3)

            [processing times]
            6(t(m2))    7(t(m3))    5(t(m1))

            [order machines]
            2(m2)	    3(m3)	    1(m1)

    :param instance_path:   path to the jsp instance
    :return:                jsp as numpy array, Taillard style jsp numpy array
    """

    log.info(f"parsing file '{instance_path}'")

    with open(instance_path) as f:
        lines = f.readlines()

    first_line = lines.pop(0)
    split_data = first_line.split()

    if len(split_data) != 2:
        raise ValueError(f"file does not follow specification. "
                         f"Expected 2 values in the first line but got {len(split_data)}")
    # number of jobs and machines (n = number)
    n_jobs, n_machines = (int(n) for n in split_data)

    log.info(f"instance appears to contain {n_jobs} jobs, {n_machines} machines")

    # lines str array to numpy array
    taillard_matrix = np.array([[int(entry) for entry in strg_line.split()] for strg_line in lines])

    # infer instance size from data and check validity
    n, m = taillard_matrix.shape

    assert n == 2 * n_jobs  # in Taillard specification there are two rows per job
    assert m == n_machines

    job_processing_times = taillard_matrix[:n_jobs]
    job_machine_order = taillard_matrix[n_jobs:] - 1  # shift machine starting index from 1 to 0

    res = np.array([
        job_machine_order,
        job_processing_times
    ])

    return res, taillard_matrix


def get_instance_by_name(name: str, /) -> np.ndarray:
    """
    returns a jsp benchmark instance as numpy array.
    This function used Taillard instances for parsing.

    Make sure that you have downloaded the instance first (run `jsp_instance_downloader.download_instances` beforehand).

    :param name: the name of the instance
    :return: jsp instance numpy array
    """
    path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath(f"{name}.txt")
    instance, _ = parse_jps_taillard_specification(path)
    return instance


if __name__ == '__main__':
    abz7_std_path = PATHS.JSP_INSTANCES_STANDARD_PATH.joinpath("ft06.txt")
    abz7_ta_path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath("ft06.txt")

    jps_instance_from_std, std_matrix = parse_jps_standard_specification(abz7_std_path)
    jps_instance_from_ta, taillard_matrix = parse_jps_taillard_specification(abz7_ta_path)

    assert np.array_equal(jps_instance_from_std, jps_instance_from_ta)

    print(repr(jps_instance_from_std))
