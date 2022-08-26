import collections
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details

from ortools.sat.python import cp_model

from jss_utils.jss_logger import log
from jss_graph_env.disjunctive_graph_jss_visualizer import DisjunctiveGraphJspVisualizer


def solve_jsp(jsp_instance: np.ndarray, plot_results: bool = True) -> (float, str, pd.DataFrame, dict):
    # Create the model.
    model = cp_model.CpModel()

    machine_order = jsp_instance[0]
    processing_times = jsp_instance[1]

    machines_count = machine_order.max() + 1  # first machine is indexed 0
    all_machines = range(machines_count)

    horizon = np.sum(np.ravel(processing_times))

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, (job_machine_order, job_processing_times) in enumerate(zip(machine_order, processing_times)):
        for task_id, (machine, duration) in enumerate(zip(job_machine_order, job_processing_times)):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job_machine_order in enumerate(machine_order):
        for task_id in range(len(job_machine_order) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job_machine_order) - 1].end
        for job_id, job_machine_order in enumerate(machine_order)
    ])

    model.Minimize(obj_var)

    # Solve model.
    start = time.perf_counter()
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    end = time.perf_counter()

    solving_duration = end - start

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"

        log.info(f"{status} solution found.")

        # Create one list of assigned tasks per machine.
        assigned_jobs = []
        for job_id, (job_machine_order, job_processing_times) in enumerate(zip(machine_order, processing_times)):
            for task_id, (machine, duration) in enumerate(zip(job_machine_order, job_processing_times)):
                assigned_jobs.append({
                    'Task': f'Job {job_id}',
                    'Start': solver.Value(all_tasks[job_id, task_id].start),
                    'Finish': solver.Value(all_tasks[job_id, task_id].start) + duration,
                    'Resource': f'Machine {machine}'
                })

        df = pd.DataFrame(assigned_jobs)

        if plot_results:
            # generate colors for visualizer
            c_map = plt.cm.get_cmap("rainbow")  # select the desired cmap
            arr = np.linspace(0, 1, machines_count)  # create a list with numbers from 0 to 1 with n items
            machine_colors = {m_id: c_map(val) for m_id, val in enumerate(arr)}
            colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in machine_colors.items()}

            visualizer = DisjunctiveGraphJspVisualizer(dpi=80)
            visualizer.gantt_chart_console(df, colors)

        makespan = solver.ObjectiveValue()

        log.info(f"or tools solving duration: {solving_duration:2f} sec")
        log.info(f'makespan: {makespan} ({status} solution)')
        # visualizer.render_gantt_in_window(df, colors=colors, wait=None)

        info = {
            "makespan": makespan,
            "solving_duration": solving_duration,
            "gantt_df": df,
            "or_tools_status": status,
            "or_tools_solving_duration": solving_duration,
            "or_tools_makespan": makespan
        }

        return makespan, status, df, info
    else:
        log.info("could not solve jsp instance. Check if your instance is well defined.")
        raise RuntimeError("could not solve jsp instance. Check if your instance is well defined.")


if __name__ == '__main__':
    instance_name = "ta01"

    jsp = parser.get_instance_by_name(instance_name)
    makespan, status, *_ = solve_jsp(jsp_instance=jsp)

    if status == "OPTIMAL":
        instance_details = details.get_jps_instance_details(instance_name)
        if instance_details["lb_optimal"]:
            lb = instance_details["lower_bound"]
            if lb != makespan:
                log.error(f"the calculated optimal makespan ({makespan}) differs from the literature ({lb}). "
                          f"Apparently something went wrong. Check the implementation")
            else:
                log.info("the makespan value for the optimal solution is consistent with the literature. :)")
