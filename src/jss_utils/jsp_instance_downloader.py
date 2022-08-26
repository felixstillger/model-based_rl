import requests
import re

import pathlib as pl

import jss_utils.PATHS as PATHS

from jss_utils.jss_logger import log


def download_instances(start_id: int = 1, end_id: int = 243, format: str = None,
                       skip_on_warnings: bool = True) -> None:
    """
    Downloads JPS instances from

        http://jobshop.jjvh.nl

    and save them in the 'resources' directory

    The download links follow a pattern that is used here to download the instances automatically.
    The website lists all instances by an incremental id.
    See legend down below.

    If the default parameters are used the function will download all instances in both formats.

    :param start_id:            starting id of the instance
    :param end_id:              starting id of the instance
    :param format:              the format of the file. valid options: 'standard', 'taillard'.
                                see more: http://jobshop.jjvh.nl/explanation.php
    :param skip_on_warnings:    if a problem occurs while downloading an instance and 'skip_on_warnings' is set to
                                'True', the script will continue with the next instance. If is set to 'False' the script
                                crashes in that case.
    :return:                    None


    legend for indexes (date: 24.05.22):

        abz5 - abz9 :   start_id = 1, end_id = 5

        ft06 - ft20:    start_id = 6, end_id = 8

        la01 - la40:    start_id = 10, end_id = 49

        orb01 - orb10:  start_id = 50, end_id = 59

        swv01 - swv20:  start_id = 60, end_id = 79

        yn01 - yn04:    start_id = 80, end_id = 83

        ta01 - ta80:    start_id = 84, end_id = 163

        dmu01 - dmu80:  start_id = 164, end_id = 243
    """
    # handle invalid arguments
    if start_id < 1:
        raise ValueError(f"only ids form 1 to 243 are valid")
    if start_id > 243:
        raise ValueError(f"only ids form 1 to 243 are valid")

    if format is not None:
        if format == "standard":
            formats = ["standard"]
        elif format == "taillard":
            formats = ["taillard"]
        else:
            raise ValueError(f"format={format} is an invalid argument. "
                             f"only 'taillard' and 'standard' are valid arguments.")
    else:
        formats = ["standard", "taillard"]

    for jps_spec in formats:
        log.info(f"processing instances in {jps_spec} specification")

        spec_dir = PATHS.JSP_INSTANCES_TAILLARD_PATH if jps_spec == "taillard" \
            else PATHS.JSP_INSTANCES_STANDARD_PATH

        # create dir if not exists
        pl.Path(spec_dir).mkdir(parents=True, exist_ok=True)

        # iterate through the specified instances
        for jps_instance_index in range(start_id, end_id + 1):

            # somehow there is no instance for the id 9 on http://jobshop.jjvh.nl, so just ignore that index
            if jps_instance_index == 9:
                continue

            # build url
            jps_instance_url = f"http://jobshop.jjvh.nl/specification_file.php" \
                               f"?instance_id={jps_instance_index}&specification={jps_spec}"

            log.info(f"fetching from downloaded '{jps_instance_url}'")
            resp = requests.get(jps_instance_url)

            # infer filename from download and do saftyx chechs
            cd = resp.headers["Content-Disposition"]
            file_name_matches = re.findall('filename="(.+)"', cd)

            if len(file_name_matches) <= 0:
                warning_msg = f"""
                    something went wrong while downloading from:
                    {jps_instance_url}

                    Content-Disposition: {cd}
                    the filename does not match the pattern, that this script requires.  
                """
                if skip_on_warnings:
                    log.warning(warning_msg)
                    log.warning(f"skipping this file.")
                    continue
                else:
                    raise ValueError(warning_msg)

            file_name = file_name_matches[0]
            file_extension = pl.Path(file_name).suffix

            if file_extension != ".txt":
                warning_msg = f"""
                            something went wrong while downloading from:
                            {jps_instance_url}

                            file: {file_name}
                            the file_extension is not .txt
                        """
                if skip_on_warnings:
                    log.warning(warning_msg)
                    log.warning(f"skipping this file.")
                    continue
                else:
                    raise ValueError(warning_msg)

            log.info(f"successfully downloaded file '{file_name}' (ecoding: {resp.encoding})")

            # write to file
            file_path = spec_dir.joinpath(file_name)
            log.info(f"writing instance to '{file_path}'")
            with open(file_path, "w+") as f:
                f.writelines(resp.text)  # resp.text contains the spec as a multiline string
            log.info(f"successfully saved instance as '{file_name}' ({jps_spec})")


if __name__ == '__main__':
    download_instances()
