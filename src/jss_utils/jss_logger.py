import logging
import shutil

from rich.logging import RichHandler

from jss_utils.banner import big_banner, small_banner

# print banner when logger is imported
w, h = shutil.get_terminal_size((80, 20))

#disable banner:
#print(small_banner if w < 140 else big_banner)

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False)]
)

# create file handler which logs messages
# fh = logging.FileHandler(PATHS.LOGS_FILE_PATH)
# fh.setLevel(logging.INFO)

log = logging.getLogger("rich")
# log.addHandler(fh)

if __name__ == '__main__':
    log.info("fancy")

