#!/usr/local_rwth/bin/zsh
 
### Job name

#SBATCH --mail-user=felix.stillger@rwth-aachen.de
#SBATCH --mail-type=ALL

#SBATCH -J cpu_serial
#SBATCH -o logs/cpu_serial.%J.log
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=15G
#SBATCH --time=0-04:30:00
# Load the same python as used for installation


module load python/3.9.6

# start skript
python3 --version
# python3 /home/fs608798/seminararbeit/progra/Mask_RCNN/just_test.py
python3 eval.py

