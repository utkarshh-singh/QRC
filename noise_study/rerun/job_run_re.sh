#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=124000M
#SBATCH --ntasks-per-node=32
#SBATCH --time=50:00:00
#SBATCH --job-name=run_noise_sweep
#SBATCH --output=noise_result.slurm.log
#SBATCH --error=noise_result.slurm.err

module load intel-oneapi-compilers
module load intel-oneapi-mkl


#unset PYTHONPATH

export PYTHONPATH=$PYTHONPATH:$HOME/work/$USER/software/QRC/noise_study

PATH="$HOME/work/$USER/software/QRC/noise_study:$PATH"
export PATH

source ~/anaconda3/etc/profile.d/conda.sh
echo $PYTHONPATH
echo "\n"
echo $PATH
conda init
conda activate utkarshh
ulimit -s unlimited

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

DIR=/home/admin.joey.wang/work/admin.joey.wang/software/QRC/noise_study/rerun
NPROC=32
FILE=run_noise_sweep

USER=`whoami`

TMP1=/scratch/$USER
if [ ! -d $TMP1 ]; then
    mkdir $TMP1
fi

SCRATCH=$TMP1/qrc_job.$$

echo "Scratch directory: " $SCRATCH
mkdir $SCRATCH

##########################################
# Copy files to the scratch directory
##########################################
cp $DIR/"$FILE".py $SCRATCH
cp -r $DIR/../configs $SCRATCH
##########################################
# Move to the scratch directory and run
# the job
##########################################
cd $SCRATCH

python run_noise_sweep.py --workers $NPROC

##########################################
# Copy files back and clean up
##########################################
cp -r $SCRATCH/* $DIR
cd $DIR
rm -rf $SCRATCH
