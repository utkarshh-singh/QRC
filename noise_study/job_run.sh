#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=124000M
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --job-name=soap_gas
#SBATCH --output=result.slurm.log
#SBATCH --error=result.slurm.err

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



run_soapgas_dharm /home/admin.joey.wang/work/admin.joey.wang/software/QRC/noise_study 32 run_noise_sweep
