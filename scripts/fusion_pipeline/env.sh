# Environment setup for the fusion pipeline.
# Sourced by all SLURM submission scripts — edit here to change modules or venv.

module load stack/2024-06 cuda/12.8.0   # ADAPT HERE
module load stack/2024-06 python/3.11.6  # ADAPT HERE
source skeleplexenv/bin/activate          # ADAPT HERE — path to your virtual environment
