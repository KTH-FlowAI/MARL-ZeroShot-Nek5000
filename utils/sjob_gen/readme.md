# SLURM Job Script Generator

This directory contains tools to generate customized SLURM job scripts based on the template in `sjob-template.sh`.

## Files

- `generate_slurm_job.py` - Main Python script for generating SLURM job scripts
- `create_job.sh` - Simple shell wrapper for easier usage
- `sjob-template.sh` - Original SLURM job template
- `README_job_generator.md` - This documentation

## Quick Start

### Using the Shell Wrapper (Recommended)

The simplest way to create a job script:

```bash
# Basic usage
./create_job.sh -o my_job.sh

# Custom configuration
./create_job.sh -o custom_job.sh \
  -a myaccount \
  -t 24:00:00 \
  -j "My-DRL-Experiment" \
  -c "MC16-TD3.yml" \
  -e "myemail@domain.com"

# Schedule job to start in 2 hours
./create_job.sh -o scheduled_job.sh -s 2
```

### Using the Python Script Directly

For more advanced customization:

```bash
# Basic usage
python generate_slurm_job.py -o my_job.sh

# Full customization
python generate_slurm_job.py -o custom_job.sh \
  --account myaccount \
  --time-limit 24:00:00 \
  --job-name "My-DRL-Experiment" \
  --config-name "MC16-TD3.yml" \
  --nodes 2 \
  --ntasks-per-node 24 \
  --nek5000-mpi-ranks 64 \
  --mail-user "myemail@domain.com"
```

## Available Parameters

### Account and Resources
- `--account` - SLURM account (default: deepwing)
- `--time-limit` - Time limit in HH:MM:SS format (default: 12:00:00)
- `--partition` - SLURM partition (default: batch)
- `--nodes` - Number of nodes (default: 1)
- `--ntasks-per-node` - Number of tasks per node (default: 48)
- `--cpus-per-task` - Number of CPUs per task (default: 1)
- `--no-exclusive` - Remove exclusive flag

### Job Identification
- `--job-name` - Job name (default: DDPG-MINI)
- `--mail-type` - Mail notification type (default: ALL)
- `--mail-user` - Email for notifications (default: yuninw@umich.edu)
- `--output-file` - Output file path (default: ./log-files/log-ddpg-mini)
- `--error-file` - Error file path (default: ./log-files/log-ddpg-mini)

### Scheduling
- `--begin-hours` - Start job N hours from now
- `--begin-time` - Specific start time (YYYY-MM-DDTHH:MM:SS)

### Configuration
- `--config-name` - Configuration file name (default: MC16-DDPG.yml)
- `--run-mode` - Run mode (default: run)

### Execution
- `--python-mpi-ranks` - MPI ranks for Python process (default: 1)
- `--nek5000-mpi-ranks` - MPI ranks for Nek5000 (default: 32)
- `--mpi-pml` - MPI PML (default: ucx)
- `--mpi-io` - MPI IO (default: ompio)

### Environment
- `--bashrc-openmpi` - OpenMPI bashrc file (default: ~/.bashrc.openmpi_ucx)
- `--bashrc-miniforge` - Miniforge bashrc file (default: ~/.bashrc.miniforge)

## Examples

### Example 1: Basic Job
```bash
./create_job.sh -o basic_job.sh
```

### Example 2: Long-running Job
```bash
./create_job.sh -o long_job.sh \
  -t 48:00:00 \
  -j "Long-Training" \
  -n 4 \
  -r 128
```

### Example 3: Scheduled Job
```bash
./create_job.sh -o scheduled_job.sh \
  -s 6 \
  -j "Scheduled-Experiment" \
  -e "researcher@university.edu"
```

### Example 4: Different Configuration
```bash
./create_job.sh -o td3_job.sh \
  -c "MC16-TD3.yml" \
  -j "TD3-Experiment" \
  -t 18:00:00
```

## Submitting Jobs

After generating a job script:

```bash
# Submit the job
sbatch my_job.sh

# Check job status
squeue -u $USER

# Cancel a job
scancel JOB_ID
```

## Validation

The generator includes validation for:
- Time format (HH:MM:SS, allows up to 24:00:00)
- Email format (basic validation)
- Positive integers for numeric parameters
- Required parameters

## Error Handling

The script will:
- Show clear error messages for invalid inputs
- Validate all parameters before generating the script
- Provide helpful usage information with `--help`

## Customization

To modify the template or add new parameters:

1. Edit the `template_params` dictionary in `generate_slurm_job.py`
2. Add corresponding command-line arguments
3. Update the `generate_job_script()` method to include new parameters
4. Test with various configurations

## Troubleshooting

### Common Issues

1. **Permission denied**: Make sure the scripts are executable
   ```bash
   chmod +x generate_slurm_job.py create_job.sh
   ```

2. **Python not found**: Ensure Python 3 is available in your PATH

3. **Invalid time format**: Use HH:MM:SS format (e.g., 12:00:00, 24:00:00)

4. **Invalid email**: Provide a valid email address format

### Getting Help

- Use `--help` or `-h` for usage information
- Check the examples in this README
- Verify your SLURM configuration matches the generated parameters

