#!/usr/bin/env python3
"""
SLURM Job Script Generator

This script generates customized SLURM job scripts based on the template
found in sjob-template.sh. It allows users to customize various parameters
like account, time limits, configuration files, and execution settings.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path


class SlurmJobGenerator:
    def __init__(self):
        self.template_params = {
            # Account and Resource Settings
            'account': 'deepwing',
            'time_limit': '12:00:00',
            'partition': 'batch',
            'exclusive': True,
            'nodes': 1,
            'ntasks_per_node': 48,
            'cpus_per_task': 1,
            
            # Job Identification
            'job_name': 'DDPG-MINI',
            'mail_type': 'ALL',
            'mail_user': 'yuninw@umich.edu',
            'output_file': './log-files/log-ddpg-mini',
            'error_file': './log-files/log-ddpg-mini',
            'begin_time': None,  # Will be set if specified
            
            # Environment
            'bashrc_openmpi': '~/.bashrc.openmpi_ucx',
            'bashrc_miniforge': '~/.bashrc.miniforge',
            'unset_vars': ['MPI_UCX_ROOT', 'UCX_ROOT'],
            'export_vars': {'UCX_WARN_UNUSED_ENV_VARS': 'n'},
            
            # Configuration
            'config_name': 'MC16-DDPG.yml',
            'run_mode': 'run',
            
            # Execution
            'python_mpi_ranks': 1,
            'nek5000_mpi_ranks': 32,
            'mpi_pml': 'ucx',
            'mpi_io': 'ompio'
        }
    
    def validate_time_format(self, time_str):
        """Validate time format (HH:MM:SS)"""
        try:
            parts = time_str.split(':')
            if len(parts) != 3:
                return False
            hours, minutes, seconds = map(int, parts)
            # Allow up to 24 hours (24:00:00 is valid for SLURM)
            return 0 <= hours <= 24 and 0 <= minutes <= 59 and 0 <= seconds <= 59
        except (ValueError, TypeError):
            return False
    
    def validate_email(self, email):
        """Basic email validation"""
        return '@' in email and '.' in email.split('@')[-1]
    
    def set_begin_time(self, hours_from_now=None, specific_time=None):
        """Set begin time for the job"""
        if specific_time:
            self.template_params['begin_time'] = specific_time
        elif hours_from_now:
            begin_time = datetime.now() + timedelta(hours=hours_from_now)
            self.template_params['begin_time'] = begin_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    def update_params(self, **kwargs):
        """Update template parameters with provided values"""
        for key, value in kwargs.items():
            if key in self.template_params:
                # Validate specific parameters
                if key == 'time_limit' and not self.validate_time_format(value):
                    raise ValueError(f"Invalid time format: {value}. Use HH:MM:SS format.")
                if key == 'mail_user' and not self.validate_email(value):
                    raise ValueError(f"Invalid email format: {value}")
                if key in ['nodes', 'ntasks_per_node', 'cpus_per_task', 'python_mpi_ranks', 'nek5000_mpi_ranks']:
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(f"{key} must be a positive integer")
                
                self.template_params[key] = value
            else:
                print(f"Warning: Unknown parameter '{key}' will be ignored")
    
    def generate_job_script(self, output_file=None):
        """Generate the SLURM job script"""
        params = self.template_params
        
        script_content = f"""#!/bin/bash -l
#### A Template for submitting your JOB

#-------- Account ------
#SBATCH -A {params['account']}

#-------- Partions ------------
#SBATCH -t {params['time_limit']}
#SBATCH -p {params['partition']}"""
        
        if params['exclusive']:
            script_content += "\n#SBATCH --exclusive"
        
        script_content += f"""
#SBATCH -N {params['nodes']}
#SBATCH --ntasks-per-node={params['ntasks_per_node']}
#SBATCH --cpus-per-task={params['cpus_per_task']}

#-------- Outputs and notification ------
#SBATCH -J {params['job_name']}
#SBATCH --mail-type={params['mail_type']}
#SBATCH --mail-user={params['mail_user']}
#SBATCH --output={params['output_file']}
#SBATCH --error={params['error_file']}"""
        
        if params['begin_time']:
            script_content += f"\n#SBATCH --begin={params['begin_time']}"
        
        script_content += f"""

# Environment 
#------------------------
source {params['bashrc_openmpi']}
source {params['bashrc_miniforge']}"""
        
        for var in params['unset_vars']:
            script_content += f"\nunset {var}"
        
        for var, value in params['export_vars'].items():
            script_content += f"\nexport {var}={value}"
        
        script_content += f"""
echo "Environment ALL SET"
#------------------------


# Summary
#------------------------
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
#------------------------

CONFIG_NAME="{params['config_name']}"

RUN_MODE="{params['run_mode']}"
echo "DRL CONFIG: ${{CONFIG_NAME}}, RUN MODE: ${{RUN_MODE}}"
mpirun  --mca io {params['mpi_io']} \\
        -n {params['python_mpi_ranks']} python -m nek_MARL initial ../conf/$CONFIG_NAME
## Identify the path-to-go
RUN_PATH=$(head -n 1 RUN_PATH.txt)
AGENT=$(tail -n 1 RUN_PATH.txt)
## Sort the log history
if [ "$RUN_MODE" = "run" ]; then
    fileNUM=$(ls "${{RUN_PATH}}"/round* -dq | wc -l)
    fileNUM=$((fileNUM + 1))
    printf -v fileNUM "%03d" "$fileNUM"  # Zero-pad to 3 digits
    HISTORY_PATH="${{RUN_PATH}}/round${{fileNUM}}"
    mv "${{RUN_PATH}}/history" "${{HISTORY_PATH}}"
    echo "MV HISTORY: ${{HISTORY_PATH}}"
fi

## RUN! To use mpi_split, one should launch two programs at once. Therefore 
mpirun --mca pml {params['mpi_pml']} \\
       --mca io {params['mpi_io']} \\
       -n {params['python_mpi_ranks']} python -m nek_MARL ${{RUN_MODE}} ../conf/$CONFIG_NAME runner.policy=${{AGENT}} :\\
       -n {params['nek5000_mpi_ranks']} bash -c "cd ${{RUN_PATH}} && ./nek5000"
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(script_content)
            print(f"Job script generated: {output_file}")
        else:
            print(script_content)
        
        return script_content


def main():
    parser = argparse.ArgumentParser(
        description='Generate customized SLURM job scripts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python generate_slurm_job.py -o my_job.sh
  
  # Customize job parameters
  python generate_slurm_job.py -o custom_job.sh \\
    --account myaccount \\
    --time-limit 24:00:00 \\
    --job-name "My-DRL-Experiment" \\
    --config-name "MC16-TD3.yml" \\
    --nodes 2 \\
    --ntasks-per-node 24
  
  # Schedule job to start in 2 hours
  python generate_slurm_job.py -o scheduled_job.sh \\
    --begin-hours 2 \\
    --mail-user "myemail@domain.com"
        """
    )
    
    # Output file
    parser.add_argument('-o', '--output', 
                       help='Output file for the generated job script')
    
    # Account and Resource Settings
    parser.add_argument('--account', default='deepwing',
                       help='SLURM account (default: deepwing)')
    parser.add_argument('--time-limit', default='12:00:00',
                       help='Time limit in HH:MM:SS format (default: 12:00:00)')
    parser.add_argument('--partition', default='batch',
                       help='SLURM partition (default: batch)')
    parser.add_argument('--no-exclusive', action='store_true',
                       help='Remove exclusive flag')
    parser.add_argument('--nodes', type=int, default=1,
                       help='Number of nodes (default: 1)')
    parser.add_argument('--ntasks-per-node', type=int, default=48,
                       help='Number of tasks per node (default: 48)')
    parser.add_argument('--cpus-per-task', type=int, default=1,
                       help='Number of CPUs per task (default: 1)')
    
    # Job Identification
    parser.add_argument('--job-name', default='DDPG-MINI',
                       help='Job name (default: DDPG-MINI)')
    parser.add_argument('--mail-type', default='ALL',
                       help='Mail notification type (default: ALL)')
    parser.add_argument('--mail-user', default='yuninw@umich.edu',
                       help='Email for notifications (default: yuninw@umich.edu)')
    parser.add_argument('--output-file', default='./log-files/log-ddpg-mini',
                       help='Output file path (default: ./log-files/log-ddpg-mini)')
    parser.add_argument('--error-file', default='./log-files/log-ddpg-mini',
                       help='Error file path (default: ./log-files/log-ddpg-mini)')
    
    # Scheduling
    parser.add_argument('--begin-hours', type=float,
                       help='Start job N hours from now')
    parser.add_argument('--begin-time',
                       help='Specific start time (YYYY-MM-DDTHH:MM:SS)')
    
    # Configuration
    parser.add_argument('--config-name', default='MC16-DDPG.yml',
                       help='Configuration file name (default: MC16-DDPG.yml)')
    parser.add_argument('--run-mode', default='run',
                       help='Run mode (default: run)')
    
    # Execution
    parser.add_argument('--python-mpi-ranks', type=int, default=1,
                       help='MPI ranks for Python process (default: 1)')
    parser.add_argument('--nek5000-mpi-ranks', type=int, default=32,
                       help='MPI ranks for Nek5000 (default: 32)')
    parser.add_argument('--mpi-pml', default='ucx',
                       help='MPI PML (default: ucx)')
    parser.add_argument('--mpi-io', default='ompio',
                       help='MPI IO (default: ompio)')
    
    # Environment
    parser.add_argument('--bashrc-openmpi', default='~/.bashrc.openmpi_ucx',
                       help='OpenMPI bashrc file (default: ~/.bashrc.openmpi_ucx)')
    parser.add_argument('--bashrc-miniforge', default='~/.bashrc.miniforge',
                       help='Miniforge bashrc file (default: ~/.bashrc.miniforge)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SlurmJobGenerator()
    
    # Update parameters
    try:
        generator.update_params(
            account=args.account,
            time_limit=args.time_limit,
            partition=args.partition,
            exclusive=not args.no_exclusive,
            nodes=args.nodes,
            ntasks_per_node=args.ntasks_per_node,
            cpus_per_task=args.cpus_per_task,
            job_name=args.job_name,
            mail_type=args.mail_type,
            mail_user=args.mail_user,
            output_file=args.output_file,
            error_file=args.error_file,
            config_name=args.config_name,
            run_mode=args.run_mode,
            python_mpi_ranks=args.python_mpi_ranks,
            nek5000_mpi_ranks=args.nek5000_mpi_ranks,
            mpi_pml=args.mpi_pml,
            mpi_io=args.mpi_io,
            bashrc_openmpi=args.bashrc_openmpi,
            bashrc_miniforge=args.bashrc_miniforge
        )
        
        # Set begin time if specified
        if args.begin_hours:
            generator.set_begin_time(hours_from_now=args.begin_hours)
        elif args.begin_time:
            generator.set_begin_time(specific_time=args.begin_time)
        
        # Generate script
        generator.generate_job_script(args.output)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
