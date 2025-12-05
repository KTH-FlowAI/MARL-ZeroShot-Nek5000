#!/bin/bash
# Simple wrapper script for the SLURM job generator
# This provides a more user-friendly interface for common use cases

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATOR="$SCRIPT_DIR/generate_slurm_job.py"

# Default values
ACCOUNT="deepwing"
TIME_LIMIT="12:00:00"
JOB_NAME="DDPG-MINI"
CONFIG_NAME="MC16-DDPG.yml"
OUTPUT_FILE=""
MAIL_USER="yuninw@umich.edu"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --account ACCOUNT        SLURM account (default: $ACCOUNT)"
    echo "  -t, --time TIME_LIMIT       Time limit in HH:MM:SS (default: $TIME_LIMIT)"
    echo "  -j, --job-name NAME         Job name (default: $JOB_NAME)"
    echo "  -c, --config CONFIG_FILE    Config file name (default: $CONFIG_NAME)"
    echo "  -o, --output OUTPUT_FILE    Output script file (required)"
    echo "  -e, --email EMAIL           Email for notifications (default: $MAIL_USER)"
    echo "  -n, --nodes NODES           Number of nodes (default: 1)"
    echo "  -r, --ranks RANKS           MPI ranks for Nek5000 (default: 32)"
    echo "  -s, --schedule HOURS        Schedule job to start in N hours"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -o my_job.sh"
    echo "  $0 -o custom_job.sh -a myaccount -t 24:00:00 -j \"My-Experiment\""
    echo "  $0 -o scheduled_job.sh -s 2 -e myemail@domain.com"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--account)
            ACCOUNT="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -j|--job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -e|--email)
            MAIL_USER="$2"
            shift 2
            ;;
        -n|--nodes)
            NODES="$2"
            shift 2
            ;;
        -r|--ranks)
            NEK5000_RANKS="$2"
            shift 2
            ;;
        -s|--schedule)
            SCHEDULE_HOURS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if output file is specified
if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: Output file is required. Use -o or --output to specify."
    echo ""
    show_usage
    exit 1
fi

# Build the command
CMD="python $GENERATOR -o $OUTPUT_FILE"
CMD="$CMD --account $ACCOUNT"
CMD="$CMD --time-limit $TIME_LIMIT"
CMD="$CMD --job-name \"$JOB_NAME\""
CMD="$CMD --config-name $CONFIG_NAME"
CMD="$CMD --mail-user $MAIL_USER"

# Add optional parameters
if [[ -n "$NODES" ]]; then
    CMD="$CMD --nodes $NODES"
fi

if [[ -n "$NEK5000_RANKS" ]]; then
    CMD="$CMD --nek5000-mpi-ranks $NEK5000_RANKS"
fi

if [[ -n "$SCHEDULE_HOURS" ]]; then
    CMD="$CMD --begin-hours $SCHEDULE_HOURS"
fi

# Execute the command
echo "Generating SLURM job script..."
echo "Command: $CMD"
echo ""

eval $CMD

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Job script generated successfully: $OUTPUT_FILE"
    echo ""
    echo "To submit the job:"
    echo "  sbatch $OUTPUT_FILE"
    echo ""
    echo "To view job status:"
    echo "  squeue -u \$USER"
else
    echo ""
    echo "❌ Failed to generate job script"
    exit 1
fi
