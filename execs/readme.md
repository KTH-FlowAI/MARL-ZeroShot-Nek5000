# Running Jobs 

# In Local 

    ./unified-script MC16-TD3 run


# For SBATCH: 

1. Please generate the script first: 

   For for customize configs please look at the [../utils/sjob_gen/readme.md](/utils//sjob_gen/readme.md), some examples: 


    Basic usage
    
        ../utils/sjob_gen/create_job.sh -o my_job.sh
        
    Custom configuration
        
        ../utils/sjob_gen/create_job.sh -o custom_job.sh \
          -a myaccount \
          -t 24:00:00 \
          -j "My-DRL-Experiment" \
          -c "MC16-TD3.yml" \
          -e "myemail@domain.com"
        
    Schedule job to start in 2 hours
    
        ../utils/sjob_gen/create_job.sh -o scheduled_job.sh -s 2

2. submit your job via: 

        sbatch mysjob.sh
    
