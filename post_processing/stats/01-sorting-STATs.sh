#--------------------------------------
#A script for coping the sts file from the folder 
#--------------------------------------

#=======================
# PARAM
CASE_NAME="tcf"
#Locate the PATH 
DATA_PATH="../../03-results"
# 
TARGET_PATH='DATA'
# 
DATA_TYPE='sts'
MESH_TYPE='c2D'
#=======================

fileNUM=$(ls $DATA_PATH/res_* -dq |  wc -l)
echo "[FOLDER] ${fileNUM} FOLDERS"


# Step2: A for loop 
NUM_TAR=0
for i in `seq 1 $fileNUM`
do
    echo "[ITER] At ${i}"
    NUM_STS=$(ls $DATA_PATH/res_${i}/sts_data/${DATA_TYPE}${CASE_NAME}* -dq | wc -l)
    echo "[FOLDER] ${NUM_STS} FILES in $DATA_PATH/res_${i}/sts_data/${DATA_TYPE}"
	for j in `seq 1 $NUM_STS`
	do
		# Global number 
        ((k=j+NUM_TAR))
        
        # Name the files 
        id_padd1=$(printf "%05d" $j)
        id_padd2=$(printf "%05d" $k)

        fromData="$DATA_PATH/res_${i}/sts_data/${DATA_TYPE}${CASE_NAME}0.f${id_padd1}"
        toData="$TARGET_PATH/${DATA_TYPE}${CASE_NAME}0.f${id_padd2}"
        
        # If it does not exist then copy that 
        if [ -f "$toData" ]; then
            echo "[DATA] ${toData} exists."
        else
            echo "[DATA] COPY from ${fromData}"
            echo "[DATA] TARGET ${toData}."
		    cp $fromData $toData
        
        fi

	done
    ((NUM_TAR=${NUM_TAR}+${NUM_STS}))
    echo "[FOLDER] ${NUM_TAR} IN TARGET FODLER"
    
done
# ((TOT_NUM=NUM_RUNS*NUM_STS))