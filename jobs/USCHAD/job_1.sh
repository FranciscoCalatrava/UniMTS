#!/usr/bin/env bash
#SBATCH -A NAISS2024-22-1123 -p alvis
#SBATCH -N 1 --gpus-per-node=A40:1
#SBATCH -t 0-20:00:00

module load Python/3.9.6-GCCcore-11.2.0;
source /cephyr/users/fracal/Alvis/UniMTS/SET_ENVIRONMENT_VARIABLES.sh;
source /mimer/NOBACKUP/groups/focs/virtualenvs/.unimts/bin/activate;

python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 1 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_1_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 2 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_2_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 3 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_3_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 4 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_4_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 5 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_5_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 6 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_6_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 7 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_7_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 8 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_8_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 9 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_9_Multisensor_Seed_1.log
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 10 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_10_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 11 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_11_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 12 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_12_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 13 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_13_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 14 --seed 1 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_14_Multisensor_Seed_1.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 1 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_1_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 2 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_2_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 3 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_3_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 4 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_4_Multisensor_Seed_2.log ;
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 5 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_5_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 6 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_6_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 7 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_7_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 8 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_8_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 9 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_9_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 10 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_10_Multisensor_Seed_2.log & 
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 11 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_11_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 12 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_12_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 13 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_13_Multisensor_Seed_2.log ;
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 14 --seed 2 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_14_Multisensor_Seed_2.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 1 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_1_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 2 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_2_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 3 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_3_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 4 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_4_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 5 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_5_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 6 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_6_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 7 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_7_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 8 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_8_Multisensor_Seed_3.log;
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 9 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_9_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 10 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_10_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 11 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_11_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 12 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_12_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 13 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_13_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 14 --seed 3 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_14_Multisensor_Seed_3.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 1 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_1_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 2 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_2_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 3 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_3_Multisensor_Seed_4.log
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 4 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_4_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 5 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_5_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 6 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_6_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 7 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_7_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 8 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_8_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 9 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_9_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 10 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_10_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 11 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_11_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 12 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_12_Multisensor_Seed_4.log 
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 13 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_13_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 14 --seed 4 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_14_Multisensor_Seed_4.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 1 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_1_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 2 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_2_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 3 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_3_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 4 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_4_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 5 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_5_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 6 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_6_Multisensor_Seed_5.log 
wait
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 7 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_7_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 8 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_8_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 9 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_9_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 10 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_10_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 11 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_11_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 12 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_12_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 13 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_13_Multisensor_Seed_5.log &
python3 /cephyr/users/fracal/Alvis/UniMTS/finetune_1.py --mode full --batch_size 32 --num_epochs 200  --checkpoint './checkpoint/UniMTS.pth' --data_path /mimer/NOBACKUP/groups/focs/ --device cuda:0 --dataset USCHAD --experiment 14 --seed 5 >> /mimer/NOBACKUP/groups/focs/log/USCHAD_UNIMTS_14_Multisensor_Seed_5.log 
wait