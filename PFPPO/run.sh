# python FL_DQN.py 1 &
# python FL_DQN.py 2 &
# wait

# for i in {1..12};
# do
#     python test_DPG.py $i;
# done

# python Cent_PPO.py 5 0.05 1 & 
# wait



# python Dist_PPO.py 5 0.05 2 &
# python Dist_PPO.py 5 0.1 2 &
# python Dist_PPO.py 5 0.15 2 &
# python Dist_PPO.py 5 0.2 2 &
# wait

# python Dist_PPO.py 5 0.05 4 &
# wait


# python FL_PPO.py 5 0.15 100 5 &
# python FL_PPO.py 5 0.2 100 5 &
# python FL_PPO.py 5 0.15 1000 5 &
# wait



# python PFPPO.py 5 0.05 10 0.5 0.5 1 &
# wait

# python PFPPO.py 5 0.05 10 0.2 0.8 1 &
# wait

python PFPPO.py 5 0.05 10 0.2 0.8 2 &
wait

python PFPPO.py 5 0.05 10 0.5 0.5 2 &
wait

python PFPPO.py 5 0.05 10 0.8 0.2 2 &
wait