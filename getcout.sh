algos=("comix_divide")
#envs=("manyagent_swimmer" "HalfCheetah-v2" "manyagent_ant")
envs=("Humanoid-v2")
for env in ${envs[@]}
do
  for algo in ${algos[@]}
  do
        for i in {0..3}
        do
                mkdir -p /home/huxh/divide/mujoco_results/${algo}/$env/$i/
                docker cp huxh-${algo}-${env}-$i:/home/user/pymarl/results/sacred/3/cout.txt /home/huxh/divide/mujoco_results/${algo}/$env/$i/cout.txt
        done
  done
done
