envs=("academy_3_vs_1_with_keeper" "academy_counterattack_hard" "academy_counterattack_easy")
algo="PER_weight"
for env in ${envs[@]}
do
    for i in {0..3}
    do
        name1="huxh-${algo}-warm_up-0.8-${env}-$i"
        docker stop $name1
        docker rm $name1
	    device=$(($i))
        docker run --gpus "device=$device" -itd \
                --name=$name1 --shm-size 8G -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro\
                -w /root/pymarl pymarl:gf1 /bin/bash
        docker cp /home/huxh/divide/src $name1:/root/pymarl/
        docker cp /home/huxh/divide/gf/gf_code/academy_3_vs_1_with_keeper.yaml  $name1:/root/pymarl/src/config/envs/academy_3_vs_1_with_keeper.yaml
        docker cp /home/huxh/divide/gf/gf_code/academy_counterattack_easy.yaml  $name1:/root/pymarl/src/config/envs/academy_counterattack_easy.yaml
        docker cp /home/huxh/divide/gf/gf_code/academy_counterattack_hard.yaml  $name1:/root/pymarl/src/config/envs/academy_counterattack_hard.yaml
        docker cp /home/huxh/divide/gf/gf_code/envs  $name1:/root/pymarl/src/
        docker cp /home/huxh/divide/gf/gf_code/football_env_core.py  $name1:/opt/conda/lib/python3.8/site-packages/gfootball/env/
        docker cp /home/huxh/divide/gf/gf_code/observation_processor.py  $name1:/opt/conda/lib/python3.8/site-packages/gfootball/env/
        docker exec -d -u root -it $name1 /bin/bash -c "python3 src/main.py --config=qmix --env-config=$env with learner=q_divide_learner selected=${algo} warm_up=True selected_alpha=0.8" 
    done
done
