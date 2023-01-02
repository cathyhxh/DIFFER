env="Humanoid-v2"
agent_conf="9|8"
algo="comix_divide"
#device="1"
for i in {0..3}
do
        name1="huxh-${algo}-${env}-$i"
        docker stop $name1
        docker rm $name1
        docker run --gpus "device=$i" -itd --user $(id -u ${USER}):$(id -g ${USER})\
                --name=$name1 --shm-size 8G -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro\
                 pymarl:facmac /bin/bash
        docker cp /home/huxh/divide/src $name1:/home/user/pymarl
        #docker exec -d -u root -it $name1 /bin/bash -c "python3 src/main.py --config=$algo --env-config=mujoco_multi with env_args.scenario_name=$env env_args.agent_conf=$agent_conf env_args.agent_obsk=0" 
        docker exec -d -u root -it $name1 /bin/bash -c "python3 src/main.py --config=comix_pp --env-config=mujoco_multi with env_args.scenario_name=Humanoid-v2 env_args.agent_conf='9|8' env_args.agent_obsk=0" 
done

