docker stop `docker ps -a | grep 'comix_pp' | awk '{print $1}'`
docker rm `docker ps -a | grep 'comix_pp' | awk '{print $1}'`

