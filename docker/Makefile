SHELL := /bin/bash
default:
	echo 'make {build, pull}'

build:
	docker build -t life .
	#docker login -u rliston -p FYxOyEkz+1E3QV315mX3amN128urdoLIBPuWCNS1TUn6oU3a7UY+2Yg+jPCYtl9h containers.cisco.com
	#docker push containers.cisco.com/rliston/deepphy

pull:
	for HOST in hydra-01 hydra-02 hydra-03 hydra-04 hydra-05 hydra-06 hydra-07 hydra-08 hydra-09 hydra-10 ; do \
		echo $$HOST ; \
		docker -H ssh://$$HOST pull containers.cisco.com/rliston/deepphy:latest ; \
	done
