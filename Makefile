SHELL := /bin/bash
STACK_NAME = "pytrade-stack"
MY_IP = `dig +short myip.opendns.com @resolver1.opendns.com`
SSH_KEY_FILE = "~/Documents/aws_keys/pytradejobs.pem"

build:
	docker build -t pytrade .

run:
	docker run -d --name pytrade -v /export/pytrade/:/export/pytrade/ pytrade python init.py

rm:
	docker rm pytrade

stop:
	docker stop pytrade

logs:
	docker logs pytrade --follow=true --tail=1000

speak:
	echo ${PHRASE}

create-aws-stack:
	aws cloudformation create-stack --stack-name=${STACK_NAME} --template-body=file://./pytrade-aws.yaml --parameters ParameterKey=MyIp,ParameterValue=${MY_IP} ParameterKey=ProjectName,ParameterValue='Pytrade'

delete-aws-stack:
	aws cloudformation delete-stack --stack-name=${STACK_NAME}

get-aws-stack_status:
	aws cloudformation  describe-stacks --stack-name=${STACK_NAME} --query 'Stacks[0].StackStatus' --output text

ssh-into-instance:
	set -e ;\
	INSTANCE_IP=$$(aws cloudformation  describe-stacks --stack-name=${STACK_NAME} --query "Stacks[0].Outputs[?OutputKey=='Ec2InstancePublicIp'].OutputValue" --output text) ;\
	ssh -i ${SSH_KEY_FILE} ec2-user@$${INSTANCE_IP} ;\