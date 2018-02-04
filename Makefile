SHELL := /bin/bash

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