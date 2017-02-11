FROM python:2.7.13-slim

RUN apt-get update -y && apt-get install wget unzip gcc git python-tk make -y
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr && \
  make && \
  make install

RUN pip install pandas TA-Lib matplotlib texttable git+https://github.com/gabrielpreti/pyalgotrade.git

COPY . /opt/pytrade/
WORKDIR /opt/pytrade
ENTRYPOINT ["python"]
