FROM python:3.10.6-slim-bullseye
#USER root
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh

WORKDIR /simulation

COPY system.json .
COPY components.json .
COPY LocalFeeder LocalFeeder
COPY recorder recorder
COPY evcs_federate evcs_federate
COPY evcs.ipynb .

RUN mkdir -p outputs build

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--no-browser"]