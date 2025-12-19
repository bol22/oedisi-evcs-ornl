FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN git clone https://github.com/openEDI/oedisi.git /oedisi
RUN pip install /oedisi 
RUN mkdir EVCS
COPY  . ./EVCS
WORKDIR ./EVCS
RUN pip install -r requirements.txt
EXPOSE 5683/tcp
CMD ["python", "server.py"]
