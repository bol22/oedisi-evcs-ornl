FROM python:3.10.6-slim-bullseye
RUN apt-get update
RUN apt-get install -y git ssh
RUN git clone https://github.com/openEDI/oedisi.git /oedisi
RUN pip install /oedisi 
RUN mkdir Recorder
COPY  . ./Recorder
WORKDIR ./Recorder
RUN pip install -r requirements.txt
EXPOSE 5680/tcp
CMD ["python", "server.py"]
