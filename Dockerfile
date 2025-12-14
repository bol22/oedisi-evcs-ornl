FROM python:3.10.6-slim-bullseye
RUN apt-get update && apt-get install -y git ssh

RUN mkdir -p /root/.ssh

WORKDIR /simulation

# Copy configuration files
COPY system.json .
COPY components.json .

# Copy component directories
COPY LocalFeeder LocalFeeder
COPY recorder recorder
COPY evcs_federate evcs_federate
COPY broker broker

# Copy notebook
COPY evcs.ipynb .

RUN mkdir -p outputs build

# Install dependencies
COPY LocalFeeder/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install oedisi==2.0.2 jupyter xarray pyarrow

EXPOSE 8888

# Default: Run oedisi build and run
CMD ["sh", "-c", "oedisi build --system system.json --component-dict components.json && oedisi run"]