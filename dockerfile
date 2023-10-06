# Use Python 3.11.4 as the base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# RUN mkdir app

# # Set the working directory to /app
WORKDIR /usr/src/app

# Copy the requirements.txt file to the image
COPY requirements.txt /usr/src/app

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y \
    python3-pip \
    python3
    # Install the dependencies using pip
RUN pip install -r requirements.txt

# Copy the app.py file to the image
ADD . /usr/src/app

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

# Run the app when the container starts
# ENTRYPOINT [ "python3" ]
