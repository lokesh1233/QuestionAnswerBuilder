# Stage 1, "build-stage-1", based on Ubuntu, to clone git repo from bitbucket

FROM alpine:latest as build-stage-1

LABEL author="priyank.mishra@speridian.com"

# Update and install SSH
RUN apk update && apk add openssh

# Install GIT
#RUN apk add git
# Make ssh dir
RUN mkdir /root/.ssh/

# Copy over private key, and set permissions
#COPY ./git/ssh-key/id_rsa /root/.ssh/id_rsa
#RUN chmod 600 /root/.ssh/id_rsa

# Create known_hosts
RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
#RUN ssh-keyscan -T 60 git.speridian.com >> /root/.ssh/known_hosts

# Clone the conf files into the docker container
#RUN git clone git@git.speridian.com:AI-ML/S3PO/qna-builder.git
#RUN cd qna-builder && git fetch

# Stage 2, "build-stage-2", based on python 3, to build and load python
FROM python:3.8.0 as build-stage-2


WORKDIR /app

#COPY --from=build-stage-1 /qna-builder/ /app/

#ARG CACHE_BUST

#COPY --from=build-stage-1 /qna-builder/requirements.txt /tmp/
#RUN pip install --requirement /tmp/requirements.txt
#COPY . /tmp/

#RUN python3 -m pip install -r requirements.txt
#RUN pip install torch  torchvision -f https://download.pytorch.org/whl/torch_stable.html















RUN pip install tensorflow==2.3.0
RUN pip install torch -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tensorflow_datasets==3.2.1