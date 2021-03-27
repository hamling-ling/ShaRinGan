# How to setup
#
# Build docker image
# > docker build -f Dockerfile_AVX \
#	--build-arg user=$(id -un) \
#	--build-arg user_id="$(id -u)" \
#   --build-arg user_grp="$(id -gn)" \
#	--build-arg user_gid="$(id -g)" \
#	--build-arg pass=<YOUR PASSWORD> \
#	-t tensor-plus .
#
# See image just created
# > docker images
#
# Run container
# > docker run --rm -it tensor-plus

FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter
ARG user="tfadmin"
ARG pass="tfadmin"
ARG user_id=1000
ARG user_grp="tfadmin"
ARG user_gid=1000

LABEL maintainer="hamling-ling"

RUN apt-get update && \
    apt-get install -y \
        libsndfile1-dev \
        sudo

RUN pip install --upgrade pip
RUN pip install seaborn keras
RUN pip install pysoundfile

# User
RUN echo ${user}
RUN addgroup --gid ${user_gid} ${user_grp}
RUN useradd -u ${user_id} -g ${user_gid} -G sudo -s /bin/bash -d /home/${user} -m ${user}

RUN echo ${user}:${pass} | chpasswd
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo ${user}" ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${user}
WORKDIR /home/${user}

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password=''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token=''" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.ip='0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py 
RUN echo "c.NotebookApp.port=8888" >> ~/.jupyter/jupyter_notebook_config.py
