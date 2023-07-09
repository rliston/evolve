FROM tensorflow/tensorflow:latest
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt-get upgrade -y

# MISC
RUN apt-get update && apt-get install -y cmake git yasm pkg-config vim eog bc mplayer sudo curl zip tcpdump tcptrace wireshark-gtk tshark openssh-client
RUN apt-get update && apt-get install -y less time
RUN apt-get update && apt-get install -y gedit 
RUN apt-get update && apt-get install -y x11-apps
# LIFE
RUN pip install python-lifelib
RUN python -c 'import lifelib ; sess = lifelib.load_rules("b3s23")'
RUN pip install pygments prompt_toolkit
RUN pip install jupyterlab
RUN apt-get update && apt-get install -y expect
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install treelib

RUN ldconfig
