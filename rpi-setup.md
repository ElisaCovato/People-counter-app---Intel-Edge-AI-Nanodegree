# RaspberryPi - Initial Setup

### Install Intel® Distribution of OpenVINO™ Toolkit

Refer to [this page?](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html) for more information about how to install and setup the Intel® Distribution of OpenVINO™ Toolkit.


### Install Nodejs and its dependencies

Follow the guide [here](https://www.instructables.com/id/Install-Nodejs-and-Npm-on-Raspberry-Pi/) to install NodeJS and npm.

### Install the following dependencies

```
sudo apt update
sudo apt-get install python3-pip
pip3 install numpy
pip3 install paho-mqtt
sudo apt install libzmq3-dev libkrb5-dev
sudo apt-get install cmake
```


### Install ffserver
This project makes use of FFmpeg’s `ffserver` functionality, which was deprecated in an older version. As such, a new install of ffmpeg will not include it if you do it directly from `brew. You can use the below to install the older version containing `ffserver`, as detailed in [this post](https://superuser.com/questions/1296377/why-am-i-getting-an-unable-to-find-a-suitable-output-format-for-http-localho/1297419#1297419).

```
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg
cd ffmpeg
git checkout 2ca65fc7b74444edd51d5803a2c1e05a801a6023
./configure
make -j4
```