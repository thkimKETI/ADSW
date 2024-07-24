# Description
1. Installation

   
1.1 Nvidia-Docker Installation
   ```ruby
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update

   sudo apt-get install -y nvidia-docker2

   sudo systemctl restart docker 
   ```

1.2 Nvidia-Docker Container 
```ruby
git clone https://github.com/WongKinYiu/yolov7.git

sudo nvidia-docker run --name yolov7 -v /home/keti/tkim/yolov7/:/yolov7/ -v /data/AD2/AD_2023_0210ver/:/yolov7/AD2_DB/ -it --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3

apt update
apt install -y zip htop screen libgl1-mesa-glx
pip install seaborn thop

cd /yolov7
```

3. Validation
4. Demo
5. ONNX Transform
