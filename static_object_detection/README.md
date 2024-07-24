# Description

1. Nvidia-Docker 환경 구성 및 YOLOv7 설치    
1.1 Nvidia-Docker 설치
   ```ruby
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   ```
   ```ruby
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker 
   ```
   1.2 YOLOv7 Container 생성
   ```ruby
   git clone https://github.com/WongKinYiu/yolov7.git
   ```
   ```ruby
   sudo nvidia-docker run --name yolov7 -v /home/keti/tkim/yolov7/:/yolov7/ -v /data/AD2/AD_2023_0210ver/:/yolov7/AD2_DB/ -it --shm-size=64g nvcr.io/nvidia/pytorch:21.08-py3
   ```
   ```ruby
   apt update
   apt install -y zip htop screen libgl1-mesa-glx
   pip install seaborn thop
   cd /yolov7
   ```
   1.3 YOLOv7 설치 확인 
   ```ruby
   wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
   ```
   ```ruby
   python detect.py --weights yolov7.pt --conf 0.25 --imag-size --source infernce/image3.jpg
   ```

2. Data Preparation
   2.1 AI-hub, 신호등/도로표지판 인지 영상(수도권) 활용
   ```ruby
   https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=188
   ```
   2.2 AI-hub 라벨링 형식 변경 (AI-Hub -> YOLOtxt
   ```ruby
   python aihub_to_yolotxt.py
   ```
4. Demo
5. ONNX Transform
