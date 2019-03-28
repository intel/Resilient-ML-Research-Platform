[![Black Hat Arsenal](https://raw.githubusercontent.com/toolswatch/badges/master/arsenal/usa/2018.svg?sanitize=true)](http://www.toolswatch.org/2018/05/black-hat-arsenal-usa-2018-the-w0w-lineup/)

# Link to <img src="../master/atdml/static/atdml/img/mlsploit.png" height="20"> 
[MLsploit](https://github.com/mlsploit)

# MLsploit Module: Resilient-ML-Research-Platform 

This is a web platform to demo Machine Learning as a Service (MLaaS) on security researches. 
It has a machine learning (ML) pipeline to build and tune models. It also has a portal to demo adversarial ML and countermeasures.

* MLaaS:
  - ML classifier creation and inference 
* MLsploit:
  - Adversarial ML and demos

## Getting Started
### Dependancies
* CentOS or Ubuntu
* Apache Hadoop, Spark & MongoDB
* Python 2.7 and related Python packages, e.g. Scikit-Learn, Numpy, Keras etc 
* Django & SQLite
* Docker (optional for demo containers)

### Installation
* Demo cluster by Docker containers - tiny bigdata platform on your Linux laptop:
```
docker login                    # Login to Docker Hub by your id & password
cd ./docker                     # cd to folder "docker" in git cloned project
chmod 755 *.sh                  # Change scripts to be executable
sudo ./setup_docker_linux.sh    # Create users on Linux and copy related files
./run_container_linux.sh        # Pull images from Docker Hub and run 4 containers:
                                #   HDFS/Spark master & slave1, mongo & Django web 
                                # Access at http://<your machine dns>:8000/ id=demo pwd=demo123
```
* For full installation, please follow the [Setup_Guide_CentOS7.pdf](Setup_Guide_CentOS7.pdf) 
  - Modify web configuation files for setting Hadoop/Spark/web/MongoDB hostnames
    * app.config
    * myml/settings.py
    * atdml/settings.py etc.

## Design Diagrams
### Data Flow:
<p align="center">
  <img src="../master/atdml/static/atdml/img/mlaas_arch_gpu.png" height="320">
</p>

### Software Stack:
<p align="center">
  <img src="../master/atdml/static/atdml/img/sw_stack.png" height="200">
</p>
Note: DNN worker to be released...

## License
This project is licensed under the Apache 2.0 


