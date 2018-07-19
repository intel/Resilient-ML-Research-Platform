[![Black Hat Arsenal](https://raw.githubusercontent.com/toolswatch/badges/master/arsenal/usa/2018.svg?sanitize=true)](http://www.toolswatch.org/2018/05/black-hat-arsenal-usa-2018-the-w0w-lineup/)

# Resilient-ML-Research-Platform 

This is a web platform to demo Machine Learning as a Service (MLaaS) on security researches. 
It has a machine learning (ML) pipeline to analyze data and also is a portal for backend user to emulate uploaded binaries.
It demos adversarial ML and countermeasures.

## Getting Started
### Dependancies
* CentOS or Ubuntu
* Apache Hadoop & Spark
* MongoDB
* Scikit-Learn, Numpy, Keras and related Python packages
* Django & SQLite
* for Python 2.7

### Installation
* Big data platform on one machine demo by Docker containers
```
docker login                    # login to Docker Hub
cd ./docker..                   # cd to docker after git clone
chmod 755 *.sh                  # change script to executable
sudo ./setup_docker_linux.sh    # create users and copy files
./run_container_linux.sh        # run containers
```
* For full installation, follow the [Setup_Guide_CentOS7.pdf](Setup_Guide_CentOS7.pdf) 
  - Modify web files app.config, myml/settings.py, atdml/settings.py for Hadoop/Spark/web/MongoDB hostnames

## Design Diagrams
### Architecture:
<img src="../master/atdml/static/atdml/img/mlaas_arch_gpu.png" height="320">

### Software Stack:
<img src="../master/atdml/static/atdml/img/sw_stack.png" height="200">

Note: backend robots and DNN worker are not included in this project.

## License
This project is licensed under the Apache 2.0 


