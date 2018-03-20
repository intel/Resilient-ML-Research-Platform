# Resilient-ML-Research-Platform 
(This doc is under construction)

This is a web platform to demo Machine Learning as-a Service on security researches. 
It has machine learning (ML) pipeline to analyze data and also is a portal for backend robot to emulate uploaded binaries.
It demos adversarial ML and countermeasures.

## Getting Started
### Prerequisites
* Apache Hadoop & Spark
* MongoDB
* Scikit-Learn, Numpy, Keras and related Python packages
* Django & SQLite

### Installing
* Please follow the [Setup_Guide_CentOS7.pdf](Setup_Guide_CentOS7.pdf) for installation
* Modify web files app.config, myml/settings.py, atdml/settings.py for Hadoop/Spark/web/MongoDB hostnames

## Design Diagrams
### Architecture:
<img src="../master/atdml/static/atdml/img/mlaas_arch_gpu.png" height="320">

### Software Stack:
<img src="../master/atdml/static/atdml/img/sw_stack.png" height="200">

## License
This project is licensed under the Apache 2.0 


