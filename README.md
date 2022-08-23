# P8_DUCHEMIN_Marin

Please note that due to AWS security standard, the current repository is not functionnal. You need to create your own bucket and security key.

In this project, we will create a scalable fruit classifier based on photos. 

![](https://github.com/Marin-GHUB/P8_DUCHEMIN_Marin/blob/main/Soutenance/Ressources/Diagram.png?raw=true)

Thanks to an AWS S3 storage solution and an AWS EC2 server, we will make a model than can stock and process images to classify them. 
This need some ssh protocol to communicate between the server, and the use of parallelization for the scalability (using PySpark). For the processing of the images, the technique is similar to the one found in the 6th project (https://github.com/Marin-GHUB/P6_DUCHEMIN_Marin).
