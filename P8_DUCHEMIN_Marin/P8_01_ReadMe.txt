Pour ce projet, un notebook jupyter écrit en python a été créé et utilisé sur un serveur EC2. Ce nobebook récupère les images stockées sur un autre serveur S3, les traite puis enregistre le résultat sur le serveur S3.

Un exemple d'image et du résultat de son traitement sont fournis dans ce dossier.

Des GIF montrant la navigation sur les différents serveurs sont également fournis dans ce dossier.
Le GIF à propos du S3 montre les différents dossiers et fichiers contenu sur le serveur. Il dure environ 1 minute.

Le GIF à propos du EC2 montre comment se connecter en protocole ssh sur celui-ci à l'aide du terminal. Il parcourt ensuite les dossiers et fichiers puis lance une session jupyter lab en localhost. Avec un second terminal, toujours en protocole ssh, on se connecte à la session jupyter du serveur EC2 puis y accède depuis un navigateur web via un token. IL dure environ 1 minute et 20 secondes.



Ci-dessous les différents liens des deux serveurs :
    EC2 :
Elastic IP Address : 54.228.168.107
public IPV4 adress : ec2-54-228-168-107.eu-west-1.compute.amazonaws.com

    S3 :
bucket : https://s3.console.aws.amazon.com/s3/buckets/md-ocr-p8-bucket?region=eu-west-1&tab=objects
