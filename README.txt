DATA MINING GROUP 3 - DOCKER

This project is a part of UMich BDSI. To be shared with other groups in the Data Mining group.


DOCKER PULL COMMAND

docker pull sashalioutikova/bdsi-dm3-python


GIT PULL COMMAND

git clone https://github.com/ironic-oq-squad/final.git


RUNNING THE CONTAINER
NOTE: You should run this in your local Git repository

docker run -it --name dm3-python -v "$PWD":/var/app/bdsi -w /var/app/bdsi bdsi-dm3-python


RUNNING THE FILE(S)

python3 ./src/P1_P3_extended.py

To run particular Python files, run
python3 ./src/file.py

Specific instructions:
python3 ./src/P1_DistanceMatrix.py [filepath to data] [tweet index column in csv]
python3 ./src/P3_synonyms.py [filepath to clean data]
python3 ./src/P1_P3_extended.py [filepath to data] [tweet index column in csv]
e.g. python3 ./src/P1_P3_extended.py ./data/Twitter_mani.csv 4


NOTA BENE
If any Python packages are missing, please install them using
pip3 install package
Also do tell Data Mining Group 3 of any missing packages.

