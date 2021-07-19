DATA MINING GROUP 3 - DOCKER

This project is a part of UMich BDSI. To be shared with other groups in the Data Mining group.


DOCKER PULL COMMAND

docker pull sashalioutikova/bdsi-dm3-python

If the pull fails, make your own Docker image using files in the GitHub with the following command:
docker build -t bdsi-dm3-python . 


GIT PULL COMMAND

git clone https://github.com/ironic-oq-squad/final.git

Note that you should bring your own data files into the ./data folder on your local repo to run this code on your own data.


RUNNING THE CONTAINER
NOTE: You should run this in your local Git repository

docker run -it --name dm3-python -v "$PWD":/var/app/bdsi -w /var/app/bdsi sashalioutikova/bdsi-dm3-python


RUNNING THE FILE(S)

python3 ./src/P1_P3_extended.py

To run particular Python files, run
python3 ./src/file.py

Specific instructions:
python3 ./src/P1_DistanceMatrix.py [filepath to data] [tweet index column in csv]
python3 ./src/P3_synonyms.py [filepath to clean data]
python3 ./src/P1_P3_extended.py [filepath to data] [tweet index column in csv]
e.g. python3 ./src/P1_P3_extended.py ./data/Twitter_mani.csv 4


JUPYTER
If you need to use Docker to use Jupyter notebooks, use the following instructions:
Run the container:
docker run -p 8888:8888 -it --name dm3-python -v "$PWD":/var/app/bdsi -w /var/app/bdsi sashalioutikova/bdsi-dm3-python

Install Jupyter:
pip3 install jupyter

Run Jupyter notebook:
jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root


NOTA BENE
If any Python packages are missing, please install them using
pip3 install package
Also do tell Data Mining Group 3 of any missing packages.

