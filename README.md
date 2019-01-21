# nlp_097215
This guide was tested on ubuntu 18.04
This folder contains:
	code/ -  source code for experiments reproduction
	comp.labeled - the annotated competition file.
	report.pdf - report of the models development process.

Installation:
Open a terminal and run the following commands:
	virtualenv --python=/usr/bin/python3 venv
	source venv/bin/activate
	pip install -r requirements.txt
	
Now, when the virtual environment is installed, you can run the following commands:
	cd code/
	python ex2m2.py                    - this trains Model 2 from scratch and annotate the competition file by the last updated model.
	python ex2m2_pretrained.py    - this run a pre-trained model to reproduce the reported results.
	python ex2m1.py                    - this trains Model 1 from scratch and annotate the competition file by the last updated model.
	
After running the code you can view the learning curve and confusion matrix in the results directory: code/saved_models/<date>/

