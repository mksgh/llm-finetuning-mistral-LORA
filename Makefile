install: requirements.txt
	pip install --upgrade pip &&\
    pip install -r requirements.txt

format:
	black *.py
