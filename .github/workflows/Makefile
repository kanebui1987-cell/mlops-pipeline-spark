# ============================
# Makefile for Python Project
# Compatible with Windows (Git Bash), Linux, WSL
# ============================

PYTHON=python
PIP=$(PYTHON) -m pip
VENV=.venv
ACTIVATE=$(VENV)/Scripts/activate

# -------- ENV SETUP ----------
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Run: source $(ACTIVATE) to activate environment"

install:
	$(ACTIVATE) && $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt

freeze:
	$(ACTIVATE) && $(PIP) freeze > requirements.txt

# -------- CODE QUALITY ----------
format:
	$(ACTIVATE) && black .
	$(ACTIVATE) && isort .

lint:
	$(ACTIVATE) && flake8 .
	$(ACTIVATE) && pylint **/*.py || true

# -------- TESTING ----------
test:
	$(ACTIVATE) && pytest -v --disable-warnings --maxfail=1

coverage:
	$(ACTIVATE) && pytest --cov=src --cov-report=html

# -------- RUN ----------
run:
	$(ACTIVATE) && $(PYTHON) main.py

cli:
	$(ACTIVATE) && $(PYTHON) -m src.cli

# -------- DOCKER ----------
docker-build:
	docker build -t my-python-app:latest .

docker-run:
	docker run -it --rm -p 8000:8000 my-python-app:latest

# -------- CLEAN UP ----------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf $(VENV)

# -------- UTIL ----------
info:
	$(ACTIVATE) && python -V
	$(ACTIVATE) && which python
	$(ACTIVATE) && pip list

