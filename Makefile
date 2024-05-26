create-env:
	python3 -m venv venv

setup:
	source venv/bin/activate

stop:
	deactivate

review:
	python3 agent.py