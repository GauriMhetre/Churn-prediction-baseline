.PHONY: data db sql test lint clean

data:
	python scripts/download_data.py

db:
	python -m src.data --build-db

sql:
	python -m src.sqlrun

test:
	pytest -q

lint:
	ruff check .

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('data').glob('*.db') if p.is_file()]"
