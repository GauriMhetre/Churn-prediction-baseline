.PHONY: data db sql split stats train eval final test lint clean all

data:
	python scripts/download_data.py

db:
	python -m src.data --build-db

sql:
	python -m src.sqlrun

split:
	python -m src.data --split

stats:
	python -m src.stats

train:
	python -m src.train

eval:
	python -m src.evaluate

final:
	python -m src.evaluate --final

test:
	pytest -q

lint:
	ruff check .

clean:
	python -c "import pathlib; [p.unlink() for p in pathlib.Path('data').glob('*.db') if p.is_file()]"

all: data db sql split stats train eval
