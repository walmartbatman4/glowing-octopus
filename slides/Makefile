.PHONY: all pdf notes clean check tables

all: pdf notes

tables:
	python scripts/make_tables.py

pdf: tables
	latexmk -pdf -interaction=nonstopmode main.tex

notes: tables
	latexmk -pdf -jobname=main_notes -interaction=nonstopmode -pdflatex="pdflatex '\PassOptionsToPackage{show notes}{beamer}\input{main}'" main.tex

check:
	python scripts/check_slides.py

clean:
	latexmk -C
	rm -rf *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk
