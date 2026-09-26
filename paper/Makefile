PDF=main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex references.bib
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.bcf main.blg main.log main.out main.pdf main.run.xml
