Rebuild all_tables.pdf

1) (if needed) install TeX packages:

```bash
sudo apt-get update && \
  sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

2) compile (run twice):

```bash
cd '/home/frederik/Code/Heyen & Lehtomaa 2021  - Dynamic coalition formation/results'
pdflatex -interaction=nonstopmode all_tables.tex
pdflatex -interaction=nonstopmode all_tables.tex
```

Or as a one-liner:

```bash
cd '/home/frederik/Code/Heyen & Lehtomaa 2021  - Dynamic coalition formation/results' && pdflatex -interaction=nonstopmode all_tables.tex && pdflatex -interaction=nonstopmode all_tables.tex
```
