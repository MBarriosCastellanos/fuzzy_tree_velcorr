# Motor: pdflatex con codificacion y errores linea a linea
$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# BibTeX: 2 = ejecuta bibtex aunque el .bbl este en el arbol de salida
$bibtex_use = 2;

# Deja que latexmk repita pdflatex las veces necesarias hasta resolver
# todas las referencias/citas (evita tener que compilar 2 veces a mano)
$max_repeat = 5;

# Extensiones auxiliares que 'latexmk -c' debe limpiar
$clean_ext = 'aux bbl blg out spl fdb_latexmk fls synctex.gz';
