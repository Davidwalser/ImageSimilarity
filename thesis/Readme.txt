The common Latex implementation: https://miktex.org/download
Windows version is bundeled with the TexWorks Editor and a package manager 
for addons.

Editors for Latex:
TexMaker: https://www.xm1math.net/texmaker/
TexStudio: https://www.texstudio.org/
OverLeaf (online): https://www.overleaf.com/

Official documentation: https://www.latex-project.org/help/documentation/usrguide.pdf
Tutorial from Uni-Wien: https://homepage.univie.ac.at/albert.georg.passegger/doc/LaTeX-tut.pdf
List of mathematical symbols: https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols

Changing style and font size:
Line 18 in file thesis.tex allows to change the documentstyle, font size and format:
\documentclass[11pt,a4paper,oneside]{scrbook}

Default style is {scrbook} which mimics a book like format with chapters.
Change this to {thesis} for a more paper like style or insert an own style package.

To append new chapters use: 
\chapter{Title}

If you don't want your chapter to be listet in the table of contents, use this instead:
\chapter*{Title}

If you download citations from the internet, choose the BibTex format and add the citations to
the testBib.bib file (can be renamed). Your citations will be handled automatically by Latex.

The file is written using UTF-8 enocoding. Use the same encoding or Latin1 (ISO 8859-1) in your editor 
to avoid issues with certain symbols and letters.
