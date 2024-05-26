#!/bin/bash

# Render the current directory with Quarto
quarto render .

# Convert all .qmd files to .ipynb
for file in notebooks/*.qmd; do
  quarto convert "$file"
done