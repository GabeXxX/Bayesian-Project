# Bayesian-Project
presentazione 24/11 https://www.overleaf.com/project/65521ef6a20cdec448e00397
palette, labels plots  [Azzurro: #3399ff, Arancione: #f4963e, Rosso-Fragola: #f94e56, Verde: #62a87c]

TO DO LIST
1. studiare e fare allineamento:
\\1.1capire come si fa in maniera basic
  1.2 farlo sulle nostre curve in maniera basic
  1.3 capire come si fa per via bayesiana e se c è tempo farlo bayesianamente
Starting Point: https://cran.r-project.org/web/packages/fdasrvf/fdasrvf.pdf

Per ora: abbiamo provato ad utilizzare la libreria di python "fdasrsf" analogo di ‘fdasrvf’ per R ma la funzione che fa alignment in framework bayesiano allinea due curve alla volta
Proveremo ad utilizzare ‘fdasrvf’ su python o in alternativa direttamente su R


3. testare a fondo l'unconstrained model:
  2.1 aprire un nuovo script jupyter notebook, importare la classe che abbiamo creato dal modello definitivo
  2.2 testarlo su vari kernel della prior, aumentando il numero di covariate

 - For now: tested unconstrained model on simple white noise kernel, builded a function that allow to scale testing easily on different kernel with some error statistics

4. risistemare il codice prodotto finora:
  3.1 controllare il codice
  3.2 aggiornarlo se serve, commentare/aggiungere descrizioni
  3.3 aggiornare github con le versioni riviste e condividere ai boss la repository
