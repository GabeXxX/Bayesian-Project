# Bayesian-Project

Presentation 24/11: https://www.overleaf.com/project/65521ef6a20cdec448e00397

### Every file is named as "TaskNumber_Version_Task", to ensure continuity.

### We invite you to rely majorly on the last available version of every file 
(002_3_Unconstrained_Model, 003_2_Constrained_Model)

## To do list
### Alignment:
1. capire come si fa in maniera basic
2. farlo sulle nostre curve in maniera basic
3. capire come si fa per via bayesiana e se c è tempo farlo bayesianamente

Starting Point: https://cran.r-project.org/web/packages/fdasrvf/fdasrvf.pdf

Per ora: abbiamo provato ad utilizzare la libreria di python "fdasrsf" analogo di ‘fdasrvf’ per R ma la funzione che fa alignment in framework bayesiano allinea due curve alla volta
Proveremo ad utilizzare ‘fdasrvf’ su python o in alternativa direttamente su R


### Testing:  
1. aprire un nuovo script jupyter notebook, importare la classe che abbiamo creato dal modello definitivo
2. testarlo su vari kernel della prior, aumentando il numero di covariate

For now: tested unconstrained model on simple white noise kernel, builded a function that allow to scale testing easily on different kernel with some error statistics

### (DONE) code checking
