# machine-learning

Piccola libreria per machine learning in java.
E' disponibile qualche classe pronta all'uso in ml/endpoints.
In ml/core si trovano le classi che eseguono effettivamente il lavoro.
La fase di training è multi thread, il programma userà tutti i core disponibili per velocizzare il lavoro e il carico della cpu sarà vicino al 100% per qualche minuto. Consiglio eventualmente di chiudere i programmi non necessari al momento del training.

Disposizione e funzioni delle classi:

Package core:
NeuralNetwork -> classe principale, qui viene eseguito l'algoritmo di training, è presente il metodo per ottenere la previsione della rete neurale dopo il training.
NNActivation -> enumerazione delle funzioni di attivazione
NNError -> enumerazione delle funzioni di errore
NNParameter -> classe che contiene tutti i parametri per una rete neurale
SupervisedNetwork -> classe astratta che definisce i metodi e gli attributi base per una rete neurale supervisionata, NeuralNetwork è una sottoclasse

Package endpoints:
Classifier -> classificatore, è una rete neurale a 2 strati il cui primo è lineare e il secondo (essendo un classificatore) con attivazione sigmoide, può essere usato con 2 o più etichette di output.
LinearRegressor -> regressione lineare n:1, non so a cosa possa servire ma se serve c'è

Package utility:
DataEditor -> i metodi statici di questa classe servono per modellare il proprio dataset, il metodo split divide il dataset in train set, validation set e test set secondo le percentuali passate come parametri. Ci sono poi i metodi per normalizzare e standardizzare i dati.
DataLoader -> comprende i metodi statici per caricare i dati da file (al momento solo csv)
