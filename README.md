# Off-Road Image Segmentation

Questo progetto implementa una pipeline completa per la **segmentazione semantica** di immagini catturate in ambienti rurali e off-road. L'obiettivo è identificare diverse tipologie di terreno, vegetazione e ostacoli per supportare la navigazione autonoma in contesti non strutturati.

## 🚀 Tecnologie Utilizzate

Il progetto è sviluppato in **Python** utilizzando le seguenti librerie principali:
- **PyTorch**: Framework principale per la definizione e l'addestramento del modello di Deep Learning.
- **NumPy & PIL (Pillow)**: Per la manipolazione efficiente delle immagini e delle matrici.
- **Matplotlib**: Per la visualizzazione dei risultati e dei grafici di addestramento.
- **Scikit-learn**: Utilizzata per il calcolo delle metriche di classificazione (Precision, Recall, F1-score).

## 🏗️ Architettura del Modello

Il cuore del progetto è la `SegmentationModel`, un'architettura personalizzata ispirata alla **U-Net**. Le sue caratteristiche principali includono:
- **Encoder (Downsampling)**: Quattro blocchi di "Double Convolution" seguiti da Max Pooling per estrarre feature gerarchiche.
- **Decoder (Upsampling)**: Blocchi di upsampling bilineare con **Skip Connections** (concatenazione) per preservare i dettagli spaziali ad alta risoluzione.
- **Output Layer**: Una convoluzione 1x1 con attivazione **Softmax** che produce una mappa di probabilità per ciascuna delle 9 classi supportate.

## 📊 Dataset e Classi

Il dataset è gestito dalla classe `RuralDataset`, che elabora coppie di immagini RGB (`rgb.jpg`) e maschere di segmentazione (`labels.png`).

### Classi di Segmentazione
Grazie al `LabelMapper`, il modello è in grado di distinguere 9 classi specifiche:
1. **Background**: Aree non classificate.
2. **Sky**: Cielo.
3. **Rough Trail**: Sentieri accidentati.
4. **Smooth Trail**: Sentieri battuti/lisci.
5. **Traversable Grass**: Erba percorribile.
6. **High Vegetation**: Vegetazione alta (alberi, cespugli densi).
7. **Non Traversable Low Vegetation**: Vegetazione bassa non percorribile.
8. **Puddle**: Pozzanghere.
9. **Obstacle**: Ostacoli generici.

## ⚙️ Pipeline di Addestramento e Validazione

La classe `Trainer` gestisce l'intero ciclo di vita dell'addestramento:
- **Early Stopping**: Monitora la **mIoU (mean Intersection over Union)** sulla validazione per evitare l'overfitting.
- **Class Weighting**: Bilancia l'importanza delle classi meno frequenti nel dataset per migliorare la precisione su ostacoli e pozzanghere.
- **Metriche**: Durante ogni epoca vengono calcolate la Loss e la mIoU per monitorare i progressi.
- **Salvataggio Modello**: Viene salvato automaticamente solo il modello che ottiene le migliori performance di validazione (`saved_model.pth`).

## 🧪 Valutazione e Predizione

- **Evaluator**: Permette di calcolare metriche avanzate (Accuracy, Precision, Recall, F1) e di confrontare visivamente le predizioni del modello con le label reali. Include funzionalità per estrarre campioni casuali o predire da specifiche cartelle.
- **Predictor**: Una classe semplificata per eseguire inferenza su batch di immagini, automatizzando il preprocessing e il postprocessing.

## 🛠️ Come Iniziare

1. **Requisiti**: Assicurati di avere installato PyTorch, torchvision, numpy, pillow e matplotlib.
2. **Notebook Principale**: Esplora `main.ipynb` per vedere un esempio completo di addestramento e valutazione. 
3. **Predizione**: Puoi utilizzare la classe `Predictor` caricando il file `saved_model.pth` per testare il modello su nuove immagini.

---
*Progetto sviluppato per la segmentazione semantica in scenari off-road.*
