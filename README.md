# Ist das echt? Eine Evaluierung der Qualität von simulierten EEG-Signalen

Der von mir geschriebene Code befindet sich in bachelorarbeit/eeggan/Bachelorarbeit.
Um das Programm auszuführen muss nur die Datei main.py gestartet werden. 
Die erforderlichen Bibliotheken sind in der 'requirement.txt'-Datei gelistet.

Die Schritte die in main.py durchlaufen werden, e.g. das Training von GANs oder Klassifikatoren, 
wird durch die Parameter in der Datei 'config.ini' geregelt. Diese befindet sich im obersten Verzeichnis. 
Neben dem Kontrollfluss befinden sich dort auch Parameter die das Training und andere Schritte beeinflussen (e.g. Anzahl Epochen beim Training).
Zum Testen des Programms gibt es einen Testmodus, der in der Konfigurationsdatei aktiviert werden kann. Dieser beschränkt die Anzahl
der Daten, die Anzahl der Epochen etc., um grundlegende Funktionalitäten zu überprüfen. 

Insgesamt gibt es 8 Schritte in dieser Pipeline (Abschnitt [PIPELINE_STEPS] in config.ini):

1. Erzeugung der Datensätze
2. Training der GANs
3. Training der Klassifikatoren auf den nicht-augmentieren Datensätzen
4. Training der Klassifikatoren auf den augmentieren Datensätzen
5. Erzeugung der Fragen/Bilder für die Umfrage
6. Die Evaluierung der GANs
7. Die Evaluierung der Klassifikatoren
8. Die Evaluierung der Umfrage

Um einen Schritt zu aktivieren muss der jeweilige Wert auf True gesetzt werden. 

Die Daten für das Training der Modelle sind hier zu finden: 
https://seafile.zfn.uni-bremen.de/d/a5a060bf6240466f983e/

Das Programm erwartet, dass sich der Ordner mit den Daten ('Data') eine Ebene über dem Hauptverzeichnis des Projekts befindet. 

Die Ergebnisse der Evaluierung werden in einem Ordner 'results' gespeichert, der automatisch erzeugt wird. 
Dieser befindet sich dann ebenfalls eine Ebene über dem Hauptverzeichnis des Projekts.

Zum Reproduzieren der Ergebnisse der Bachelorarbeit ist der vorgefertigte  Ordner 'results' herunterzuladen, welcher den neu erzeugten ersetzt. Dieser beinhaltet die vortrainierten Modelle und auch die vorberechneten Ergebnisse der Evaluierung in Form von gespeicherten Dictionaries ('result_dict.npy' bei den Klassifikatoren und 'scores_dict.npy' bei den GANs). 
Wenn die Evaluierung der Klassifikatoren und/oder GANs mit diesen Daten geschehen soll, dann sollte in der config.ini sichergestellt werden,
dass im Abschnitt "Evaluation" die Variablen 'c_calc_metrics' und 'gan_calc_metrics' auf False gesetzt sind.
Wenn diese auf True gesetzt sind werden die Ergebnisse komplett neu berechnet und die dictionaries im Results-Ordner überschrieben. 
Weiter ist Vorsicht geboten, wenn das Training der GANs, Klassifikatoren, etc. erneut ausgeführt wird, da diese alte Ergebnisse überschreiben.






