# Pix2coords
Tento repozistár slúži, ako uložistko projektu aplikácie na detekciu karteziánskych súradníc bodového grafu. Jendá sa o bakalárskú prácu.
- Priečinok dataset obsahuje skript tvorbaDatasetu.py, slúši na vytvorenie datasetu, ktorý sme využili v práci na trénovanie modelu neurónovéj siete YOLO11l. Nachádza sa tu aj ukážka samotného datasetu. Nedokázali sme ho zverejniť, kvoli svojej veľkosti, preto sa v priečinku nachádza iba menší počet obrázkov a labelov na ukážku.
- Ďalším priečinkom je trenovanie_modelu, obsahuje súbory data.yaml a train.py, využili sme ich na trénovanie modelu.
- Webová stránka je umiestnená v priečinku webovaStranka. Nachádza sa tu väčšina potrebných súborov pre spustenie aplikácie na lokálnom serveri, až na model best_YOLOl.pt, ktorý je potrebné dodatočne stiahnuť a vložiť do priečinka webovaStranka. Umiestnili sme ho na Google Drive: https://drive.google.com/drive/folders/1d46HVzzB1HZeSQwxTmrEE3YDdNIYZjlK?usp=sharing. Na drive je možné nahliadnuť aj do výsledkov trénovanie modelu.

Adresa webovéj aplikácie dostupnéj na internete: https://pix2coordsfinal-817614432845.europe-central2.run.app/
