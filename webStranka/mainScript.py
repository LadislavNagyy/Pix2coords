import cv2
import os
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from statistics import mode
import pandas as pd
from ultralytics import YOLO

# globalny slovnik
glob_prem = {}

# vytvorena inicializacia, aby sa do Flasku preniesli aj globalne premenne
def globalne_premenne():
    global glob_prem
    glob_prem["vlastny_config_tesseract"] = "-c tessedit_char_whitelist=-,.1234567890 --psm 6 --oem 3"
    glob_prem["padding"] = "30"
    glob_prem["pozadovana_vyska_textu"] = "31"


def priprava_obrazka(vstupny_obrazok):
    
    tesseractVersion = pytesseract.get_tesseract_version()
    print(f"Verzia pouzivaneho tesseractu: {tesseractVersion}")
    
    #sedy obrazok
    gray = cv2.cvtColor(vstupny_obrazok, cv2.COLOR_BGR2GRAY)
    
    #vystupny obrazok, do ktoreho sa zakresluju detekovane prvky
    finalImg = vstupny_obrazok.copy() # na tento obrazok za kresli vsetok postup detekcie 
    return gray, finalImg


#____Detekcia osi grafu____
def detekcia_ciar(gray, finalImg):
    print("\n____Detekcia osi grafu____")
    
    blur = cv2.GaussianBlur(gray,(1,1),1)

    # dolny prah - ak gradient intenzity (zmena jasnosti medzi pixelmi) je menší ako tento prah, nebude považovaný za hranu
    # horny prah - Ak gradient intenzity je väčší ako tento prah, je vždy považovaný za hranu
    cannyEdge = cv2.Canny(image=blur, threshold1=150, threshold2=250)# horny aj dolny threshold nastaveny(testovane na viacerych obrazkoch)
    #cv2.imwrite("spolocnyKod/processedImages/OSI_cannyEdge.png", cannyEdge)
    
    distanceResolution = 1 #krok velkosti - vzdialenost
    angleResolution = np.pi/180 #1 stupen 
    threshold = 200 #akumulator threshold
    
    #Hough Line Transform
    lines = cv2.HoughLines(cannyEdge, distanceResolution, angleResolution, threshold) #vrati zoznam detekovanych priamok
    k = 1000
    
    horizUsecky = [] # premenna pre vsetky horizontalne usecky v obrazku
    vertikUsecky = [] # premenna pre vsetky vertikalne usecky v obrazku
    osX = [] # pozicia osi x - najnizsie polozena horizontalna ciara na obrazku
    osY = [] # pozicia osi y - najnizsie polozena horizontalna ciara na obrazku
    hornaCiaraGrafu = []
    pravaCiaraGrafu = []
    
    if lines is None:
        print("Pozíciu osí sa nepodarilo detekovať. (skontrolujte obrázok, či sú osi dobre viditeľné)")
    
    if lines is not None:
        for curLine in lines:
            distance, angle = curLine[0]
            dhat = np.array([[np.cos(angle)], [np.sin(angle)]])
            d = distance*dhat
            lhat = np.array([[-np.sin(angle)], [np.cos(angle)]])
            p1 = d + k*lhat #body
            p2 = d - k*lhat
            p1 = p1.astype(int)
            p2 = p2.astype(int)
            
            #najde horizontalne a vertikalne usecky 
            uholUsecky = np.arctan2(p2[1][0] - p1[1][0], p2[0][0] - p1[0][0]) * 180 / np.pi
            if abs(uholUsecky)<2: #ak je usecka pod uhlom mensim ako 2 stupne, tak sa definuje ako horizontalna
                horizUsecky.append((p1[0][0],p1[1][0],p2[0][0],p2[1][0])) #hot
            if 92>abs(uholUsecky)>88: #ak je usecka pod uhlom mensim ako 2 stupne od 90, tak sa definuje ako vertikalna
                vertikUsecky.append((p1[0][0],p1[1][0],p2[0][0],p2[1][0]))
            
            # zakresli vsetky detekovane ciary do obrazka
            #cv2.line(finalImg, (p1[0][0],p1[1][0]),  (p2[0][0],p2[1][0]),  (150,255,0),1)
            #print(f"Usecka -> A:({p1[0][0]}, {p1[1][0]}), B:({p2[0][0]}, {p2[1][0]})")
        
        if horizUsecky:
            #najde najnizsie polozenu usecku - os x
            osX = max(horizUsecky, key=lambda line: max(line[1],line[3])) # najde najvacsiu y suradnicu
            x1_osX,y1_osX,x2_osX,y2_osX = osX
            croppOsX = min(y1_osX,y2_osX)
            cv2.line(finalImg, (x1_osX,croppOsX), (x2_osX,croppOsX),  (0,140,255),1) 
            print(f"OS X -> A:({x1_osX}, {y1_osX}), B:({x2_osX}, {y2_osX})")
            
            #najde hornu ciaru grafu -> vyuzije sa nasledne na orezenie udajoveho okna 
            hornaCiaraGrafu = min(horizUsecky, key=lambda line: min(line[1],line[3])) # najde najmensiu y suradnicu 
            x1_horCiara,y1_horCiara,x2_horCiara,y2_horCiara = hornaCiaraGrafu
            croppHornaCiara = min(y1_horCiara,y2_horCiara)
            cv2.line(finalImg, (x1_horCiara,croppHornaCiara), (x2_horCiara,croppHornaCiara),  (0,140,255),1)
        
        if vertikUsecky:
            #najde usecku najviac na lavo - os y
            osY = min(vertikUsecky, key=lambda line: max(line[0],line[2]))
            x1_osY,y1_osY,x2_osY,y2_osY = osY
            croppOsY = min(x1_osY,x2_osY)
            cv2.line(finalImg, (croppOsY,y1_osY), (croppOsY,y2_osY),  (0,140,255),1)
            print(f"OS Y -> A:({x1_osY}, {y1_osY}), B:({x2_osY}, {y2_osY})")
            
            #najde pravu ciaru -> vyuzije sa nasledne na orezenie udajoveho okna 
            pravaCiaraGrafu = max(vertikUsecky, key=lambda line: max(line[0],line[2]))
            x1_pravCiara,y1_pravCiara,x2_pravCiara,y2_pravCiara = pravaCiaraGrafu
            croppPravaCiara = max(x1_pravCiara,x2_pravCiara)
            cv2.line(finalImg, (croppPravaCiara,y1_pravCiara), (croppPravaCiara,y2_pravCiara),  (0,140,255),1) 
        
        print(f"Os X - y pozicia: {croppOsX}")
        print(f"Os Y - x vzdialenost: {croppOsY}")
        print(f"Najvysia ciara - y pozicia: {croppHornaCiara}")
        print(f"Ciara na pravo - x pozicia: {croppPravaCiara}")
        
        pozicia_detekovanych_ciar = []
        pozicia_detekovanych_ciar.append([croppOsY, croppPravaCiara, croppOsX, croppHornaCiara])
        
    return pozicia_detekovanych_ciar #vystupom su pozicie detekovanych ciar

#____Detekcia pozicie cisel na osiach____
def ohranicenie_cisel_osi(gray, finalImg, pozicia_detekovanych_ciar):   #funkcia vytvori bounding box pre cisla na osiach
    print("\n____Detekcia pozicie cisel na osiach____")
    
    # Prahovanie
    #blur = cv2.GaussianBlur(gray, (1, 1), 0)
    #cv2.imwrite("tempCislaOsi/index_blur.png", blur)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #cv2.imwrite("spolocnyKod/processedImages/CISLA_prahovanie.png", thresh)
    
    # Dilatácia(zvacsi biele casti) na zvýraznenie čísel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    #cv2.imwrite("spolocnyKod/processedImages/CISLA_dilatacia.png", dilation)

    # Najdenie kontur - vystupom je x y š v 
    # cv2.findContours(dilatovany obrazok, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1] # zabezpeci aby kod fungoval spravne bez onhldu na verziu OpenCV
    
    #narkresli vsetky zistene kontury
    #cv2.drawContours(finalImg, contours, -1, (0, 255, 0), 1)  # zelene kontury
    #cv2.imwrite("spolocnyKod/processedImages/CISLA_vsetkyKontury.png", img_contours)

    # Ukladanie kontúr pre každú os
    y_axis_contours = [] # vestky ciselne kontury nalavo od osi osi y
    x_axis_contours = [] # vsetky ciselne kontury pod osou x 
    
    y_axis_boundingBox = [] # rozmer ramika pre cisla na y osi
    x_axis_boundingBox = [] # rozmer ramika pre cisla na x osi
    
    #premenna pre prisku textu pri osiach
    y_axis_textHeight = []
    x_axis_textHeight = []
    
    textHeight = []
    
    bodyYosi = [] #
    
    # Filtrovanie kontur
    for c in contours:
        x, y, w, h = cv2.boundingRect(c) # zisti rozmery pre vsetky zistene kontury
        
        # Kriterium pre cisla na osiach
        if 3 < w < 70 and 7 < h < 30:  # podmienka na velkost cisla
            
            # ak je kontura pod osou x
            if y > pozicia_detekovanych_ciar[0][2]:
                x_axis_contours.append((x, y, w, h))    # uloz kontury 
                #cv2.rectangle(finalImg, (x, y), (x+w, y+h), (128,0, 255), 1) # nakresli obrys pre jednotlive kontury cisel
            
            # ulozi kontury ktore su na lavo od osi y         
            elif x < pozicia_detekovanych_ciar[0][0]:  
                y_axis_contours.append((x, y, w, h))    #uloz kontury 

    vyskaXkontur = [y[1] for y in x_axis_contours] # cykus ulozi y-ovu suradnicu zaciatku cisla(vlavo hore) pod x osou do pola
    najpocetnejsiaYpoziciaXkontur = mode(vyskaXkontur) # vypocita najpocetnejsiu y-ovu hodnotu pre kontury cisel pod x osou
    
    x_axis_textHeight = [h[3] for h in x_axis_contours] # cyklus ulozi vysku textu 
    x_axis_textHeight = mode(x_axis_textHeight) # vypocita najpocetnejsiu vysku textu
    print(f"Vyska textu x os: {x_axis_textHeight}px")
    
    
    #definuje cisla na y osi -> ak je cislo nalavo od osi y a zaroven vysie ako cisla na x osi priradi ho do bodyYosi 
    for x, y, w, h in y_axis_contours:
        if y + h < najpocetnejsiaYpoziciaXkontur:
            bodyYosi.append((x, y, w, h))
            #cv2.rectangle(finalImg, (x, y), (x+w, y+h), (128,0, 255), 1) # nakresli obrys pre jednotlive kontury cisel
    
    y_axis_textHeight = [h[3] for h in bodyYosi] # cyklus ulozi vysku textu 
    y_axis_textHeight = mode(y_axis_textHeight) # vypocita najpocetnejsiu vysku textu
    print(f"Vyska textu y os: {y_axis_textHeight}px")
    
    textHeight = int((y_axis_textHeight + x_axis_textHeight)/2)
    if textHeight == 0:
        return print("Velkost pisma nesmie byt 0.")

    print(f"Pocet kontur cisel na x osi: {len(x_axis_contours)}") # zisti kolko cisel sa detekovalo ako kontura
    print(f"Pocet kontur cisel na y osi: {len(bodyYosi)}")
    #cv2.imwrite("spolocnyKod/processedImages/CISLA_test.png", img_numbersBoundingBox)

    # velky bounding box pre os y
    if bodyYosi:
        x_min = min([x for x, y, w, h in bodyYosi])
        y_min = min([y for x, y, w, h in bodyYosi])
        x_max = max([x+ w for x, y, w, h in bodyYosi])
        y_max = max([y+ h for x, y, w, h in bodyYosi])
        #y_axis_boundingBox.append((x_min, y_min, x_max-x_min, y_max-y_min)) # x y š v
        y_axis_boundingBox.append((x_min, y_min, (x_max-x_min), (y_max-y_min))) # x y š v
        cv2.rectangle(finalImg, (x_min, y_min), (x_max, y_max), (237, 160, 45), 1)
        
    # velky bounding box pre os x
    if x_axis_contours:
        x_min = min([x for x, y, w, h in x_axis_contours])
        y_min = najpocetnejsiaYpoziciaXkontur
        x_max = max([x+w for x, y, w, h in x_axis_contours])
        y_max = max([y + h for x, y, w, h in x_axis_contours])
        #x_axis_boundingBox.append((x_min, y_min, x_max-x_min, y_max-y_min)) # x y š v
        x_axis_boundingBox.append((x_min, y_min, (x_max-x_min), (y_max-y_min))) # x y š v
        cv2.rectangle(finalImg, (x_min, y_min), (x_max, y_max), (237, 160, 45), 1)
    #cv2.imwrite("spolocnyKod/processedImages/CISLA_boundingBoxy.png", img_numbersBoundingBox)
    
    #vystup je bounding box/ obdlznik ohranicujuci osi x a y a vyska textu
    return x_axis_boundingBox, y_axis_boundingBox, textHeight

# Funkcia na konverziu textu na cislo
def text_na_cislo(text):  
    try:
        return float(text.replace(",", "").replace(" ", ""))  # Zmena textu na cislo
    except ValueError:
        return None 
    
# Funkcia na detekovanie cisla v konture pomocou tesseractu/OCR
def rozpoznanie_cisla_v_konture(obrazok, kontura):
    x, y, w, h = cv2.boundingRect(kontura)                      #ulozenie rozmerov kontury
    vyska_obrazka = obrazok.shape[0]
    sirka_obrazka = obrazok.shape[1] 
    
    roi = obrazok[y:y+h, x:x+w]                                                 # oreze vstupny obrazok iba na jednu konturu/cislo
    text = pytesseract.image_to_string(roi, config=glob_prem["vlastny_config_tesseract"])    # OCR - tesseract zisti cislo
    text = text.strip()                                                         # odstráni biele znaky z detekovaneho textu
    
    if sirka_obrazka > vyska_obrazka: # jedna sa od obrazok z osi X              
        stred_kontury = x + (w // 2) - int(glob_prem["padding"])                    # ulozenie stredu (x) kontury kontur
        
    elif vyska_obrazka > sirka_obrazka:
        stred_kontury = y + (h // 2) - int(glob_prem["padding"])                     # ulozenie stredu (y) kontury kontur
    
    detekovane_cislo = text_na_cislo(text)          # konverzia textu na cislo

    return stred_kontury, detekovane_cislo
    
#____Rozpoznanie cisel pomocou OCR____
def rozpoznanie_textu_osi(x_axis_boundingBox, y_axis_boundingBox, textHeight, gray, finalImg):
    print("\n____Rozpoznanie cisel pomocou OCR____")
    
    # pomocna premenna do ktorej sa uklada zoznam -> (kontura a x-sova surdnica kontury)
    relevantne_kontury_x = []
    relevantne_kontury_y = []
    
    stredy_kontur_cisel = []
    cisla_kontur_cisel = []
    
################## OCR  ox X-> rozpoznanie cisle a ich pozicie na x-ovej osi
    
    x_cislaX, y_cislaX, w_cislaX, h_cislaX = x_axis_boundingBox[0]              #zisi rozmery v ktorych sa cisla nachadzaju
    cislaOsX = gray[y_cislaX:y_cislaX+h_cislaX, x_cislaX:x_cislaX+w_cislaX]     #vyreze cast obrazka s cislami na x osi
    
    # zvacsenie tak by mal text vysku 32px pred vstupom do OCR
    scale_factor = round(int(glob_prem["pozadovana_vyska_textu"])/textHeight, 1) 
    vyska_zvacseneho_obrazka, sirka_zvacseneho_obrazka = cislaOsX.shape[:2]             # do premennych ulozi rozmer vyrezanej casti obrazka s cislami
    cislaOsX = cv2.resize(cislaOsX, (int(sirka_zvacseneho_obrazka * scale_factor), int(vyska_zvacseneho_obrazka * scale_factor)), interpolation=cv2.INTER_CUBIC) # zvacsi obrazok podal hodnoty v premennej a pouzije inerpolaciu
    cislaOsX = cv2.copyMakeBorder(cislaOsX, int(glob_prem["padding"]), int(glob_prem["padding"]), int(glob_prem["padding"]), int(glob_prem["padding"]), cv2.BORDER_CONSTANT, value=(255,255,255))                                        # prida padding/medzeru na pre cistatelnejsie cisla
    
    # prahovanie zvaceneho obrazka
    _, thresh = cv2.threshold(cislaOsX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imwrite("spolocnyKod/processedImages/TESSERACT_prahovanie_osX.png", thresh)
    
    # algoritmus na detekciu hran 
    hrany = cv2.Canny(thresh, 30, 200) 
    
    # dilatacia - zvacseni biele plochy, aby sa kontury lahsie dali najst, kernel je dost velky, lebo vznikal problem pri oddelenych cislach
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    dialtacia = cv2.dilate(hrany, kernel, iterations=1)
    
    # nadjenie kontur
    najdene_kontury = cv2.findContours(dialtacia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontury = najdene_kontury[0] if len(najdene_kontury) == 2 else najdene_kontury[1]
    
    # zoradenie kontur x od najmensej po najvaciu
    for kontura in kontury:
        x_kont, y_kont, w_kont, h_kont = cv2.boundingRect(kontura)
        if w_kont > 5 and (h_kont > (int(glob_prem["pozadovana_vyska_textu"])-2)):     # vyfiltruje kontury iba s urcitou velkostou
            relevantne_kontury_x.append((kontura, x_kont))           # ulozia sa do pomocneho zoznamu

    # ak nenasiel kontury ktore splnily podmienku, vypise error
    if not relevantne_kontury_x:
        print("Ziadne kontury nezodpovedaju podmienke, nenajdene.")
        return None
    
    #zoradi kontury x vzostupne
    vzostupne_zoradene_kontury_x = sorted(relevantne_kontury_x, key=lambda item: item[1]) 
    
    lava_kontura = vzostupne_zoradene_kontury_x[0][0] # 0 prvy prvok podla -> najde konturu najiac nalavo
    prava_kontura = vzostupne_zoradene_kontury_x[-1][0] # -1 posledny prvok pola -> najde konturu najiac napravo
    
    #rozpozannie cisel
    stred_lavej_x_kontury, cislo_lavej_x_kontury = rozpoznanie_cisla_v_konture(thresh, lava_kontura)       
    stred_pravej_x_kontury, cislo_pravej_x_kontury = rozpoznanie_cisla_v_konture(thresh, prava_kontura) 
    
    ''' TESTOOVACIA CAST KODU, vykresli vsetky bboxy okolo kontur a ich zistene cisla 
    for kontura in kontury:
        x_kont, y_kont, w_kont, h_kont = cv2.boundingRect(kontura)
        #cv2.rectangle(img, (x_kont, y_kont), (x_kont + w_kont, y_kont + h_kont), (0, 255, 0), 2)

        if w_kont > 5 and (h_kont > (pozadovana_vyska_textu-2)): # vyfiltruje kontury iba s urcitou velkostou    
            roi = thresh[y_kont:y_kont+h_kont, x_kont:x_kont+w_kont]           # oreze na jednotlive cislo
            text = pytesseract.image_to_string(roi, config=vlastny_config_tesseract) # rozpozan jednotlive cislo v kontrure pomocou OCR
            text = text.strip()                                                # odstrani vsetky biele znaky
            detekovane_cislo_x = text_na_cislo(text)                           # prevedie text na cislo
                
            print(f"Detegované číslo x: {detekovane_cislo_x} na pozícii ({x_kont}, {y_kont})")
            cv2.rectangle(thresh, (x_kont, y_kont), (x_kont + w_kont, y_kont + h_kont), (0, 255, 0), 2)
            cv2.putText(thresh, str(detekovane_cislo_x), (x_kont, y_kont+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detekovane_cislo_x = text_na_cislo(text)                           # prevedie text na cislo
    cv2.imwrite("spolocnyKod/processedImages/TESSERACT_test_ciselX.png", thresh)
    '''  
             
    
################## OCR os Y-> rozpoznanie cisle a ich pozicie na y-ovej osi
    x_cislaY, y_cislaY, w_cislaY, h_cislaY = y_axis_boundingBox[0]              #zisi rozmery v ktorych sa cisla nachadzaju
    cislaOsY = gray[y_cislaY:y_cislaY+h_cislaY, x_cislaY:x_cislaY+w_cislaY]     #vyreze cast obrazka s cislami na y osi
    
    # zvacsenie tak by mal text vysku 32px pred vstupom do OCR
    #scale_factor = round(pozadovana_vyska_textu/textHeight, 1) 
    vyska_zvacseneho_obrazka, sirka_zvacseneho_obrazka = cislaOsY.shape[:2]             # do premennych ulozi rozmer vyrezanej casti obrazka s cislami
    cislaOsY = cv2.resize(cislaOsY, (int(sirka_zvacseneho_obrazka * scale_factor), int(vyska_zvacseneho_obrazka * scale_factor)), interpolation=cv2.INTER_CUBIC) # zvacsi obrazok podal hodnoty v premennej a pouzije inerpolaciu
    cislaOsY = cv2.copyMakeBorder(cislaOsY, int(glob_prem["padding"]), int(glob_prem["padding"]), int(glob_prem["padding"]), int(glob_prem["padding"]), cv2.BORDER_CONSTANT, value=(255,255,255))                                        # prida padding/medzeru na pre cistatelnejsie cisla
    
    # prahovanie zvaceneho obrazka
    _, thresh_Osy = cv2.threshold(cislaOsY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #cv2.imwrite("spolocnyKod/processedImages/TESSERACT_prahovanie_osY.png", thresh_Osy)
    
    # algoritmus na detekciu hran 
    hrany = cv2.Canny(thresh_Osy, 30, 200) 
    
    # dilatacia - zvacseni biele plochy, aby sa kontury lahsie dali najst, kernel je dost velky, lebo vznikal problem pri oddelenych cislach
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    dialtacia = cv2.dilate(hrany, kernel, iterations=1)
    
    # nadjenie kontur
    najdene_kontury = cv2.findContours(dialtacia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kontury = najdene_kontury[0] if len(najdene_kontury) == 2 else najdene_kontury[1]
    
    # zoradenie kontur x od najmensej po najvaciu
    for kontura in kontury:
        x_kont, y_kont, w_kont, h_kont = cv2.boundingRect(kontura)
        if w_kont > 5 and (h_kont > (int(glob_prem["pozadovana_vyska_textu"])-2)):     # vyfiltruje kontury iba s urcitou velkostou
            relevantne_kontury_y.append((kontura, y_kont))           # ulozia sa do pomocneho zoznamu

    # ak nenasiel kontury ktore splnily podmienku, vypise error
    if not relevantne_kontury_y:
        print("Ziadne kontury nezodpovedaju podmienke, nenajdene.")
        return None
    
    #zoradi kontury y vzostupne, kontury su zoraden z hora dole
    vzostupne_zoradene_kontury_y = sorted(relevantne_kontury_y, key=lambda item: item[1]) 
    
    horna_kontura = vzostupne_zoradene_kontury_y[0][0] # 0 prvy prvok podla -> najde konturu ktora je najviac hore
    dola_kontura = vzostupne_zoradene_kontury_y[-1][0] # -1 posledny prvok pola -> najde konturu ktora je najviac dole

    #rozpozannie cisel
    stred_hornej_y_kontury, cislo_hornej_y_kontury = rozpoznanie_cisla_v_konture(thresh_Osy, horna_kontura)       
    stred_dolnej_y_kontury, cislo_dolnej_y_kontury = rozpoznanie_cisla_v_konture(thresh_Osy, dola_kontura) 
    
    """
    for kontura in dola_kontura:
        x_kont, y_kont, w_kont, h_kont = cv2.boundingRect(kontura)
        cv2.rectangle(test_osY, (x_kont, y_kont), (x_kont + w_kont, y_kont + h_kont), (0, 255, 0), 2)
        cv2.imwrite("spolocnyKod/processedImages/TESSERACT_test_ciselY.png", test_osY)
        
        if w_kont > 5 and (h_kont > (pozadovana_vyska_textu-2)): # vyfiltruje kontury iba s urcitou velkostou    
            roi = thresh_Osy[y_kont:y_kont+h_kont, x_kont:x_kont+w_kont]           # oreze na jednotlive cislo
            text = pytesseract.image_to_string(roi, config=vlastny_config_tesseract) # rozpozan jednotlive cislo v kontrure pomocou OCR
            text = text.strip()                                                # odstrani vsetky biele znaky
            detekovane_cislo_y = text_na_cislo(text)                           # prevedie text na cislo
                
            print(f"Detegované číslo y: {detekovane_cislo_y} na pozícii ({x_kont}, {y_kont})")
            #cv2.rectangle(test_osY, (x_kont, y_kont), (x_kont + w_kont, y_kont + h_kont), (0, 255, 0), 2)
            #cv2.putText(test_osY, str(detekovane_cislo_y), (x_kont, y_kont+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detekovane_cislo_y = text_na_cislo(text)                           # prevedie text na cislo
    """
    
    
    print(f"Rozsah osi x ({cislo_lavej_x_kontury}, {cislo_pravej_x_kontury})")
    print(f"Rozsah osi y ({cislo_dolnej_y_kontury}, {cislo_hornej_y_kontury})")
    
    if scale_factor+x_cislaX == 0:
        print("Nie je mozne delit 0.")
        return None
    
    stred_lavej_x_kontury = int(stred_lavej_x_kontury/scale_factor + x_cislaX)
    stred_pravej_x_kontury = int(stred_pravej_x_kontury/scale_factor + x_cislaX)
    stred_hornej_y_kontury = int(stred_hornej_y_kontury/scale_factor + y_cislaY)
    stred_dolnej_y_kontury = int(stred_dolnej_y_kontury/scale_factor + y_cislaY)
    
    cv2.line(finalImg, (stred_lavej_x_kontury, 0),(stred_lavej_x_kontury, finalImg.shape[0]),(0, 255, 0),1)
    cv2.line(finalImg, (stred_pravej_x_kontury, 0),(stred_pravej_x_kontury, finalImg.shape[0]),(0, 255, 0),1)
    cv2.line(finalImg, (0, stred_hornej_y_kontury),(finalImg.shape[1], stred_hornej_y_kontury),(0, 255, 0),1)
    cv2.line(finalImg, (0, stred_dolnej_y_kontury),(finalImg.shape[1], stred_dolnej_y_kontury),(0, 255, 0),1)
    
    # ulozenie stredov suradnic vsetkych cisel do jedneho zoznamu
    stredy_kontur_cisel.append([stred_lavej_x_kontury, stred_pravej_x_kontury, stred_dolnej_y_kontury, stred_hornej_y_kontury])  
    
    # ulozenie hodnot vsetkych cisel do jedneho zoznamu
    cisla_kontur_cisel.append([cislo_lavej_x_kontury, cislo_pravej_x_kontury, cislo_dolnej_y_kontury, cislo_hornej_y_kontury])   
    
    return  stredy_kontur_cisel, cisla_kontur_cisel
 
#____Detekcia bodov grafu na pomocou vlastneho modelu____
def predikcia_bodov(pozicia_detekovanych_ciar, stredy_kontur_cisel, cisla_kontur_cisel, finalImg, vstupny_obrazok):
    print("\n____Predikcia bodov - yolo model____")
    # prazdne polia pre ulozenie detekovanych hodnot
    detekovane_body_VrozsahuOsi = []
    pocet_bodov = 0
    
    # rozsah osi x v pixeloch
    #stred_lavej_x_kontury = stredy_kontur_cisel[0][0] - croppOsY
    stred_lavej_x_kontury = stredy_kontur_cisel[0][0] - pozicia_detekovanych_ciar[0][0]
    #stred_pravej_x_kontury = stredy_kontur_cisel[0][1] - croppOsY
    stred_pravej_x_kontury = stredy_kontur_cisel[0][1] - pozicia_detekovanych_ciar[0][0]
    # rozsah osi y v pixeloch
    #stred_dolnej_y_kontury = stredy_kontur_cisel[0][2] - croppHornaCiara
    stred_dolnej_y_kontury = stredy_kontur_cisel[0][2] - pozicia_detekovanych_ciar[0][3]
    #stred_hornej_y_kontury = stredy_kontur_cisel[0][3] - croppHornaCiara
    stred_hornej_y_kontury = stredy_kontur_cisel[0][3] - pozicia_detekovanych_ciar[0][3]
    
    # rozsah hodnot/cisel osi x podla mierky
    cislo_lavej_x_kontury = cisla_kontur_cisel[0][0]
    cislo_pravej_x_kontury = cisla_kontur_cisel[0][1]
    # rozsah hodnot/cisel osi y podla mierky
    cislo_dolnej_y_kontury = cisla_kontur_cisel[0][2]
    cislo_hornej_y_kontury = cisla_kontur_cisel[0][3]
    
    # vypocet mierky-> aku hodnotu ma jedne pixel
    if(stred_pravej_x_kontury - stred_lavej_x_kontury) !=0 :
        mierka_osX = (cislo_pravej_x_kontury - cislo_lavej_x_kontury) / (stred_pravej_x_kontury - stred_lavej_x_kontury)
    else:
        print("Chyba: Nesmie sa delit cislom 0!")    
    if(stred_hornej_y_kontury - stred_dolnej_y_kontury) !=0 :
        mierka_osY = (cislo_hornej_y_kontury - cislo_dolnej_y_kontury) / (stred_hornej_y_kontury - stred_dolnej_y_kontury)
    else:
        print("Chyba: Nesmie sa delit cislom 0!")    
    
    #vstupny obrazok sa oreze na udajovu cast, teda iba na cast kde sa nachadzaju body grafu
    #udajova_cast_obrazka = vstupny_obrazok[croppHornaCiara:croppOsX, croppOsY:croppPravaCiara]
    udajova_cast_obrazka = vstupny_obrazok[pozicia_detekovanych_ciar[0][3]:pozicia_detekovanych_ciar[0][2], pozicia_detekovanych_ciar[0][0]:pozicia_detekovanych_ciar[0][1]]
    
    # orezany obrazok na udajovu cast, ktora je vstupom do predtrenovaneho modelu
    #cv2.imwrite("spolocnyKod/processedImages/udajova_cast.png", udajova_cast_obrazka) # ulozenie orezaneho obrzka
    
    
    #nacitane a spustenie predickie modela 
    model = YOLO("./spolocnyKod/model_v2_rtx4070ti.pt") #  JE POTREBNE ODKOMENTOVAT AK CHETE SPUSTIT NA LOCALHOSTE 
    #model = YOLO("./model_v2_rtx4070ti.pt") # JE POTREBNE ODKOMENTOVAT, AK CHETE SPUSTIT V DOCKER KONTAJNERI
    
    vysledky_predikcie = model.predict(source=udajova_cast_obrazka, show=False, line_width = 1, device="cpu") #conf=0.5

    for vysledok in vysledky_predikcie:
        for bod in vysledok.boxes.xyxy:
            x_vlavo, y_vlavo, x_vpravo, y_vpravo =  map(int,bod.tolist())       #prevedie tensor na obycajny python list, nasleden prejde prvky v zozname a kazdy prevedie na cele cislo
            cv2.rectangle(finalImg, (x_vlavo+pozicia_detekovanych_ciar[0][0], y_vlavo+pozicia_detekovanych_ciar[0][3]), (x_vpravo+pozicia_detekovanych_ciar[0][0], y_vpravo+pozicia_detekovanych_ciar[0][3]), (128,0, 255), 1) #nakreslenie bboxov pre body
        for bod in vysledok.boxes.xywh:
            pocet_bodov += 1
            x_stred_pixel, y_stred_pixel, _, _ =  map(int,bod.tolist())                     #prevedie tensor na obycajny python list, nasleden prejde prvky v zozname a kazdy prevedie na cele cislo
            
            #prevod suradnice x z pixelov na hodnoty podla mierky grafu
            rozdiel_pixelov = x_stred_pixel - stred_lavej_x_kontury
            suradnica_osX = cislo_lavej_x_kontury + (rozdiel_pixelov * mierka_osX)
            
            #prevod suradnice y z pixelov na hodnoty podla mierky grafu
            rozdiel_pixelov = y_stred_pixel - stred_dolnej_y_kontury
            suradnica_osY = cislo_dolnej_y_kontury + (rozdiel_pixelov * mierka_osY)
            
            detekovane_body_VrozsahuOsi.append((suradnica_osX,suradnica_osY))  #ulozi body(x,y) do pola
    
    print(f"\nPočet detekovaných bodov: {pocet_bodov}\n")
    
    # zoradenie bodov podla x surdanice od najmensej po najvacsiu    
    detekovane_body_VrozsahuOsi.sort(key=lambda bod: bod[0]) 
    
    # rozdelenie suradnic bodov do jednotlivych zoznamov
    suradniceBodovX_VrozsahuOsi = [x for x, y in detekovane_body_VrozsahuOsi]
    suradniceBodovY_VrozsahuOsi = [y for x, y in detekovane_body_VrozsahuOsi]
    
    print(f"\nSuradnice podla mierky osi x: {[round(x, 2) for x in suradniceBodovX_VrozsahuOsi]}")
    print(f"\nSuradnice podla mierky osi y: {[round(x, 2) for x in suradniceBodovY_VrozsahuOsi]}\n")
    
    # ulozi sa vysledny obrazok, na ktorom su detekovane prvky
    #cv2.imwrite("spolocnyKod/processedImages/final.png", finalImg)

    #ukladanie finalneho obrazka do static/processed_img/final.png
    base_dir = os.path.dirname(os.path.abspath(__file__))  # cesta k priecinku, kde je ulozeny mainScript.py
    output_path = os.path.join(base_dir, "static", "processed_img", "final.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # vytvor priecinky ak neexistuju
    cv2.imwrite(output_path, finalImg)

    
    return suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi

# ulozenie suradnic do 'suradnice_bodov.csv'
def ulozenie_suradnic_doCSV(suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi):
    #ukladanie csv do suboru processed_img
    base_dir = os.path.dirname(os.path.abspath(__file__))  # cesta k priecinku, kde je ulozeny mainScript.py
    output_path = os.path.join(base_dir, "static", "processed_img", "suradnice_bodov.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # vytvor priecinky ak neexistuju
    
    ramec = pd.DataFrame({"x": suradniceBodovX_VrozsahuOsi, "y": suradniceBodovY_VrozsahuOsi}) # vytvori ramec/frame
    ramec.to_csv(output_path, index=False, sep=";", decimal=",") # index=False - neulozi index do csv
    
    
     
if __name__ == '__main__': 
    
    #nacitanie obrazka a zavedenie globalnych premennych
    subor_obrazka = "./vlastneObrazkyGrafov/grafyBezNazvuOsi/graf7croppedWeb.png" 
    vstupny_obrazok = cv2.imread(subor_obrazka)
    globalne_premenne()
    
    # spracovanie obrazka
    gray, finalImg = priprava_obrazka(vstupny_obrazok)
    #croppOsY, croppPravaCiara, croppOsX, croppHornaCiara = detekcia_ciar(gray, finalImg)
    pozicia_detekovanych_ciar = detekcia_ciar(gray, finalImg)
    x_axis_boundingBox, y_axis_boundingBox, textHeight = ohranicenie_cisel_osi(gray, finalImg, pozicia_detekovanych_ciar)
    stredy_kontur_cisel, cisla_kontur_cisel = rozpoznanie_textu_osi(x_axis_boundingBox, y_axis_boundingBox, textHeight, gray, finalImg)
    #suradniceOsiX, suradniceOsiY = dotDetection_houghCircleTransform(croppOsY, croppPravaCiara, croppOsX, croppHornaCiara, x_range, y_range)
    suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi = predikcia_bodov(pozicia_detekovanych_ciar, stredy_kontur_cisel, cisla_kontur_cisel, finalImg, vstupny_obrazok) #pouziva predtrenovany YOLOv11 model na detekciu bodov
    ulozenie_suradnic_doCSV(suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi)
