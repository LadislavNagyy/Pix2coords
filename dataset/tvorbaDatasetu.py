import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import os
print("Current working directory:", os.getcwd())
# nastaveny fixny seed na pre generovanie cisel

# nastavenie seedu pre opatovne nahodne generovanie
random.seed(17) 
np.random.seed(17)

# Dataset parameters
NUM_IMAGES = 27  # pocet vygenerovanycg grafov (idealne musi byt delitelne poctom STYL_BODOV teda 9)
IMG_WIDTH = 1000 #1000 # Image size (pixels), min 400 max 900
IMG_HEIGHT = 1000 #1000 # Height of image, min 400 max 900
DPI = 100

RANGE = 9  # najmesnia x suradnica bodu
MIN_NUM_POINTS = 30  # najmensi pocet bodov na grafe
MAX_NUM_POINTS = 80 # najvasi pocet bodov na grafe
VELKOSTI_BODOV = [ 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 230, 250, 270, 300, 330, 350, 370, 400, 430, 450, 470, 500, 510, 530, 550, 570, 580, 600] # velkost bodu vo formate points^2 - kedze funkcia scatter pouziva plochu od 40 200
# stary styl bodov ['.', 'v', '^', '<', '>', 's', 'P', '*', '+', 'x', 'D']
STYL_BODOV = ['.', 'v', '^', '<', '>', '*', '+', 'x', 's'] 

styl_ciar = ['-', '--', '-.', ':']
styl_ciar_pozadia = ['-', '--', '-.', ':','none']
hrubka_ciary = [1,2,3]

# pozicia a rozmer bodov
centersPix = []

# cesty pre ulozenie obrazkov 
OUTPUT_DIR_data = "./dataset/data_na_trenovanie/data"
OUTPUT_DIR_figures = "./dataset/data_na_trenovanie/figures"
OUTPUT_DIR_test = "./dataset/data_na_trenovanie/data/test"
OUTPUT_DIR_train = "./dataset/data_na_trenovanie/data/train"
OUTPUT_DIR_valid = "./dataset/data_na_trenovanie/data/valid"

# Vytvori hlavne adresare pre data, obrazky a rozne datasety (train, valid, test)
os.makedirs(OUTPUT_DIR_data, exist_ok=True)
os.makedirs(OUTPUT_DIR_figures, exist_ok=True)
os.makedirs(OUTPUT_DIR_train, exist_ok=True)
os.makedirs(OUTPUT_DIR_valid, exist_ok=True)
os.makedirs(OUTPUT_DIR_test, exist_ok=True)

# Vytvori podadresare 'images' a 'labels' pre trenovaci dataset
os.makedirs(os.path.join(OUTPUT_DIR_train, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR_train, "labels"), exist_ok=True)

# Vytvori podadresare 'images' a 'labels' pre validacny dataset
os.makedirs(os.path.join(OUTPUT_DIR_valid, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR_valid, "labels"), exist_ok=True)

# Vytvori podadresare 'images' a 'labels' pre testovaci dataset
os.makedirs(os.path.join(OUTPUT_DIR_test, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR_test, "labels"), exist_ok=True)


# funkcia na generovanie nahondej farby bodu 
def nahodna_farba():
    while True:
        rnd_farba = np.random.rand(3,)  # pouzije sa normalizovane rgb, teda generuje 3 hodnoty od 0-1, pricom 111 je biela farba
        if np.sum(rnd_farba) < 2.3:     # osetrenie proti generovanie bielej farby - porovnava sa "prah", teda ak je sucet mensi, ako 2.6, tak sa farba vyberie
            return rnd_farba


# funkcia na generovanie nahodneho stylu ciary
def nahodny_styl_ciary():
    s_ciary = random.choice(styl_ciar)
    return s_ciary

# funkcia na generovanie nahodneho stylu ciary pozadia
def nahodny_styl_ciary_pozadia():
    s_ciary = random.choice(styl_ciar)
    return s_ciary


# funkcia na generovanie nahodnej hrubky ciary
def nahodna_hrubka_ciary():
    h_ciary = random.choice(hrubka_ciary)
    return h_ciary


# funkcia na generovanie nahodnej velkosti bodu
def nahodna_velkost_bodu():
    v_bodu = random.choice(VELKOSTI_BODOV)
    return v_bodu


# generovanie bodoveho grafu, bez ciar
def generovanie_nahodnych_nespojitych_bodov():
    # generovanie bodov
    num_points = random.randint(MIN_NUM_POINTS,MAX_NUM_POINTS)          # vygeneruje nahodny pocet bodov 
    x_bod = np.random.randint(RANGE, IMG_WIDTH, size=num_points).tolist()   # vygeneruje zoznam x-ovych suradnic bodov v rozsahu od RANGE po IMG_WIDTH
    y_bod = np.random.randint(RANGE, IMG_HEIGHT, size=num_points).tolist()  # vygeneruje zoznam y-ovych suradnic bodov v rozsahu od RANGE po IMG_HEIGHT
    return x_bod, y_bod


# nahodne generovane sinusiody, na priebeh su vynesene nahodne body
def sinusoidy():
    pocet_sinusiod = 10
    #medzera = 0.5 # padding zabezpeci, ze sa body nebudu generovat na ciarach osi
    pocet_bodov = random.randint(10, 50)    # nahodny podcet bodov na sinusiodach 10 az 40
    min_vzdialenost_bodov = 0.5             # min vzdialenost medzi dvoma bodmi, ktore sa vykresluju pomocou scatter

    # generovanie bodov ciar na vykreslenie suvislej ciary
    x_ciara = np.linspace(0,pocet_bodov, 200)   # vytvori 500 hodnot x podla rozsahu pre nasledne kreslenie ciary
    #x_ciara_medzera = x_ciara[(x_ciara > medzera) & (x_ciara < (x_ciara.max() - medzera))]  # uprava pola, aby sa zabezpecilo, ze sa body nebudu generovat na okrajoch obrazka
    y_ciara = np.zeros_like(x_ciara)            # vytvori pole rovnakeho rozmeru ako x, ale vyplnene s hodnotami 0 pre nasledne kreslenie ciary
    
    #generovanie bodov na ciarach
    x_nahodne_vzostupne = np.sort(np.random.choice(x_ciara, size = pocet_bodov *3, replace=False))      # nahodne vyberie pocet_bodov*3 a zoradi ich vzostupne 
    x_body = x_nahodne_vzostupne[np.insert(np.diff(x_nahodne_vzostupne) >= min_vzdialenost_bodov, 0, True)]     # vysledkom su cisla, medzi ktorymi je aspon minimalny rozdiel min_vzdialenost_bodov
    y_body = np.zeros_like(x_body)              # vytvori pole naplene s 0, s rozmermy ako x_body

    for _ in range(pocet_sinusiod):
        amplituda = np.random.uniform(0.3, 4)                       # nahodne generovana amplituda
        frekvencia = np.random.uniform(0.3, 6)                      # nahodne generovana frekvencia
        faza = np.random.uniform(0, 2 * np.pi)                      # nahodne generovany fazovy posun
        y_ciara += amplituda * np.sin(frekvencia * x_ciara + faza)  # do pola y_ciara sa prida vypocitana hodnota 
        y_body += amplituda * np.sin(frekvencia * x_body + faza)    # do pola y_bod sa prida vypocitana hodnota
    return x_ciara, y_ciara, x_body, y_body


# nahodne generovany brownow pohyb, na priebeh su vynesene nahodne body
def borwnov_pohyb():
    krok = random.randint(5, 40)            # generuje sa nahodny pocet krokov od 5 do 40
    x_krok = np.random.normal(0, 1.0, krok) # generuje nahodne kroky v smere x
    y_krok = np.random.normal(0, 1.0, krok) # generuje nahodne kroky v smere y
    
    x = np.cumsum(x_krok)                   # kumulatinvy sucet x
    y = np.cumsum(y_krok)                   # kumulatinvy sucet y   
    return x, y


#nahodne generovana kombinacia funckii,  na priebeh su vynesene nahodne body
def kombinacia_funkcii():  
    pocet_bodov = random.randint(10, 40)
    min_vzdialesnost = 0.5

    x_ciara = np.linspace(0,pocet_bodov, 500)                                                                 # vytvori 500 hodnot x podla rozsahu pre nasledne kreslenie ciary
    x_nahodne_vzostupne = np.sort(np.random.choice(x_ciara, size=pocet_bodov * 3, replace=False))             # vysledkom su rozdiely medzi prvkami x_ciara, ktore 
    x_body = x_nahodne_vzostupne[np.insert(np.diff(x_nahodne_vzostupne) >= min_vzdialesnost, 0, True)]        # Vyberieme len body s medzerou

    # generovanie nahodnych koef. pre trigonometricku cast 
    a = np.random.uniform(-pocet_bodov, pocet_bodov, 5)  # nahodne generovana amplituda
    b = np.random.uniform(0.5, 5, 5)                     # nahodne generovana frekvencia
    c = np.random.uniform(0, np.pi, 5)                   # nahodne generovany fazovy posun

    # vytvorenie funkcie kombinaciou sin a cos a maleho polynomu
    y_ciara = sum(a[i] * np.sin(b[i] * x_ciara + c[i]) for i in range(5))
    y_ciara += np.polyval(np.random.uniform(-1, 1, size=4), x_ciara)  # + maly polynom

    #vyberie body pre scatter plot
    y_body = np.interp(x_body, x_ciara, y_ciara)  # pouzitie interpolovanie na najdenie hodnot pre y
    return x_ciara, y_ciara, x_body, y_body


def konverzia_suradnic(x_ciara=0, y_ciara=0, x_body=0, y_body=0, velkost_bodu=0, styl_bodu=''): 
    fig, graf = plt.subplots(figsize=(IMG_WIDTH/DPI, IMG_HEIGHT/DPI), dpi=DPI)                          # figsize je v zaklade v placoch,takze je potrebna konverzia na pixel (vytvori sa graf)
    graf.plot(x_ciara, y_ciara, linestyle=nahodny_styl_ciary(), color = nahodna_farba(), linewidth = nahodna_hrubka_ciary())    # vykresli ciaru podla suradnic s nahodnym stylom, farbou a hrubkou
    graf.scatter(x_body, y_body, color = nahodna_farba(), s = velkost_bodu, marker = styl_bodu, zorder=4)   # vykresli body do grafu s nahodnou farbou, velkostou, stylom a poradim prekrytia
    graf.yaxis.grid(color='gray', linestyle=nahodny_styl_ciary_pozadia(), alpha=0.6)
    graf.xaxis.grid(color='gray', linestyle=nahodny_styl_ciary_pozadia(), alpha=0.6)
    
    # vykresli vodorovnu a zvislu pomocnu ciaru cez stred grafu s ciernou farbou a polovicnou hrubkou
    graf.axhline( color='black', linewidth=nahodna_hrubka_ciary()/2)
    graf.axvline( color='black', linewidth=nahodna_hrubka_ciary()/2)            
    graf.margins(0.02)              # nastavi male okraje okolo grafu, aby sa nic neodrezalo
    graf.axis('on')                 # zapne zobrazenie osi

    # takto sekcia sluzi na upravu bounding boxov pre jednotlive styly bodov, 
    # je to potrebné, kedze sa samozna kniznica matplotlib neposkytuje pozadovanu presnot
    if styl_bodu==".":
        # vypocet rozmeru bodu
        if velkost_bodu <= 40:
            pointBboxPadding = 0
        elif velkost_bodu > 40 and velkost_bodu <= 50:
            pointBboxPadding = -0.05
        elif velkost_bodu > 50 and velkost_bodu <= 60:
            pointBboxPadding = -0.1
        elif velkost_bodu > 60 and velkost_bodu <= 80:
            pointBboxPadding = -0.12
        elif velkost_bodu > 70 and velkost_bodu <= 90:
            pointBboxPadding = -0.12
        elif velkost_bodu > 90 and velkost_bodu <= 100:
            pointBboxPadding = -0.18
        elif velkost_bodu > 100 and velkost_bodu <= 110:
            pointBboxPadding = -0.23
        elif velkost_bodu > 110 and velkost_bodu <= 170:
            pointBboxPadding = -0.23
        elif velkost_bodu > 170 and velkost_bodu <= 190:
            pointBboxPadding = -0.28
        elif velkost_bodu > 190 and velkost_bodu <= 1500:
            pointBboxPadding = -0.33
    
    if styl_bodu=="v" or styl_bodu=="^" or styl_bodu=="<" or styl_bodu==">" or styl_bodu=="*" or styl_bodu=="+" or styl_bodu=="x":
        # vypocet rozmeru bodu
        if velkost_bodu <= 40:
            pointBboxPadding = 0.5
        elif velkost_bodu > 40 and velkost_bodu <= 90:
            pointBboxPadding = 0.3
        elif velkost_bodu > 90 and velkost_bodu <= 100:
            pointBboxPadding = 0.2
        elif velkost_bodu > 100 and velkost_bodu <= 120:
            pointBboxPadding = 0.1
        elif velkost_bodu > 120 and velkost_bodu <= 140:
            pointBboxPadding = 0.2
        elif velkost_bodu > 140 and velkost_bodu <= 160:
            pointBboxPadding = 0.1
        elif velkost_bodu > 160 and velkost_bodu <= 170:
            pointBboxPadding = 0.2
        elif velkost_bodu > 170 and velkost_bodu <= 1500:
            pointBboxPadding = 0.1
    
    if styl_bodu=="+":
        # vypocet rozmeru bodu
        if velkost_bodu <= 40:
            pointBboxPadding = 0.2
        elif velkost_bodu > 40 and velkost_bodu <= 50:
            pointBboxPadding = 0.3
        elif velkost_bodu > 50 and velkost_bodu <= 60:
            pointBboxPadding = 0.1
        elif velkost_bodu > 60 and velkost_bodu <= 80:
            pointBboxPadding = 0.2
        elif velkost_bodu > 80 and velkost_bodu <= 90:
            pointBboxPadding = 0.3
        elif velkost_bodu > 90 and velkost_bodu <= 100:
            pointBboxPadding = 0.2
        elif velkost_bodu > 100 and velkost_bodu <= 110:
            pointBboxPadding = 0.1
        elif velkost_bodu > 110 and velkost_bodu <= 170:
            pointBboxPadding = 0.2
        elif velkost_bodu > 170 and velkost_bodu <= 1500:
            pointBboxPadding = 0.1
            
    if styl_bodu=="s":
        # vypocet rozmeru bodu
        if velkost_bodu >= 40 and velkost_bodu <= 100:
            pointBboxPadding = 0.4
        if velkost_bodu > 100 and velkost_bodu <= 110:
            pointBboxPadding = 0.3
        if velkost_bodu > 110 and velkost_bodu <= 120:
            pointBboxPadding = 0.4
        if velkost_bodu > 120 and velkost_bodu <= 170:
            pointBboxPadding = 0.3
        if velkost_bodu > 170 and velkost_bodu <= 180:
            pointBboxPadding = 0.2
        if velkost_bodu > 180 and velkost_bodu <= 190:
            pointBboxPadding = 0.3
        if velkost_bodu > 190 and velkost_bodu <= 1500:
            pointBboxPadding = 0.2
    
    pointDimensionPix = round(fig.dpi*np.sqrt(velkost_bodu)/72) + pointBboxPadding*(round(fig.dpi*np.sqrt(velkost_bodu)/72)) # prepocet velkosti bodu na rozmer v pixeloch
    
    #pointDimensionPix = round(fig.dpi*np.sqrt(velkost_bodu)/72) 
    
    # prevod suradnic z datovych na pixelove
    graf.get_xbound() # vynuti aktualizacie transformacie 
    pixels = graf.transData.transform(np.vstack([x_body,y_body]).T)
    xpix, ypix = pixels.T
    
    # spravi sa prepocet na spravny suradnicovy system - zaciatok v lavom hornom rohu
    xpix = np.round(xpix).astype(int)
    ypix = np.round(-(ypix - IMG_HEIGHT)).astype(int)
    centersPix = list(zip(xpix, ypix)) # pozicia bodov v pixeloch

    # ulozi figuru do preprocess suboru 
    img_path_figures = os.path.join(OUTPUT_DIR_figures, f"scatter_{i}.png")
    plt.savefig(img_path_figures, bbox_inches=None, pad_inches=0)
    plt.close()

    return centersPix, pointDimensionPix, img_path_figures # vystupom su surdanice bodov, rozmer bodu, cesta kde sa ulozi vygenerovana figura

def ulozenie_udajov(centersPix, pointDimensionPix, img_path_figures, id):
    
    # zisti sa rozmer udajovej casti grafu
    img = cv2.imread(img_path_figures)
    # presne udaje som odvodil ,tak te som spustil jeden krat hough_transform(), aby som zistil, presne pixely na orezanie = urychluje to generovanie figur oproti pouziti funkcie
    croppOsX = 890
    croppOsY = 126
    croppHornaCiara = 120
    croppPravaCiara = 900
    
    # vystrihne sa obrazok bez osi a ulozi sa 
    roi = img[croppHornaCiara:croppOsX, croppOsY:croppPravaCiara ]
    hieghtCroppedIMG, widthCroppedIMG, _ = roi.shape 
    
    # 60% - TRAIN
    if i >= 0 and i < (0.6*NUM_IMAGES):
        # vytvori sa cesta pre image a label
        img_path = os.path.join(OUTPUT_DIR_train, "images", f"scatter_{i}.png")
        label_path = os.path.join(OUTPUT_DIR_train, "labels", f"scatter_{i}.txt")
    
    # 20% - VALIDATION
    #if i >= (0.6*NUM_IMAGES) and i < (0.8*NUM_IMAGES):
    elif i >= (0.6*NUM_IMAGES) and i < (0.8*NUM_IMAGES):
        # vytvori sa cesta pre image a label
        img_path = os.path.join(OUTPUT_DIR_valid, "images", f"scatter_{i}.png")
        label_path = os.path.join(OUTPUT_DIR_valid, "labels", f"scatter_{i}.txt")
    
    # 20% - TESTING
    elif i >= (0.8*NUM_IMAGES) :
        # vytvori sa cesta pre image a label
        img_path = os.path.join(OUTPUT_DIR_test, "images", f"scatter_{i}.png")
        label_path = os.path.join(OUTPUT_DIR_test, "labels", f"scatter_{i}.txt")  
    
    #ulozenie orezaneho obrazka 
    cv2.imwrite(img_path, roi)
        
    # ulozi suradnice bodov grafu do txt 
    with open(label_path, "w") as f:
        for center in centersPix:

            # prepocet suradnic x a y do normalizovanej formy xywh pre YOLO
            x_normalised = (center[0]-croppOsY)/widthCroppedIMG
            y_normalised = (center[1]-croppHornaCiara)/hieghtCroppedIMG
            pointWidthPix_normalised = pointDimensionPix/widthCroppedIMG
            pointHeightPix_normalised = pointDimensionPix/hieghtCroppedIMG
            
            #ulozenie normalizovanych suradnic pre YOLO
            f.write(f"{id} {x_normalised} {y_normalised} {pointWidthPix_normalised} {pointHeightPix_normalised} \n")
            
            #ulozenie klasickych suradnic so zaciatkom v lavom hornom rohu
            #f.write(f"0 {(center[0]-croppOsY)} {(center[1]-croppHornaCiara)} {pointDimensionPix} {pointDimensionPix} \n")
    return img_path, croppOsY, croppHornaCiara

def kontrola_datasetu(img_path,croppOsY,croppHornaCiara,pointDimensionPix):     
    # kontrola datasetu s opencv - nakreslenie bboxov
    image_file = img_path
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for center in centersPix:
        x = center[0]-croppOsY
        y = center[1]-croppHornaCiara
        cv2.rectangle(img_rgb, (x-int(pointDimensionPix/2), y-int(pointDimensionPix/2)), (x+int(pointDimensionPix/2), y+int(pointDimensionPix/2)), (0, 255, 0), 1)
    
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()
    
    
    
if __name__ == "__main__":
    
    id = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # ku kazdemu stylu bodu sa prida jendo id z pola
    styl_id = {
        styl: id for styl, id in zip(STYL_BODOV, id)
    }
    #print(styl_id)

    opakovanie_stylov = NUM_IMAGES // len(STYL_BODOV)  # kolko krat sa styl v datasete objavi - zabezpecuje rovnomerny pocet stylov datasetu
    vybrate_styly = STYL_BODOV * opakovanie_stylov  # vsetky styly * rovnomerny pocet - zabezpeci sa, ze v datasete bude rovnomerne zastupenie kazdeho stylu bodu
    random.shuffle(vybrate_styly) # zamiesa sa poradie stylov, aby sa zabezpecila roznorodost datasetu
    
    #vytvarenie grafov
    for i, styl_bodu in enumerate(vybrate_styly): #enumerate poskytuje index prvku, s ktorym sa akurat preacuje, uklada ho do i
        print(i)
        velkost_bodu = nahodna_velkost_bodu() # kazdy novy graf ma nahodnu velkost bodu 
        id = styl_id[styl_bodu]
        
        #nahondy vyber typu grafu
        typ_grafu = random.randint(0, 3)
        
        if typ_grafu == 0: # nahodne nespojite body
            x_body, y_body = generovanie_nahodnych_nespojitych_bodov()
            x_ciara = 0 
            y_ciara = 0
        elif typ_grafu == 1: # sinusoidy
            x_ciara, y_ciara, x_body, y_body = sinusoidy()
        elif typ_grafu == 2:
            x_body, y_body = borwnov_pohyb()
            x_ciara = x_body
            y_ciara = y_body
        elif typ_grafu == 3: # kombinacia viacerych funkcii v jednom grafe
            x_ciara, y_ciara, x_body, y_body = kombinacia_funkcii()
        
        centersPix, pointDimensionPix, img_path_figures = konverzia_suradnic(x_ciara, y_ciara, x_body, y_body, velkost_bodu, styl_bodu)
        image_path,croppOsY,croppHornaCiara  = ulozenie_udajov(centersPix, pointDimensionPix, img_path_figures, id)

        # ked sa spusti kod, uzivatel moze kontrolovat vykleslene body na udajovej casti obrazka
        #kontrola_datasetu(image_path,croppOsY,croppHornaCiara,pointDimensionPix)






    
    
        