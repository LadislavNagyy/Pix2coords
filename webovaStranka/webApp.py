import os
import cv2
from flask import Flask, render_template, request, send_from_directory, url_for, redirect
from mainScript import globalne_premenne, priprava_obrazka, detekcia_ciar, ohranicenie_cisel_osi, rozpoznanie_textu_osi, predikcia_bodov, ulozenie_suradnic_doCSV 

POLOVENE_SUBORY = {'png', 'jpg', 'jpeg'}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# funkcia sluzi na vytvorenjie priecinka /static/uploads, ak este neexistuje.
def vytvorenie_priecinka_uploads():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#vymaze vsetko v /static/uploads
def vycistenie_priecinka_uploads(): 
    if os.path.exists(UPLOAD_FOLDER):
        for file_name in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            try:
                os.remove(file_path)
                print(f"Odstraneny subor: {file_path}")
            except Exception as e:
                print(f"Chyba pri odstranovani subora: {file_path}: {e}")

webApp = Flask(__name__)

#volanie funkcii na, vytvorenie priecinka na mahravanie a premazanie obsahu priecinka
vytvorenie_priecinka_uploads()
vycistenie_priecinka_uploads()

webApp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in POLOVENE_SUBORY

# inicializácia globalnych premennych z mainScript.py
globalne_premenne()

# cesta a funkcia na spracovanie domovskej stránky
@webApp.route("/", methods=['GET', 'POST'])
def run():
    return render_template('index.html')

# cesta a funkcia na spracovanie upload stranky
@webApp.route('/upload', methods=['GET', 'POST'])  
def upload_page():
    chyba = None # zabezpeci, ze je premenna chyba vzdy definovana, aby nespadla aplikacia 
    if request.method == 'POST':
        try:
            # prevzatie priecnika
            file = request.files['file']

            if file and allowed_file(file.filename):
                
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                vstupny_obrazok = cv2.imread(file_path) 
                
                print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------") # vizualne oddelenie pre prehladnejsei haldanie v logu servera
                print(f"Názov súboru: {file.filename}")
                
                gray, finalImg = priprava_obrazka(vstupny_obrazok)
                pozicia_detekovanych_ciar = detekcia_ciar(gray, finalImg)
                x_axis_boundingBox, y_axis_boundingBox, textHeight = ohranicenie_cisel_osi(gray, finalImg, pozicia_detekovanych_ciar)
                stredy_kontur_cisel, cisla_kontur_cisel = rozpoznanie_textu_osi(x_axis_boundingBox, y_axis_boundingBox, textHeight, gray, finalImg)
                
                # rozsah hodnot/cisel osi x podla mierky
                cislo_lavej_x_kontury = float(cisla_kontur_cisel[0][0])
                cislo_pravej_x_kontury = float(cisla_kontur_cisel[0][1])
                # rozsah hodnot/cisel osi y podla mierky
                cislo_dolnej_y_kontury = float(cisla_kontur_cisel[0][2])
                cislo_hornej_y_kontury = float(cisla_kontur_cisel[0][3])
                
                suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi = predikcia_bodov(pozicia_detekovanych_ciar, stredy_kontur_cisel, cisla_kontur_cisel, finalImg, vstupny_obrazok) #pouziva predtrenovany YOLOv11 model na detekciu bodov
                ulozenie_suradnic_doCSV(suradniceBodovX_VrozsahuOsi, suradniceBodovY_VrozsahuOsi)
                
                # konverzia z np.float64 na klasicky float - toto bola chyba na stranke, ked bezala v dockri
                suradniceBodovX_VrozsahuOsi = [float(x) for x in suradniceBodovX_VrozsahuOsi]
                suradniceBodovY_VrozsahuOsi = [float(y) for y in suradniceBodovY_VrozsahuOsi]           
                
                
                return render_template('index.html',
                                    suradnice_osx=suradniceBodovX_VrozsahuOsi,
                                    suradnice_osy=suradniceBodovY_VrozsahuOsi,
                                    cislo_lavej_x_kontury=cislo_lavej_x_kontury,
                                    cislo_pravej_x_kontury=cislo_pravej_x_kontury,
                                    cislo_dolnej_y_kontury=cislo_dolnej_y_kontury,
                                    cislo_hornej_y_kontury=cislo_hornej_y_kontury,
                                    obrazok_vlozeny='/static/uploads/' + file.filename,
                                    result='1',
                                    chyba=None)
                    
        except Exception as e:
            print(f"[CHYBA] {e}") #vypis chyby do konzoly
            chyba = "Obrázok sa nepodarilo spracovať, prosím skontrolujte, či spĺňa požiadavky aplikácie."
    
    result = request.args.get('result')
    return render_template('index.html', result=result, chyba=chyba)

if __name__ == '__main__':
    webApp.run(host="0.0.0.0", port=8080, debug=True)
