from ultralytics import YOLO    # importovanie knižnice Ultralytics
def main():
    
    #TRENOVANIE
    model = YOLO("yolo11l.pt")  # načítanie modelu zo súboru

    vysledky = model.train(data = "C:\\Users\\Machine\\Desktop\\LadislavNagy\\trenovanie_rtx4070ti_v6\\data.yaml", 
                imgsz = 640,    # velkosť obrázkov
                batch = 16,     # počet obrázkov v jednéj dávke
                epochs = 400,   # maximálny počet epôch
                workers = 4,    # počet vlákien na načítanie dát
                device = 0,     # použitie grafickéj karty
                patience = 15,  # ak sa val_loss nezlepší počas 20 epôch, tréning sa ukončí
                cache = 'disk', # cahovanie dát na disk, pre rýchlejšie načítanie
                save_period=5)  # každých 5 epoch a uloží nový model
    

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # nie je povinné, ale odporúčané na Windows
    main()


