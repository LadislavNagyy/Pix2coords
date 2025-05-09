from ultralytics import YOLO
import cv2
def main():
    
    #TRENOVANIE
    model = YOLO("yolo11m.pt")

    vysledky = model.train(data = "C:\\Users\\Machine\\Desktop\\LadislavNagy\\trenovanie_rtx4070ti_v6\\data.yaml", 
                imgsz = 640, 
                batch = 16, 
                epochs = 400, 
                workers = 4, 
                device = 0,
                patience = 15, #ak sa val_loss nezlepsi pocas 20 epoch, trening ukonci
                cache = True,
                save_period=5)
    
    """
    #test modelu na testovacom datasete
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(source="test_obr1.png", show=True, save=True, conf=0.5)
    """
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # nie je povinné, ale odporúčané na Windows
    main()


