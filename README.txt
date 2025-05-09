Toto je projekt, ktorý je zároveň aj mojou bakalárskou prácou: Aplikácia na detekciu karteziánskych súradnic na obrázku grafu.

Priečonok trenovanie_yolo

    - obsahuje trénovací skript použitý pri trénovaní siete, 
    - súbor data.yaml, ktorý je potrebný pre správne trénovanie siete,
    - nachádza sa tu aj príkladový súbor data_na_trenovanie, ktorý slúži, ako ukážka štruktúry datasetu. Nemohol nahrať celý dataset, keďže by sa do Github repozitára nezmestil.

Priečinok tvorba_datasetu

    - obsahuje skript na tvorbu datasetu, ktorý vytvorí obrázky a potrebné lable pre trénovanie

Priečinok webStranka
    - obsahuje celý projekt webovéj stránky
    - Pre spustenie aplikácie je potrebné stiahnuť si najnovší yolo model vo formate .pt z https://drive.google.com/drive/folders/17gW88IgDhLENJbM4i9HP-xdk8MvLH4PB?usp=sharing do priečinka webStranka, taktiež je potrebné stiahnuť knižnice, ktoré sa v projekte nachádzajú