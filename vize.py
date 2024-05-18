import cv2
import numpy as np

class KirmiziTopTespit:
    def __init__(self, video_yolu):
        # Video yakalama cihazını oluştur
        self.video = cv2.VideoCapture(video_yolu)
        
        # Kırmızı ve beyaz renklerin HSV aralıklarını tanımla
        self.alt_kirmizi = np.array([125, 100, 100])  # Kırmızı renk alt aralığı
        self.ust_kirmizi = np.array([180, 255, 255])  # Kırmızı renk üst aralığı
        self.alt_beyaz = np.array([0, 0, 200])        # Beyaz renk alt aralığı
        self.ust_beyaz = np.array([180, 25, 255])     # Beyaz renk üst aralığı
        
        # Kırmızı topun durma eşiğini belirle
        self.kirmizi_durma_esigi = 2
        
        # Önceki kırmızı ve beyaz topun merkezini saklamak için değişkenler
        self.onceki_merkez_kirmizi = None
        self.onceki_merkez_beyaz = None
        
        # Başlangıç ve bitiş zamanlarını saklamak için değişkenler
        self.baslangic_zamani = None
        self.bitis_zamani = None

    def top_hizi_hesapla(self, merkez_kirmizi):
        # Önceki merkez varsa, topun hızını hesapla
        if self.onceki_merkez_kirmizi is not None:
            mesafe = np.sqrt((merkez_kirmizi[0] - self.onceki_merkez_kirmizi[0]) ** 2 + (merkez_kirmizi[1] - self.onceki_merkez_kirmizi[1]) ** 2)
            zaman_farki = 1 / self.video.get(cv2.CAP_PROP_FPS)
            if mesafe < self.kirmizi_durma_esigi:
                hiz = 0
            else:
                hiz = mesafe / zaman_farki
            return hiz
        return 0

    def run(self):
        while True:
            # Videodan bir çerçeve al
            ret, frame = self.video.read()
            if not ret:
                break

            # Çerçeve HSV renk uzayına dönüştür
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # hsv=cv2.resize(hsv,(600,380))
            # cv2.imshow('Top Takibi hsv', hsv)
            
            # Kırmızı ve beyaz renkler için maske oluştur
            kirmizi_maske = cv2.inRange(hsv, self.alt_kirmizi, self.ust_kirmizi)  # Kırmızı renk maskeleme
            # kirmizi_maske=cv2.resize(kirmizi_maske,(600,380)) 
            # cv2.imshow('Top Takibi kirmizi_maske1', kirmizi_maske)
            
            kirmizi_maske = cv2.medianBlur(kirmizi_maske, 9)                       # Kırmızı maskeyi yumuşatma
            # kirmizi_maske=cv2.resize(kirmizi_maske,(600,380)) 
            # cv2.imshow('Top Takibi kirmizi_maske2', kirmizi_maske)
            
            beyaz_maske = cv2.inRange(hsv, self.alt_beyaz, self.ust_beyaz)        # Beyaz renk maskeleme

            # Kırmızı için eşikleme yap
            ret_red, kirmizi_esik = cv2.threshold(kirmizi_maske, 127, 160, cv2.THRESH_BINARY)
            # kirmizi_esik=cv2.resize(kirmizi_esik,(600,380)) 
            # cv2.imshow('Top Takibi kirmizi_esik', kirmizi_esik)
            
            # Morfolojik açma ve kapama uygula
            kernel = np.ones((6,3), np.uint8)
            kirmizi_esik = cv2.morphologyEx(kirmizi_esik, cv2.MORPH_OPEN, kernel)   # Kırmızı için morfolojik açma 
            # kirmizi_esik=cv2.resize(kirmizi_esik,(600,380)) 
            # cv2.imshow('Top Takibi kirmizi_esik', kirmizi_esik)
            
            beyaz_maske = cv2.morphologyEx(beyaz_maske, cv2.MORPH_OPEN, kernel)     # Beyaz için morfolojik açma

            # Kırmızı ve beyaz konturları bul
            kirmizi_konturlar, _ = cv2.findContours(kirmizi_esik.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
            
            beyaz_konturlar, _ = cv2.findContours(beyaz_maske.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Kırmızı topu izle ve hızını hesapla
            for kontur in kirmizi_konturlar:
                if cv2.contourArea(kontur) > 35:
                    # Konturun minimum çemberini bulma
                    ((x, y), yaricap) = cv2.minEnclosingCircle(kontur)
                    # Çembere biraz ekstra boyut verme ve çizme
                    merkez_kirmizi = (int(x), int(y))
                    cv2.circle(frame, (int(x), int(y)), int(yaricap) + 3, (255,15,21),2)
                    
                    # Kırmızı topun hızını hesaplama
                    hiz = self.top_hizi_hesapla(merkez_kirmizi)
                    
                    # Hız bilgisini ekrana yazdırma
                    text = f'Kirmizi Top Hizi: {hiz:.2f} piksel/saniye'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    text_width = text_size[0][0]
                    cv2.putText(frame, text, (int((frame.shape[1] - text_width) / 2), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                                                    # Çerçevenin genişiliği-Yazıının genişliği
                                                    #int((frame.shape[1] - text_width) / 2) ifadesi, metnin sol kenarını çerçevenin sol kenarına uzaklığını belirler
                    # Kırmızı topun etiketini çizme
                    cv2.putText(frame, 'Kirmizi Top', (int(x - yaricap), int(y - yaricap - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 15, 15), 1, cv2.LINE_AA)
                                #Metnin konumu, burada x ve y kırmızı topun merkez koordinatlarıdır ve yaricap ise kırmızı topun yarıçapıdır.
                    # Önceki kırmızı top merkezini güncelleme
                    self.onceki_merkez_kirmizi = merkez_kirmizi

            # Beyaz topun vuruş anını ve durma anını belirle
            for kontur in beyaz_konturlar:
                if cv2.contourArea(kontur) > 30:
                    ((x, y), yaricap) = cv2.minEnclosingCircle(kontur)#Beyaz topun merkezi ve yarıçapı bulunur.
                    merkez_beyaz = (int(x), int(y))#Beyaz topun merkez koordinatları alınır.

                    if self.onceki_merkez_beyaz is not None:#Eğer önceki beyaz top merkezi bilgisi mevcutsa:
                        
                        #Önceki merkez ile mevcut merkez arasındaki mesafe hesaplanır.
                        mesafe = np.sqrt((merkez_beyaz[0] - self.onceki_merkez_beyaz[0]) ** 2 + (merkez_beyaz[1] - self.onceki_merkez_beyaz[1]) ** 2)
                        
                        
                        #Başlangıç zamanı henüz belirlenmemişe ve mesafe 5'ten büyükse, hareket başlamış kabul edilir ve başlangıç zamanı belirlenir.
                        if self.baslangic_zamani is None and mesafe > 5:
                            self.baslangic_zamani = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            
                        #Başlangıç zamanı belirlenmişse ve mesafe 5'ten küçükse, hareketin bittiği kabul edilir ve bitiş zamanı belirlenir.
                        elif self.baslangic_zamani is not None and mesafe < 2:
                            self.bitis_zamani = self.video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    self.onceki_merkez_beyaz = merkez_beyaz# Mevcut beyaz top merkezi, bir sonraki iterasyonda kullanılmak üzere önceki merkez olarak saklanır.

            # Vuruş anını ve durma anını ekrana yazdır
            if self.baslangic_zamani is not None:
                text_size = cv2.getTextSize(f"Beyaz Topa Vurulma Ani: {self.baslangic_zamani:.2f}. saniyede", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_width = text_size[0][0]
                cv2.putText(frame, f"Beyaz Topa Vurulma Ani: {self.baslangic_zamani:.2f}. saniyede", (int((frame.shape[1] - text_width) / 2), frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if self.bitis_zamani is not None:
                text_size = cv2.getTextSize(f"Toplarin Durma Ani {self.bitis_zamani:.2f}. saniyede", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                text_width = text_size[0][0]
                cv2.putText(frame, f"Toplarin Durma Ani {self.bitis_zamani:.2f}. saniyede", (int((frame.shape[1] - text_width) / 2), frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            #cv2.imshow('Top Takibi kirmizi_esik', kirmizi_esik) 
            
            
            # Çerçeveyi ekrana göster
            cv2.imshow('Top Takibi', frame)
            
            # Çıkış tuşu (q) ile döngüyü sonlandır
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Sonuçları yazdır
        print("Beyaz Topa Vurulma Ani:", self.baslangic_zamani)
        print("Toplarin Durma Ani:", self.bitis_zamani)

        # Video kaynağını serbest bırak ve pencereleri kapat
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    top_tespit = KirmiziTopTespit('C:\\Goruntu_Isleme_Vize\\Veri_seti\\vid_1.avi')
    top_tespit.run()
