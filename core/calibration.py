#include "Adafruit_VL53L0X.h"

Adafruit_VL53L0X lox = Adafruit_VL53L0X();

void setup() {
  Serial.begin(115200);

  // Tunggu Serial Monitor terbuka
  while (! Serial) {
    delay(1);
  }
  
  Serial.println("Mulai membaca sensor VL53L0X...");
  
  // Inisialisasi Sensor
  if (!lox.begin()) {
    Serial.println("❌ Gagal mendeteksi VL53L0X! Cek kabel SDA/SCL.");
    while(1); // Berhenti di sini kalau gagal
  }
}

void loop() {
  VL53L0X_RangingMeasurementData_t measure;
  
  // Ambil data jarak
  lox.rangingTest(&measure, false);

  if (measure.RangeStatus != 4) {  // Angka 4 artinya out of range
    float distance = measure.RangeMilliMeter;
    
    // Mencegah error pembagian dengan nol
    if (distance > 0) {
      // Rumus Estimasi Perbesaran sesuai gambar
      float perbesaran = 60000.0 / distance;
      
      // Print ke Serial Monitor
      Serial.print("Jarak sensor: ");
      Serial.print(distance, 2);
      Serial.print(" mm | Estimasi perbesaran: ");
      Serial.print(perbesaran, 1);
      Serial.println("x");
    }
  } else {
    Serial.println(" out of range ");
  }
    
  delay(500); // Jeda setengah detik biar nggak terlalu ngebut
}
