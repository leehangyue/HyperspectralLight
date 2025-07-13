#include <Adafruit_PWMServoDriver.h>
#include <string.h>  // combines bytes into int or float
#include <Wire.h>
#define myWire Wire  // might be conveniently changed to other objects later on, if Adafruit_PWMServoDriver supports

int baud = 9600;
int delay_us = 1000000 / baud * 9;
static Adafruit_PWMServoDriver pwm(64, myWire);

void setup() {
  // Start the serial communication for USB connection
  Serial.begin(baud);
  while (!Serial); // Wait for the serial port to connect.

  // Initialize the SoftwareWire (I2C) communication
myWire.setClock(2);
  myWire.begin();

  Serial.println("Arduino Serial-to-I2C Forwarder Ready");
  while (Serial.available()) Serial.read();
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available() > 2) {
    Serial.println("Receiving data...");
    int address1 = Serial.read();  // Read the address of the I2C device 1
    int address2 = Serial.read();  // Read the address of the I2C device 2
    delayMicroseconds(delay_us);  // wait for the serial
    int numCh = Serial.read();   // Read the number of channels to send (1 byte), at least 1
    Serial.print("Address1: "); Serial.println(address1);
    Serial.print("Address2: "); Serial.println(address2);
    Serial.print("Number of channels per address: "); Serial.println(numCh);
    
    int addresses[2];
    addresses[0] = address1;
    addresses[1] = address2;

    for (int idx_addr = 0; idx_addr < 2; idx_addr++) {

      char byteList[numCh * 4];    // the bytes indicating on and off phase information for each channel
      delayMicroseconds(delay_us * numCh * 4);  // wait for the serial
      if (!Serial.available() >= numCh * 4) return;
      for (int i = 0; i < numCh * 4; i++) {
        delayMicroseconds(delay_us);  // wait for the serial
        byteList[i] = Serial.read();  // Read the number of bytes to send in each round
      }
      // for (int i = 0; i < 4; i++) Serial.read();  // flush the ending bytes

      // Create Adafruit_PWMServoDriver objects
      pwm = Adafruit_PWMServoDriver(addresses[idx_addr], myWire);
      pwm.begin();
      // pwm.setOscillatorFrequency(25000000);
      // pwm.setPWMFreq(1500);  // This is the PWM frequency in Hz

      uint16_t onPhase;
      uint16_t offPhase;
      uint8_t x1;
      uint8_t x2;
      // Serial.println("Channel ID\tOn/off phase\tReceived bytes");
      for (uint8_t ch_id = 0; ch_id < numCh; ch_id++) {
        x1 = byteList[ch_id * 4 + 0];
        x2 = byteList[ch_id * 4 + 1];
        onPhase = (x2 << 8) | x1;
        x1 = byteList[ch_id * 4 + 2];
        x2 = byteList[ch_id * 4 + 3];
        offPhase = (x2 << 8) | x1;
        // Serial.print(ch_id); Serial.print("\t\t");Serial.print(onPhase); Serial.print(", "); Serial.print(offPhase); Serial.print("\t\t");
        // for (int i=0; i<4; i++) {x1 = byteList[ch_id * 4 + i]; Serial.print(x1); Serial.print(", ");}
        // Serial.println();
        pwm.setPWM(ch_id, onPhase, offPhase);
        delayMicroseconds(delay_us * 4);  // wait for whatever
      }
    }

    Serial.println("Command sent to I2C device.");
  }
  // delay(1);  // delay in milliseconds
  return;
}
