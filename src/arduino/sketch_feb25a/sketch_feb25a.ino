#include <SoftwareWire.h>
#include <Adafruit_PWMServoDriver.h>

// Define the pins for SDA and SCL
#define SDA_PIN1 6
#define SCL_PIN1 7
#define SDA_PIN2 20
#define SCL_PIN2 21

// Create a SoftwareWire object
SoftwareWire myWire1(SDA_PIN1, SCL_PIN1);
SoftwareWire myWire2(SDA_PIN2, SCL_PIN2);

// #include <Wire.h>
// #define myWire1 Wire
// #define myWire2 Wire

void setup() {
  // Start the serial communication for USB connection
  Serial.begin(115200);
  while (!Serial); // Wait for the serial port to connect.

  // Initialize the SoftwareWire (I2C) communication
  myWire1.setClock(2);
  myWire1.begin();
  myWire2.setClock(2);
  myWire2.begin();

  Serial.println("Arduino Serial-to-I2C Forwarder Ready");
  while (Serial.available()) Serial.read();
}

void loop() {
  // Check if data is available on the serial port
  if (Serial.available()) {
    int i2cAddress = Serial.read();  // Read the I2C address (1 byte)
    int deviceID = Serial.read();    // Read the device ID (1 byte)
    int numRounds = Serial.read();   // Read the number of rounds to send (1 byte), at least 1
    char numByteList[numRounds];    // the number of bytes to send in each round
    for (int i = 0; i < numRounds; i++) {
      delayMicroseconds(50);  // wait for the serial
      numByteList[i] = Serial.read();  // Read the number of bytes to send in each round
    }

    auto myWire = myWire1;
    if (deviceID == 1) myWire = myWire1;
    else if (deviceID == 2) myWire = myWire2;
    else return;

    // Begin transmission to the I2C device
    myWire.beginTransmission(i2cAddress);

    // Allocate memory for data
    int maxNumBytes = 0;
    for (int i = 0; i < numRounds; i++) {
      int numBytes = numByteList[i];
      if (numBytes > maxNumBytes) maxNumBytes = numBytes;
      // if (i < 10) Serial.print(0);
      // Serial.print(i);
      // Serial.println(numBytes);
      // Serial.read(); Serial.read(); Serial.read(); Serial.read(); Serial.read();
    }

    // Read the bytes to send from the serial port and send them over I2C
    for (int i = 0; i < numRounds; i++) {
      int numBytes = numByteList[i];
      for (int j = 0; j < numBytes; j++) {
        delayMicroseconds(50);  // wait for the serial
        int data = Serial.read();
        myWire.write(data);        // Send the byte to the I2C device
      }
    }
    // End the I2C transmission
    myWire.endTransmission();
    Serial.println("Data forwarded to I2C device.");
    delayMicroseconds(50);
    for (int i = 0; i < 29; i++) Serial.read();
  }
  // else Serial.println("Serial not available");
  // for (int i = 0; i < 21; i++) Serial.read();
  // while(Serial.available()) {
  //   delayMicroseconds(50);
  //   Serial.read();  // clear serial
  // }
  delay(1);
}
