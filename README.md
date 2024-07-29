# Dataset of cyber attacks on a real IoT Cloud-based Architecture
....

## Table of Contents
- [Project Overview](#project-overview)
- [IoT Cloud-based Repos](#iot-cloud-based-repos)
- [Architecture](#architecture)
  - [Devices](#devices)
  - [Sensors](#sensors)
  - [Schema](#schema)
- [Setup Raspberry Pi 3 Model B](#setup-raspberry-pi-3-model-b)
- [Authors](#authors)

## Project Overview
...

## IoT Cloud-based Repos
You can find all project repos in the following locations:
- [IoT_sketches](https://github.com/Alberto-00/Thesis-IoT_Cloud_based/tree/main/IoT_sketches): here you can find a documentation of how the devices used were configured and programmed;
- [papers](https://github.com/Alberto-00/Thesis-IoT_Cloud_based/tree/main/papers): here you can find the papers we used as a reference;
- ...


## Architecture
The IoT Cloud-based architecture we are going to consider represents the configuration of a domotic door that allows access into the home via a fingerprint reader and proximity sensor (it detects the presence of strangers). 

### Devices
The devices involved in the architecture are as follows:

<br>
<div align="center">
  
| WeMos D1 ESP8266 WiFi Board | TTGO LoRa32-OLED       |
|-----------------------------|------------------------|
| <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/boards/esp8266.png" alt="Esp8266" width="300"></p>            | <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/boards/esp32.png" alt="Esp32" width="300"></p>       |
| Arduino Uno WiFi Rev2       | Raspberry Pi 3 Model B |
| <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/boards/arduino.png" alt="Arduino" width="300"></p>          | <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/boards/raspberry.png" alt="Raspberry" width="400"></p> |

</div>

### Sensors
The sensors involved in the architecture are as follows:

|               | Arduino Uno WiFi  |  ESP32 |
|---------------|-------------------|-------------------|
| **Buzzer 3 Pins**     | <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/sensors/buzzer.png" alt="buzzer" width="150"></p> | |
| **HC-SR501 Pir Motion Detector**   | <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/sensors/HC-SR501.png" alt="HC-SR501" width="150"></p>| |
| **Adafruit Fingerprint Sensor** | | <p align="center"><img src="https://github.com/Alberto-00/Thesis-IoT_Cloud_based/blob/main/documents/img/sensors/fingerprint.png" alt="fingerprint" width="200"></p> |

### Schema
The logic of the architecture is divided into several levels:

- **End nodes**: *ESP32* with fingerprint sensor and *Arduino Uno WiFi Rev2* with proximity sensors and buzzer;

- **Edge node**: *ESP8266* receives the information sent by *ESP32* and *Arduino* via *MQTT protocol*, filters it and sends it to the *Raspberry Pi*;

- **Fog node**: the *Raspberry Pi* is networked via a wired connection to the modem, while, via the WLAN, it performs the functions of DHCP server (to provide IP addresses to the boards) and MQTT broker. The WLAN interface is configured so that the *Raspberry Pi* acts as an access point to allow communication between all the boards. Configuration of the WLAN network was done by setting a static IP to the *Raspberry Pi* on a network other than the Ethernet network, thus ensuring a separation between the two networks and optimizing network traffic management. The *Raspberry Pi* also collects the information transmitted from the *ESP8266*, updates a CSV file on Google Drive and sends data from an app client to *ESP8266*.

<br>
<p align="center">
  <img src="documents/img/architecture.svg" alt="Architecture IoT" width="600">
</p>

## Setup Raspberry Pi 3 Model B


## Authors
| Name | Description |
| --- | --- |
| <p dir="auto"><strong>Alberto Montefusco</strong> |<br>Developer - <a href="https://github.com/Alberto-00">Alberto-00</a></p><p dir="auto">Email - <a href="mailto:a.montefusco28@studenti.unisa.it">a.montefusco28@studenti.unisa.it</a></p><p dir="auto">LinkedIn - <a href="https://www.linkedin.com/in/alberto-montefusco">Alberto Montefusco</a></p><p dir="auto">My WebSite - <a href="https://alberto-00.github.io/">alberto-00.github.io</a></p><br>|
