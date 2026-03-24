Design and implement a smart system using the MPU6050 sensor to collect motion data, integrate a `1D CNN` deep learning model to classify fall events, 
and send alerts via channels such as MQTT, email, and siren on the ESP32 microcontroller platform. The system achieved `98.12% accuracy` on the test set, 
responded in less than 1 second when detecting falls, and consumed low resources.
However, there are still limitations such as latency in email/IFTTT sending, risk of false alarms, and lack of real-world data for verification. 
Suggested improvements include real-world data collection, add-on sensor integration, power optimization, and user interface development.
See more here:[Demo](https://youtu.be/AjiiydAeXwQ)
