const char ssid[] = "Rasberry-AP";
const char password[] = "Password123";

#define HOSTNAME "arduino"

IPAddress MQTT_BROKER(192, 168, 14, 240);
const int MQTT_PORT = 8883;
const char MQTT_USER[] = "arduino"; 
const char MQTT_PASS[] = "%A3duino_W!F%";

const char MQTT_SUB_ARDUINO[] = "arduino/topic";
const char MQTT_PUB_ESP82[] = "esp8266/topic";


static const char* ca_cert =
  "-----BEGIN CERTIFICATE-----\n" \
  "MIIDpzCCAo+gAwIBAgIUeMyN+J/Wc+cR1grJ5PSGewcpegEwDQYJKoZIhvcNAQEL\n" \
  "BQAwYzELMAkGA1UEBhMCSVQxDjAMBgNVBAgMBUl0YWx5MRAwDgYDVQQHDAdTYWxl\n" \
  "cm5vMQ4wDAYDVQQKDAVVbmlzYTEOMAwGA1UECwwFVW5pc2ExEjAQBgNVBAMMCXVu\n" \
  "aXNhLmNvbTAeFw0yNDA3MjcxOTE5MThaFw0zNDA3MjUxOTE5MThaMGMxCzAJBgNV\n" \
  "BAYTAklUMQ4wDAYDVQQIDAVJdGFseTEQMA4GA1UEBwwHU2FsZXJubzEOMAwGA1UE\n" \
  "CgwFVW5pc2ExDjAMBgNVBAsMBVVuaXNhMRIwEAYDVQQDDAl1bmlzYS5jb20wggEi\n" \
  "MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDDA1HLRDq11RephG7MDvbstyQi\n" \
  "qwSec8GvEIOA5BsXkBOi14gxkJMoln62vVaGaDZBZN7QNSw6CMXz3KOGFY+N0oZU\n" \
  "LNkGPCKjy+UCa3QT0jxcI1y9eMvMWXIMvpkBQPCvfS7wyozn4rUnjyrp7ASt50Z/\n" \
  "mhff2d9F7uPLTUwtJt4RmUlF/4SBIKOKDRjn87M+r3Ovg4x9tnUbdWeFFGK1slHk\n" \
  "vWUMlWJq606JyE37KyRPLmtDtwX2rRqSIEcjIHNEpSlVWFweUmAujKoUmlWsmnhZ\n" \
  "Oowg4j+RxzVZ5LQhT4ZcxtvJx508W0Kl+0ZZKAM3gcnZ9+P3gip+kMXwyvVrAgMB\n" \
  "AAGjUzBRMB0GA1UdDgQWBBTixY9b3Ef5YYMldgxBJDf8pl97PzAfBgNVHSMEGDAW\n" \
  "gBTixY9b3Ef5YYMldgxBJDf8pl97PzAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3\n" \
  "DQEBCwUAA4IBAQAo8nZ6GeMyARjp8Y3XtGGy8LlcpOoNZWZOf4QLDTGHHkFy9qHS\n" \
  "Tu6EuEmZMjWCR8HO/aAa1c3RHbqcUT98Ie8uBDq56ewBa7TkfrCoTY00G9ltgYb1\n" \
  "Xgm15L022FJLfJNxjSYyZ0eL/8XIpbYy/j8VzA2PMWNXH6sHY+sxfpLpB4N+4m/c\n" \
  "vTchagvuW+UThoxqXWZItXjn28IRQeAdB/LWJNuruIDgO9V8wgFguZ2OleT2AAAy\n" \
  "md/umjrjRGqMrZQOfz0ktv37sgS/eabwB1TZYmqsBdI3EXnQkSrOlVOMz7gc/CZG\n" \
  "yHaEP8bt5pw4a+y0noP1hgS/3SjJd6vKdESy\n" \
  "-----END CERTIFICATE-----";