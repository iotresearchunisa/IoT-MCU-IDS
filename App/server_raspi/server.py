import time
import paho.mqtt.client as mqtt
import logging
import threading
import pandas as pd
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from datetime import datetime

# Configurazione MQTT
MQTT_BROKER = '192.168.14.240'
MQTT_PORT = 1883
MQTT_TOPIC = 'rasberry/topic'
CLIENT_ID = "rasberry"
MQTT_TOPIC_PUBLISH = 'esp8266/topic'
MQTT_TOPIC_SUBSCRIBE = 'rasberry/topic'

# Configurazione OAUTH2.0
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/home/alberto/watchdog_raspi/iot-raspi.json'
FILENAME = 'iot.csv'
POLL_INTERVAL = 1  # Intervallo di polling in secondi

# Configurazione logging
logging.basicConfig(filename='/home/alberto/watchdog_raspi/iot-raspi.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Lock
condition = threading.Condition()
message_received = False
message_published = False
connected = False
subscribed = False


def connect_mqtt():
    global connected

    def on_connect(client, userdata, flags, rc, properties):
        global connected
        with condition:
            if rc == 0:
                connected = True
                logging.info("Connected to MQTT Broker!")
                print("Connected to MQTT Broker!")
            else:
                connected = False
                logging.error(f"Failed to connect, return code {rc}")
                print(f"Failed to connect, return code {rc}")
            condition.notify_all()

    def on_disconnect(client, userdata, flags, rc, properties):
        global connected, subscribed
        with condition:
            connected = False
            subscribed = False
            logging.info(f"Disconnected with result code: {rc}")
            print(f"Disconnected with result code: {rc}")
            condition.notify_all()

    client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    while not connected:
        try:
            client.connect(MQTT_BROKER, MQTT_PORT)
            client.loop_start()  # Avvia il loop in un thread separato
            with condition:
                condition.wait_for(lambda: connected)  # Attende la connessione
        except ConnectionRefusedError as e:
            print(f"Connection refused, retrying in 5 seconds... ({e})")
            logging.error(f"Connection refused, retrying in 5 seconds... ({e})")
            time.sleep(5)

    return client


def publish(client, msg):
    global message_published
    result = client.publish(MQTT_TOPIC_PUBLISH, msg)
    status = result.rc

    if status == mqtt.MQTT_ERR_SUCCESS:
        print(f"Send `{msg}` to topic `{MQTT_TOPIC_PUBLISH}`\n")
        logging.info(f"Send `{msg}` to topic `{MQTT_TOPIC_PUBLISH}`\n")
    else:
        print(f"Failed to send message to topic {MQTT_TOPIC_PUBLISH}\n")
        logging.error(f"Failed to send message to topic {MQTT_TOPIC_PUBLISH}\n")

    with condition:
        message_published = True
        condition.notify_all()


#############################
# Messaggi da ESP8266 a APP #
#############################
def subscribe(client: mqtt.Client, service):

    def on_message(client, userdata, msg):
        global message_received
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        logging.info(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

        operation_map = {
            "autenticato": "Accesso consentito",
            "tentativo_di_accesso": "Tentativo di accesso",
            "persona_rilevata": "Persona rilevata",
            "persona_rilevata_end": "Persona andata via",
            "porta_aperta": "Porta aperta",
            "autenticazione_fallita": "Autenticazione fallita",
            "allarme": "Allarme"
        }

        with condition:
            operation = operation_map.get(msg.payload.decode())
            append_to_csv(service, operation)
            message_received = True
            condition.notify_all()  # Notifica che il messaggio è stato ricevuto e i comandi sono stati eseguiti

    def on_publish(client, userdata, mid, granted_qos, properties):
        global message_published
        with condition:
            message_published = True
            condition.notify_all()  # Notifica che il messaggio è stato pubblicato

    def on_subscribe(client, userdata, mid, granted_qos, properties):
        global subscribed
        print("Subscribed to topic!\n")
        logging.info("Subscribed to topic!\n")

        with condition:
            subscribed = True
            condition.notify_all()  # Notifica che la sottoscrizione è stata completata

    client.on_message = on_message
    client.on_publish = on_publish
    client.on_subscribe = on_subscribe
    client.subscribe(MQTT_TOPIC_SUBSCRIBE)


def authenticate():
    try:
        print("Autenticazione in corso attendi...")
        logging.info("Autenticazione in corso attendi...")
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        logging.error(f"Errore durante l'autenticazione: {e}")
        print(f"Errore durante l'autenticazione: {e}")
        return None


#############################
# Messaggi da APP a ESP8266 #
#############################
def process_dataframe(df, client, service):
    if not df.empty:
        # Prende l'ultima riga del DataFrame
        last_row = df.iloc[-1]

        if 'operation' in last_row and 'time' in last_row:
            action = last_row['operation'].lower()

            if action == 'blocca porta':
                publish(client, 'blocca')
                condition.wait_for(lambda: message_published)  # Attende la fine della pubblicazione

            elif action == 'apri porta':
                publish(client, 'porta_aperta')
                condition.wait_for(lambda: message_published)

            elif action == 'allarme':
                publish(client, 'allarme')
                condition.wait_for(lambda: message_published)


def append_to_csv(service, operation):
    try:
        df, file_id = read_csv_from_drive(service)
        if df is None:
            logging.error("Errore durante il download del file CSV.")
            print("Errore durante il download del file CSV.")
            return

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_row = pd.DataFrame({'time': [now], 'operation': [operation]})
        df = pd.concat([df, new_row], ignore_index=True)

        if operation == 'Autenticazione fallita':
            now_1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_row_1 = pd.DataFrame({'time': [now_1], 'operation': ["Blocca porta"]})
            df = pd.concat([df, new_row_1], ignore_index=True)

            now_2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            new_row_2 = pd.DataFrame({'time': [now_2], 'operation': ["Allarme"]})
            df = pd.concat([df, new_row_2], ignore_index=True)

        upload_csv_to_drive(service, file_id, df)
    except Exception as e:
        logging.error(f"Errore durante l'aggiunta di una nuova riga al file CSV: {e}")
        print(f"Errore durante l'aggiunta di una nuova riga al file CSV: {e}")


def read_csv_from_drive(service):
    buffer = BytesIO()
    file_id = get_file_id(service)

    if not file_id:
        print("Errore: file non trovato.")
        return None, None

    try:
        request = service.files().get_media(fileId=file_id)

        downloader = MediaIoBaseDownload(buffer, request)
        logging.info("Lettura file, attendi ...")
        print("Lettura file, attendi ...")
        done = False
        while not done:
            status, done = downloader.next_chunk()

        buffer.seek(0)
        df = pd.read_csv(buffer)
        return df, file_id
    except HttpError as error:
        print(f"Errore durante il download del file CSV: {error}")
        logging.error(f"Errore durante il download del file CSV: {error}")
        return None, None
    except Exception as e:
        print(f"Errore imprevisto durante il download del file CSV: {e}")
        logging.error(f"Errore imprevisto durante il download del file CSV: {e}")
        return None, None
    finally:
        buffer.close()


def upload_csv_to_drive(service, file_id, df):
    buffer = BytesIO()
    try:
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        media = MediaIoBaseUpload(buffer, mimetype='text/csv', resumable=True)
        service.files().update(fileId=file_id, media_body=media).execute()
    except HttpError as error:
        print(f"Errore durante l'upload del file CSV: {error}")
        logging.error(f"Errore imprevisto durante il download del file CSV: {error}")
    except Exception as e:
        print(f"Errore imprevisto durante l'upload del file CSV: {e}")
        logging.error(f"Errore imprevisto durante l'upload del file CSV: {e}")
    finally:
        buffer.close()


def get_file_id(service):
    try:
        results = service.files().list(q=f"name='{FILENAME}'", fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            return None
        return items[0]['id']
    except HttpError as error:
        logging.error(f"Errore durante la ricerca del file CSV: {error}")
        print(f"Errore durante la ricerca del file CSV: {error}")
        return None
    except Exception as e:
        logging.error(f"Errore imprevisto durante la ricerca del file CSV: {e}")
        print(f"Errore imprevisto durante la ricerca del file CSV: {e}")
        return None


def get_file_metadata(service):
    try:
        results = service.files().list(q=f"name='{FILENAME}'", fields="files(id, modifiedTime)").execute()
        items = results.get('files', [])
        if not items:
            return None, None
        return items[0]['id'], items[0]['modifiedTime']
    except HttpError as error:
        logging.error(f"Errore durante la ricerca del file CSV: {error}")
        print(f"Errore durante la ricerca del file CSV: {error}")
        return None, None
    except Exception as e:
        logging.error(f"Errore imprevisto durante la ricerca del file CSV: {e}")
        print(f"Errore imprevisto durante la ricerca del file CSV: {e}")
        return None, None


def main():
    global message_received, message_published, connected, subscribed

    service = authenticate()
    while service is None:
        logging.error("Autenticazione fallita. Ritento in 5 secondi.")
        print("Autenticazione fallita. Ritento in 5 secondi.")
        time.sleep(5)
        service = authenticate()

    file_id, last_modified_time = None, None
    while not file_id:
        logging.info(f"Attesa del file {FILENAME} su Google Drive...\n")
        print(f"Attesa del file {FILENAME} su Google Drive...\n")
        file_id = get_file_id(service)
        if not file_id:
            time.sleep(5)  # Attesa di 5 secondi prima di ritentare

    client = connect_mqtt()

    with condition:
        condition.wait_for(lambda: connected)  # Attende la connessione

    subscribe(client, service)
    with condition:
        condition.wait_for(lambda: subscribed)  # Attende la sottoscrizione

    last_processed = ''
    first_message = True
    print(f"Controllo modifiche al file {FILENAME} su Google Drive...")
    logging.info(f"Controllo modifiche al file {FILENAME} su Google Drive...")

    while True:
        with condition:
            if not connected:
                print("Attempting to reconnect...")
                while not connected:
                    try:
                        client.reconnect()
                        condition.wait_for(lambda: connected)  # Attende la riconnessione
                        subscribe(client, service)
                        condition.wait_for(lambda: subscribed)  # Attende la risottoscrizione
                    except ConnectionRefusedError as e:
                        print(f"Reconnect failed, retrying in 5 seconds... ({e})")
                        time.sleep(5)

            if not message_received and not message_published and subscribed and connected:

                try:
                    _, modified_time = get_file_metadata(service)

                    if first_message:
                        last_modified_time = modified_time
                        first_message = False

                    if modified_time != last_modified_time and not first_message:
                        print("File modificato.\n")
                        logging.info(f"{FILENAME} è stato modificato.\n")
                        df, file_id = read_csv_from_drive(service)

                        if df is not None and isinstance(df, pd.DataFrame):
                            process_dataframe(df, client, service)
                            _, last_modified_time = get_file_metadata(service)

                            print(f"Controllo modifiche al file {FILENAME} su Google Drive...")
                            logging.info(f"Controllo modifiche al file {FILENAME} su Google Drive...")
                        else:
                            logging.error("Errore durante il download del file CSV.")
                            print("Errore durante il download del file CSV.")

                except HttpError as error:
                    logging.error(f"Errore nella comunicazione con Google Drive: {error}")
                    print(f"Errore nella comunicazione con Google Drive: {error}")
                    time.sleep(5)
                    service = authenticate()
                    if service is None:
                        logging.error("Autenticazione fallita dopo un errore. Uscita.")
                        print("Autenticazione fallita dopo un errore. Uscita.")
                        break
                except Exception as e:
                    logging.error(f"Errore imprevisto: {e}")
                    print(f"Errore imprevisto: {e}")
                    break

                time.sleep(POLL_INTERVAL)  # Attesa prima del prossimo controllo
            message_received = False
            message_published = False
        time.sleep(1)  # Aggiunta una breve pausa per evitare un ciclo troppo veloce


if __name__ == '__main__':
    logging.info("---------------------------------------------------------")
    main()
