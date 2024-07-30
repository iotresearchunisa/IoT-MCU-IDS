import datetime
import time
import pandas as pd
import pyfiglet
import threading
from io import BytesIO
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.prompt import Prompt
from rich.text import Text

# Costanti
SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'iot_app.json'
FILENAME = 'iot.csv'
MIMETYPE = 'text/csv'
HEADERS = ['time', 'operation']

console = Console()


def authenticate():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('drive', 'v3', credentials=creds)


def get_file_id(service):
    try:
        results = service.files().list(q=f"name='{FILENAME}'", fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            return None
        return items[0]['id']
    except HttpError as error:
        console.print(f"[red]Errore durante la ricerca del file CSV: {error}[/red]")
        exit(1)
    except Exception as e:
        console.print(f"[red]Errore imprevisto durante la ricerca del file CSV: {e}[/red]")
        exit(1)


def check_and_create_csv(service):
    file_id = get_file_id(service)
    if not file_id:
        console.print(f"[yellow]{FILENAME} non esiste, lo creo.[/yellow]")
        df = pd.DataFrame(columns=HEADERS)
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        file_metadata = {'name': FILENAME}
        media = MediaIoBaseUpload(buffer, mimetype=MIMETYPE, resumable=True)
        try:
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            console.print(f"[green]{FILENAME} creato con successo.[/green]")
        except HttpError as error:
            console.print(f"[red]Errore durante la creazione del file CSV: {error}[/red]")
            exit(1)
        except Exception as e:
            console.print(f"[red]Errore imprevisto durante la creazione del file CSV: {e}[/red]")
            exit(1)
    else:
        console.print(f"[green]{FILENAME} già esiste.[/green]")


def read_csv_from_drive(service):
    file_id = get_file_id(service)
    if not file_id:
        console.print("[red]Errore: file non trovato.[/red]")
        return None, None

    try:
        request = service.files().get_media(fileId=file_id)
        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        buffer.seek(0)
        return pd.read_csv(buffer), file_id
    except HttpError as error:
        console.print(f"[red]Errore durante il download del file CSV: {error}[/red]")
        return None, None
    except Exception as e:
        console.print(f"[red]Errore imprevisto durante il download del file CSV: {e}[/red]")
        return None, None


def upload_csv_to_drive(service, df, file_id):
    try:
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        media = MediaIoBaseUpload(buffer, mimetype=MIMETYPE, resumable=True)
        service.files().update(fileId=file_id, media_body=media).execute()
    except HttpError as error:
        console.print(f"[red]Errore durante l'upload del file CSV: {error}[/red]")
        exit(1)
    except Exception as e:
        console.print(f"[red]Errore imprevisto durante l'upload del file CSV: {e}[/red]")
        exit(1)


def show_file(service, period=None):
    try:
        df, _ = read_csv_from_drive(service)
        if df is None:
            return

        if period:
            try:
                datetime.datetime.strptime(period, '%Y')
                filtered_df = df[df['time'].str.startswith(period)]
            except ValueError:
                try:
                    datetime.datetime.strptime(period, '%Y-%m')
                    filtered_df = df[df['time'].str.startswith(period)]
                except ValueError:
                    try:
                        datetime.datetime.strptime(period, '%Y-%m-%d')
                        filtered_df = df[df['time'].str.startswith(period)]
                    except ValueError:
                        console.print("[red]Formato data non valido. Usa 'yyyy', 'yyyy-mm' o 'yyyy-mm-dd'.[/red]")
                        return
            df = filtered_df

        table = Table(title="\nIoT Operations", show_lines=True)
        table.add_column("Time", justify="center", style="cyan")
        table.add_column("Operation", justify="center", style="magenta")

        if df.empty:
            console.print("[yellow]La tabella è vuota.[/yellow]")
        else:
            for _, row in df.iterrows():
                table.add_row(row['time'], row['operation'])

        console.print(table)
    except Exception as e:
        console.print(f"[red]Errore durante la visualizzazione del file: {e}[/red]")


def progress_bar(operation_func, *args, **kwargs):
    with Progress(SpinnerColumn(), BarColumn(), TextColumn("{task.description}"), TimeElapsedColumn()) as progress:
        task = progress.add_task("[cyan]Operazione in corso...", total=None)
        result = [None]  # To capture the result of the operation

        def wrapper():
            result[0] = operation_func(*args, **kwargs)

        thread = threading.Thread(target=wrapper)
        thread.start()
        while thread.is_alive():
            time.sleep(0.1)
            progress.advance(task)
        progress.update(task, completed=100, description="[green]Operazione completata[/green]")
        return result[0]


def log_operation(service, operation):
    def operation_logic():
        now_1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df, file_id = read_csv_from_drive(service)
        if df is None:
            return None
        new_row = pd.DataFrame({'time': [now_1], 'operation': [operation]})
        df = pd.concat([df, new_row], ignore_index=True)
        upload_csv_to_drive(service, df, file_id)
        return now_1

    now = progress_bar(operation_logic)
    if now:
        console.print(f"[green]\nOperazione '{operation}' registrata alle {now}.[/green]")


def delete_rows(service, period):
    def operation_logic():
        df, file_id = read_csv_from_drive(service)
        if df is None:
            return False
        original_len = len(df)
        try:
            datetime.datetime.strptime(period, '%Y')
            df = df[~df['time'].str.startswith(period)]
        except ValueError:
            try:
                datetime.datetime.strptime(period, '%Y-%m')
                df = df[~df['time'].str.startswith(period)]
            except ValueError:
                try:
                    datetime.datetime.strptime(period, '%Y-%m-%d')
                    df = df[~df['time'].str.startswith(period)]
                except ValueError:
                    console.print("[red]Formato data non valido. Usa 'yyyy', 'yyyy-mm' o 'yyyy-mm-dd'.[/red]")
                    return False
        if len(df) == original_len:
            return False
        else:
            upload_csv_to_drive(service, df, file_id)
            return True

    success = progress_bar(operation_logic)
    if success:
        console.print(f"\n[green]Righe con il periodo {period} eliminate con successo.[/green]")
    else:
        console.print(f"\n[yellow]Nessuna riga trovata con il periodo {period}.[/yellow]")


def clear_file(service):
    def operation_logic():
        df, file_id = read_csv_from_drive(service)
        if df is None:
            return False
        df = pd.DataFrame(columns=HEADERS)
        upload_csv_to_drive(service, df, file_id)
        return True

    success = progress_bar(operation_logic)
    if success:
        console.print(f"\n[green]File '{FILENAME}' svuotato con successo.[/green]")


def main():
    console.print('############################################\n', style="bold blue")
    console.print(Text(pyfiglet.figlet_format("IoT Cloud"), style="bold blue"))
    console.print('############################################\n\n', style="bold blue")

    service = authenticate()
    check_and_create_csv(service)

    while True:
        console.print("\n\n##################################\n"
                      "# Scegli un'operazione:          #\n"
                      "# 1. Mostra file                 #\n"
                      "# 2. Blocca porta                #\n"
                      "# 3. Apri porta                  #\n"
                      "# 4. Allarme                     #\n"
                      "# 5. Elimina righe               #\n"
                      "# 6. Svuota file                 #\n"
                      "#                                #\n"
                      "# 7. Esci                        #\n"
                      "##################################")

        choice = Prompt.ask("\nInserisci il numero dell'operazione", choices=['1', '2', '3', '4', '5', '6', '7'])

        if choice == '1':
            date = Prompt.ask('Inserisci il periodo (yyyy, yyyy-mm, yyyy-mm-dd) '
                              'oppure premi invio per mostrare tutto il file', default="")
            show_file(service, date)
        elif choice == '2':
            log_operation(service, 'Blocca porta')
        elif choice == '3':
            log_operation(service, 'Apri porta')
        elif choice == '4':
            log_operation(service, 'allarme')
        elif choice == '5':
            period = Prompt.ask("Inserisci il periodo delle righe da eliminare (yyyy, yyyy-mm, yyyy-mm-dd)")
            delete_rows(service, period)
        elif choice == '6':
            clear_file(service)
        elif choice == '7':
            break
        else:
            console.print("[red]Scelta non valida. Riprova.[/red]")


if __name__ == '__main__':
    main()
