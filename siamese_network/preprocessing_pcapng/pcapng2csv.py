"""
### PCAPNG to CSV Converter

This script converts PCAPNG files into CSV format by extracting specific network packet fields using `tshark`. It performs the following tasks:

1. **Validates PCAPNG Files:** Ensures the input files exist and have the correct `.pcap` or `.pcapng` extensions.
2. **Generates CSV Files:** Extracts selected fields from the PCAPNG files and writes them to CSV files with custom headers.
3. **Calculates Packet Rates:** Computes the rate (packets per second) based on inter-arrival times.
4. **Handles Multiple Inputs:** Supports processing single or multiple PCAPNG files, either individually or from a directory, with optional custom output filenames.
5. **Command-Line Interface:** Utilizes `argparse` for handling command-line arguments, allowing users to specify input files/directories and output destinations.

"""

import subprocess
import csv
import os
import sys
import argparse
import glob
import time


def calculate_rate(time_delta, min_time_diff=1e-6):
    """
    Calculate the rate (packets per second) based on the time difference.

    Args:
        time_delta (float): The time difference between packets.
        min_time_diff (float): Minimum time difference to avoid division by zero.

    Returns:
        float: Calculated rate or 0 if time_delta is too small.
    """
    if time_delta < min_time_diff:
        return 0  # Avoid division by zero
    return 1 / time_delta


def validate_pcap_file(pcapng_file):
    """
    Validate that the PCAPNG file exists and has a correct extension.

    Args:
        pcapng_file (str): Path to the PCAPNG file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not have a .pcap or .pcapng extension.
    """
    try:
        if not os.path.isfile(pcapng_file):
            raise FileNotFoundError(f"Error: The file '{pcapng_file}' does not exist.")

        if not pcapng_file.endswith('.pcap') and not pcapng_file.endswith('.pcapng'):
            raise ValueError(f"Error: The file '{pcapng_file}' is not a valid PCAP or PCAPNG file.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit(1)

    except ValueError as val_error:
        print(val_error)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error during validation: {e}")
        sys.exit(1)


def generate_csv(pcapng_file, output_file):
    """
    Generate a CSV file from the PCAPNG data using tshark with custom headers.

    Args:
        pcapng_file (str): Path to the input PCAPNG file.
        output_file (str): Path to the output CSV file.
    """
    fields = [
        'ip.ttl', 'ip.hdr_len', '_ws.col.Protocol', 'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset',
        'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.ece', 'tcp.flags.cwr', 'frame.len', 'frame.time_delta',
        'ip.flags.mf', 'tcp.len', 'mqtt.conack.flags', 'mqtt.conflag.cleansess', 'mqtt.conflag.qos',
        'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.willflag', 'mqtt.conflags', 'mqtt.dupflag',
        'mqtt.hdrflags', 'mqtt.kalive', 'mqtt.len', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.qos',
        'mqtt.retain', 'mqtt.ver'
    ]

    custom_headers = {
        'ip.ttl': 'Time_To_Leave',
        'ip.hdr_len': 'Header_Length',
        '_ws.col.Protocol': 'Protocol_Type',
        'tcp.flags.fin': 'TCP_Flag_FIN',
        'tcp.flags.syn': 'TCP_Flag_SYN',
        'tcp.flags.reset': 'TCP_Flag_RST',
        'tcp.flags.push': 'TCP_Flag_PSH',
        'tcp.flags.ack': 'TCP_Flag_ACK',
        'tcp.flags.ece': 'TCP_Flag_ECE',
        'tcp.flags.cwr': 'TCP_Flag_CWR',
        'frame.len': 'Packet_Length',
        'frame.time_delta': 'IAT',  # Inter-arrival time
        'ip.flags.mf': 'Packet_Fragments',
        'tcp.len': 'TCP_Length',
        'mqtt.conack.flags': 'MQTT_ConAck_Flags',
        'mqtt.conflag.cleansess': 'MQTT_CleanSession',
        'mqtt.conflag.qos': 'MQTT_Conflag_QoS',
        'mqtt.conflag.reserved': 'MQTT_Reserved',
        'mqtt.conflag.retain': 'MQTT_Conflag_Retain',
        'mqtt.conflag.willflag': 'MQTT_WillFlag',
        'mqtt.conflags': 'MQTT_ConFlags',
        'mqtt.dupflag': 'MQTT_DupFlag',
        'mqtt.hdrflags': 'MQTT_HeaderFlags',
        'mqtt.kalive': 'MQTT_KeepAlive',
        'mqtt.len': 'MQTT_Length',
        'mqtt.msgtype': 'MQTT_MessageType',
        'mqtt.proto_len': 'MQTT_Proto_Length',
        'mqtt.qos': 'MQTT_QoS',
        'mqtt.retain': 'MQTT_Retain',
        'mqtt.ver': 'MQTT_Version'
    }

    cmd = ['tshark', '-r', pcapng_file, '-T', 'fields']
    for field in fields:
        cmd += ['-e', field]

    cmd += ['-E', 'separator=;']

    headers = ['type_attack', 'rate'] + [custom_headers.get(field, field) for field in fields]

    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(headers)

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Error during tshark execution: {stderr.strip()}")

            for line in stdout.splitlines():
                fields_values = line.strip().split(';')

                # Ensure the correct number of fields
                if len(fields_values) != len(fields):
                    print(f"Skipping packet: expected {len(fields)} fields, got {len(fields_values)}.")
                    continue

                try:
                    time_delta = float(fields_values[fields.index('frame.time_delta')])
                except ValueError:
                    continue  # Skip packets with invalid timestamps

                rate = calculate_rate(time_delta)

                # 'type_attack' is left empty for now
                row = [''] + [rate] + fields_values
                writer.writerow(row)

        print(f"The results have been saved to {output_file}.")
    except Exception as e:
        print(f"Error during CSV generation: {e}")


def main():
    """
    Main function to parse arguments and process PCAPNG files.
    """
    parser = argparse.ArgumentParser(
        description="Convert PCAPNG files to CSV with selected fields and custom headers.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-i', '--input', nargs='+',
        help="Path to the PCAPNG file(s) or directory containing PCAPNG files."
    )

    parser.add_argument(
        '-o', '--output',
        help="Path to the output CSV file or directory where the results will be saved."
    )

    parser.add_argument(
        '-n', '--names', nargs='*',
        help="Optional custom output CSV file names (if multiple input files are provided)."
    )

    args = parser.parse_args()

    print("=========== PCAPNG2CSV ===========")

    if not args.input:
        print("Error: The -i or --input flag is required to specify the PCAPNG file(s) or directory.")
        sys.exit(1)

    if not args.output:
        print("Error: The -o or --output flag is required to specify the output CSV file or directory.")
        sys.exit(1)

    output_dir = str(args.output)

    # If input is a directory, process all PCAPNG files within
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        pcapng_files = glob.glob(os.path.join(args.input[0], '*.pcapng'))
        if not pcapng_files:
            print(f"No PCAPNG files found in directory: {args.input[0]}")
            sys.exit(1)

        if not os.path.isdir(output_dir):
            print("Error: The output must be a directory when processing multiple files from a directory.")
            sys.exit(1)

        for pcapng_file in pcapng_files:
            input_file = str(pcapng_file)
            csv_filename = os.path.join(output_dir, os.path.basename(input_file).replace('.pcapng', '.csv'))
            generate_csv(input_file, csv_filename)

    else:
        # Handle multiple input files with optional custom names
        if args.names and len(args.names) != len(args.input):
            print("Error: The number of custom output names must match the number of input files.")
            sys.exit(1)

        for i, pcapng_file in enumerate(args.input):
            validate_pcap_file(pcapng_file)
            input_file = str(pcapng_file)

            if args.names:
                name = args.names[i]
                output_file = os.path.join(
                    output_dir,
                    name if name.endswith('.csv') else f"{name}.csv"
                )
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.pcapng', '.csv'))

            generate_csv(input_file, output_file)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds\n")
