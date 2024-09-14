"""

This Python script is designed to handle PCAPNG (Packet Capture Next Generation) files used in network traffic analysis.
The main functionalities of the program include:

1. **PCAPNG File Validation**: Checks if the input file exists and has a valid extension (.pcap or .pcapng).
2. **Splitting the PCAPNG File**: Divides the PCAPNG file into a specified number of parts using `editcap`.
3. **Conversion to CSV**: Converts each part of the PCAPNG file into a CSV file using `tshark`, enriching the data with additional information such as the packet rate per second.
4. **Merging CSV Files**: Combines all generated CSV files into a single final CSV file.
5. **Cleanup**: Removes temporary directories created during execution.

This tool is useful for analyzing large PCAPNG files by breaking them down into more manageable parts and converting them into an easily analyzable CSV format.

Prerequisites:
- `editcap` and `tshark` tools installed and accessible in the system PATH.
- Adequate permissions to read input files and write to output directories.

"""

import subprocess
import csv
import os
import sys
import argparse
import time
import shutil
import glob


def calculate_rate(time_delta, min_time_diff=1e-6):
    """
    Calculates the packet rate per second.

    Args:
        time_delta (float): Time difference between packets.
        min_time_diff (float): Minimum time difference to avoid division by zero.

    Returns:
        float: Packet rate per second.
    """
    if time_delta < min_time_diff:
        return 0
    return 1 / time_delta


def validate_pcap_file(pcapng_file):
    """
    Validates the existence and extension of the PCAPNG file.

    Args:
        pcapng_file (str): Path to the PCAPNG file.

    Raises:
        SystemExit: If the file does not exist or does not have a valid extension.
    """
    if not os.path.isfile(pcapng_file):
        print(f"Error: The file '{pcapng_file}' does not exist.")
        sys.exit(1)
    if not (pcapng_file.endswith('.pcap') or pcapng_file.endswith('.pcapng')):
        print(f"Error: The file '{pcapng_file}' is not a valid PCAP or PCAPNG file.")
        sys.exit(1)


def split_pcapng(pcapng_file, num_parts, temp_dir):
    """
    Splits a PCAPNG file into a specified number of parts using editcap.

    Args:
        pcapng_file (str): Path to the original PCAPNG file.
        num_parts (int): Number of parts to split the file into.
        temp_dir (str): Temporary directory to save the split parts.

    Raises:
        subprocess.CalledProcessError: If the editcap command fails.
    """
    base_name = os.path.splitext(os.path.basename(pcapng_file))[0]
    output_file = os.path.join(temp_dir, f"{base_name}_part.pcapng")

    # Command to split the PCAPNG file into equal parts
    cmd = ['editcap', '-c', str(num_parts), pcapng_file, output_file]

    try:
        subprocess.run(cmd, check=True)
        print(f"Splitting of '{pcapng_file}' completed.")
    except subprocess.CalledProcessError:
        print(f"Error during splitting of '{pcapng_file}'.")
        sys.exit(1)


def generate_csv(pcapng_file, output_file):
    """
    Generates a CSV file from a PCAPNG file using tshark.

    Args:
        pcapng_file (str): Path to the PCAPNG file to convert.
        output_file (str): Path to the output CSV file.
    """
    # Define the fields to extract with tshark
    fields = [
        'ip.ttl', 'ip.hdr_len', '_ws.col.Protocol', 'tcp.flags.fin', 'tcp.flags.syn',
        'tcp.flags.reset', 'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.ece',
        'tcp.flags.cwr', 'frame.len', 'frame.time_delta', 'ip.flags.mf', 'tcp.len',
        'mqtt.conack.flags', 'mqtt.conack.flags.reserved', 'mqtt.conack.flags.sp',
        'mqtt.conack.val', 'mqtt.conflag.cleansess', 'mqtt.conflag.passwd',
        'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain',
        'mqtt.conflag.uname', 'mqtt.conflag.willflag', 'mqtt.conflags',
        'mqtt.dupflag', 'mqtt.hdrflags', 'mqtt.kalive', 'mqtt.len',
        'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos',
        'mqtt.retain', 'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver',
        'mqtt.willmsg', 'mqtt.willmsg_len', 'mqtt.willtopic', 'mqtt.willtopic_len'
    ]

    # Mapping of custom fields for CSV headers
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
        'frame.time_delta': 'IAT',
        'ip.flags.mf': 'Packet_Fragments',
        'tcp.len': 'TCP_Length',
        'mqtt.conack.flags': 'MQTT_ConAck_Flags',
        'mqtt.conack.flags.reserved': 'MQTT_ConAck_Reserved',
        'mqtt.conack.flags.sp': 'MQTT_ConAck_SP',
        'mqtt.conack.val': 'MQTT_ConAck_Val',
        'mqtt.conflag.cleansess': 'MQTT_CleanSession',
        'mqtt.conflag.passwd': 'MQTT_Password',
        'mqtt.conflag.qos': 'MQTT_QoS',
        'mqtt.conflag.reserved': 'MQTT_Reserved',
        'mqtt.conflag.retain': 'MQTT_Retain',
        'mqtt.conflag.uname': 'MQTT_Username',
        'mqtt.conflag.willflag': 'MQTT_WillFlag',
        'mqtt.conflags': 'MQTT_ConFlags',
        'mqtt.dupflag': 'MQTT_DupFlag',
        'mqtt.hdrflags': 'MQTT_HeaderFlags',
        'mqtt.kalive': 'MQTT_KeepAlive',
        'mqtt.len': 'MQTT_Length',
        'mqtt.msgtype': 'MQTT_MessageType',
        'mqtt.proto_len': 'MQTT_Proto_Length',
        'mqtt.protoname': 'MQTT_Proto_Name',
        'mqtt.qos': 'MQTT_QoS',
        'mqtt.retain': 'MQTT_Retain',
        'mqtt.sub.qos': 'MQTT_Sub_QoS',
        'mqtt.suback.qos': 'MQTT_SubAck_QoS',
        'mqtt.ver': 'MQTT_Version',
        'mqtt.willmsg': 'MQTT_WillMsg',
        'mqtt.willmsg_len': 'MQTT_WillMsg_Length',
        'mqtt.willtopic': 'MQTT_WillTopic',
        'mqtt.willtopic_len': 'MQTT_WillTopic_Length'
    }

    # Build the tshark command
    cmd = ['tshark', '-r', pcapng_file, '-T', 'fields']
    for field in fields:
        cmd += ['-e', field]
    cmd += ['-E', 'separator=;']  # Set delimiter to ";"

    # Prepare custom headers for the CSV
    headers = ['type_attack', 'rate'] + [custom_headers.get(field, field) for field in fields]

    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(headers)

            # Execute the tshark command and capture the output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Error during tshark execution: {stderr.strip()}")

            # Process each line of tshark output
            for line in stdout.splitlines():
                fields_values = line.strip().split(';')
                try:
                    time_delta = float(fields_values[fields.index('frame.time_delta')])
                except (ValueError, IndexError):
                    continue  # Skip packets with invalid timestamps

                rate = calculate_rate(time_delta)
                row = ['', rate] + fields_values  # Add empty column for 'type_attack' and 'rate'
                writer.writerow(row)

        print(f"CSV saved to '{output_file}'.")
    except Exception as e:
        print(f"Error during CSV generation: {e}")


def merge_csvs(csv_files, output_file):
    """
    Merges multiple CSV files into a single CSV file.

    Args:
        csv_files (list): List of paths to CSV files to merge.
        output_file (str): Path to the final CSV file.
    """
    with open(output_file, 'w', newline='') as merged_file:
        writer = None
        for csv_file in sorted(csv_files):
            with open(csv_file, 'r') as infile:
                reader = csv.reader(infile, delimiter=';')
                headers = next(reader, None)  # Read headers

                if writer is None:
                    writer = csv.writer(merged_file, delimiter=';')
                    writer.writerow(headers)  # Write headers once

                for row in reader:
                    writer.writerow(row)
        print(f"CSV files merged into '{output_file}'.")


def main():
    """
    Main function that handles the execution of the script.
    """
    parser = argparse.ArgumentParser(
        description="Split a PCAPNG file, convert the parts to CSV, and merge them into a single final CSV."
    )
    parser.add_argument('-i', '--input', required=True, help="Path to the input PCAPNG file")
    parser.add_argument('-p', '--parts', type=int, required=True, help="Number of parts to split the PCAPNG file into")
    parser.add_argument('-o', '--output',
                        help="Destination directory for the final CSV file (default: same directory as the input file)")
    parser.add_argument('-n', '--name', help="Name of the final merged CSV file (without the .csv extension)")

    args = parser.parse_args()

    pcapng_file = args.input
    num_parts = args.parts

    # Determine the name of the final CSV file
    base_name = os.path.splitext(os.path.basename(pcapng_file))[0]
    final_csv_name = f"{args.name if args.name else base_name}.csv"

    # Determine the output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        final_csv = os.path.join(output_dir, final_csv_name)
    else:
        final_csv = os.path.join(os.path.dirname(pcapng_file), final_csv_name)

    # Create a temporary directory for the split parts
    temp_dir = os.path.join(os.path.dirname(pcapng_file), f"{base_name}_temp")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Validate the input file
        validate_pcap_file(pcapng_file)

        # Split the PCAPNG file into parts
        split_pcapng(pcapng_file, num_parts, temp_dir)

        # Retrieve the split parts
        pcapng_parts = glob.glob(os.path.join(temp_dir, "*.pcapng"))

        if not pcapng_parts:
            print(f"Error: No PCAPNG files found in '{temp_dir}'.")
            sys.exit(1)

        print(f"Found {len(pcapng_parts)} parts.")

        # Convert each part to CSV
        csv_files = []
        for i, part_file in enumerate(pcapng_parts, start=1):
            csv_output = os.path.join(temp_dir, f"part_{i}.csv")
            generate_csv(part_file, csv_output)
            csv_files.append(csv_output)

        # Merge the CSV files into the final output
        merge_csvs(csv_files, final_csv)

    finally:
        # Cleanup: remove the temporary directory
        shutil.rmtree(temp_dir)
        print(f"Temporary directory '{temp_dir}' removed.")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
