import subprocess
import csv
import os
import sys
import argparse
import glob

# Function to calculate the rate (packets per second)
def calculate_rate(time, min_time_diff=1e-6):
    """
    Calculate the rate only if the time difference is greater than a minimum value (to avoid division by zero).
    """
    if time < min_time_diff:
        return 0  # Rate infinito o molto alto
    return 1 / time


# Function to validate the PCAP file
def validate_pcap_file(pcapng_file):
    """
    Check if the file exists and if it is a valid PCAP or PCAPNG file.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(pcapng_file):
            raise FileNotFoundError(f"Error: The file '{pcapng_file}' does not exist.")

        # Check if the file has a valid extension
        if not pcapng_file.endswith('.pcap') and not pcapng_file.endswith('.pcapng'):
            raise ValueError(f"Error: The file '{pcapng_file}' is not a valid PCAP or PCAPNG file.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        sys.exit(1)

    except ValueError as val_error:
        print(val_error)
        sys.exit(1)

    except Exception as e:
        # Catch any other exceptions
        print(f"Unexpected error during validation: {e}")
        sys.exit(1)


# Function to generate the CSV
def generate_csv(pcapng_file, output_file):
    """
    Generate a CSV file from the PCAP data using tshark and save the result with custom header names.
    """
    # Lista dei campi che vogliamo estrarre con tshark
    fields = [
        'ip.ttl', 'ip.hdr_len', '_ws.col.Protocol', 'tcp.flags.fin', 'tcp.flags.syn', 'tcp.flags.reset',
        'tcp.flags.push', 'tcp.flags.ack', 'tcp.flags.ece', 'tcp.flags.cwr', 'frame.len',
        'frame.time_delta', 'ip.flags.mf', 'tcp.len', 'mqtt.conack.flags', 'mqtt.conack.flags.reserved',
        'mqtt.conack.flags.sp', 'mqtt.conack.val', 'mqtt.conflag.cleansess', 'mqtt.conflag.passwd',
        'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain', 'mqtt.conflag.uname',
        'mqtt.conflag.willflag', 'mqtt.conflags', 'mqtt.dupflag', 'mqtt.hdrflags', 'mqtt.kalive',
        'mqtt.len', 'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain',
        'mqtt.sub.qos', 'mqtt.suback.qos', 'mqtt.ver', 'mqtt.willmsg', 'mqtt.willmsg_len',
        'mqtt.willtopic', 'mqtt.willtopic_len'
    ]

    # Map of custom names for the CSV headers
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
        'mqtt.msg': 'MQTT_Message',
        'mqtt.msgid': 'MQTT_MessageID',
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

    # Build the tshark command with the specified fields
    cmd = ['tshark', '-r', pcapng_file, '-T', 'fields']
    for field in fields:
        cmd += ['-e', field]

    # Add the option for the delimiter ";"
    cmd += ['-E', 'separator=;']

    # Prepare the custom headers for the CSV
    headers = ['type_attack'] + ['rate'] + [custom_headers.get(field, field) for field in fields]

    # Apri il file CSV per la scrittura
    try:
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            # Write the CSV header with custom names
            writer.writerow(headers)

            # Run the tshark command and read the output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

            # Check if there are any errors during the execution of tshark
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Error during tshark execution: {stderr.strip()}")

            # Iterate over each packet
            for line in stdout.splitlines():
                fields_values = line.strip().split(';')

                # Extract frame.time_delta and calculate rate
                try:
                    time_delta = float(fields_values[fields.index('frame.time_delta')])
                except ValueError:
                    continue  # Skip packets with invalid timestamps

                # Calculate the rate
                rate = calculate_rate(time_delta)

                # Create the row to write (type_attack column empty for now)
                row = [''] + [rate] + fields_values

                # Write the row to the CSV file
                writer.writerow(row)

        print(f"The results have been saved to {output_file}.")
    except Exception as e:
        print(f"Error during CSV generation: {e}")


def main():
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Convert PCAPNG files and generate CSV(s) with various features",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Parameter for the input files or directory
    parser.add_argument('-i', '--input', nargs='+',
                        help="Path to the PCAPNG file(s) or directory containing PCAPNG files\n\n")

    # Parameter for the output file or directory
    parser.add_argument('-o', '--output',
                        help="Path to the output CSV file or directory where the results will be saved\n\n")

    # Optional parameter to specify custom output file names
    parser.add_argument('-n', '--names', nargs='*',
                        help="Optional custom output CSV file names (if multiple files are given)\n\n")

    args = parser.parse_args()

    print("=========== PCAPNG2CSV ===========")

    # Check if input and output parameters are provided
    if not args.input:
        print("Error: The -i or --input flag is required to specify the PCAPNG file(s) or directory.")
        sys.exit(1)

    if not args.output:
        print("Error: The -o or --output flag is required to specify the output CSV file or directory.")
        sys.exit(1)

    # Ensure the output is a string (to avoid issues with os.path.join())
    output_dir = str(args.output)

    # Case 1: If the input is a directory, process all PCAPNG files in the directory
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        pcapng_files = glob.glob(os.path.join(args.input[0], '*.pcapng'))
        if not pcapng_files:
            print(f"No PCAPNG files found in directory: {args.input[0]}")
            sys.exit(1)

        if not os.path.isdir(output_dir):
            print("Error: The output must be a directory when processing multiple files from a directory.")
            sys.exit(1)

        for pcapng_file in pcapng_files:
            # Ensure pcapng_file is a string
            input_file = str(pcapng_file)
            csv_filename = os.path.join(output_dir, os.path.basename(input_file).replace('.pcapng', '.csv'))
            generate_csv(input_file, csv_filename)

    # Case 2: If multiple files or a single file is given, process them
    else:
        # Validate if custom names are provided and match the number of input files
        if args.names and len(args.names) != len(args.input):
            print("Error: The number of custom output names must match the number of input files.")
            sys.exit(1)

        for i, pcapng_file in enumerate(args.input):
            validate_pcap_file(pcapng_file)

            # Ensure pcapng_file is a string
            input_file = str(pcapng_file)

            # Generate the output CSV file name
            if args.names:
                output_file = os.path.join(output_dir,
                                           args.names[i] if args.names[i].endswith('.csv') else f"{args.names[i]}.csv")
            else:
                output_file = os.path.join(output_dir, os.path.basename(input_file).replace('.pcapng', '.csv'))

            generate_csv(input_file, output_file)


if __name__ == '__main__':
    main()
