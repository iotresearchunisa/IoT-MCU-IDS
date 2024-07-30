#############################################################
# 			               PORT SCANNING			        # #############################################################
#								                        #
#	[2:00 min] normal - attack - [2:00 min normal]  	#
#								                        #													
#########################################################

# TCP Syn port scan
sudo nmap -sS -p- <target>

# TCP connect port scan (Default without root privilege)
nmap -sT -p- <target>

# TCP Null scan
sudo nmap -sN -p- <target>

# TCP FIN scan
sudo nmap -sF -p- <target>

# TCP Xmas scan
sudo nmap -sX -p- <target>

# TCP ACK port scan
sudo nmap -sA -p- <target>

# TCP Window port scan
sudo nmap -sW -p- <target>

# TCP Maimon port scan
sudo nmap -sM -p- <target>

# UDP Scan
sudo nmap -sU -p- <target>

# UDP Scan
unicornscan -m U -Iv <target>:1-65535
