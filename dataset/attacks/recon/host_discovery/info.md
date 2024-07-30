#############################################################
# 			               HOST DISCOVERY			        # #############################################################
#								                        #
#	[2:00 min] normal - attack - [2:00 min normal]  	#
#								                        #													
#########################################################

# Disable port scanning. Host discovery only.
sudo nmap -sn 192.168.14.0/24

# Disable host discovery. Port scan only.
sudo nmap -Pn 192.168.14.0/24

# TCP SYN discovery on port x.Port 80 by default
sudo nmap -PS 192.168.14.0/24

# TCP ACK discovery on port x.Port 80 by default
sudo nmap -PA 192.168.14.0/24

# UDP discovery on port x.Port 40125 by default
sudo nmap -PU 192.168.14.0/24

# ICMP Echo Scan (Ping scan)
sudo nmap -PE 192.168.14.0/24
		
# ICMP Timestamp Scan
sudo nmap -PP 192.168.14.0/24

#ICMP Address Mask Scan
sudo nmap -PM 192.168.14.0/24

# ARP discovery on local network
sudo nmap -PR 192.168.14.0/24
