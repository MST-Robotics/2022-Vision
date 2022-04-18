systemctl enable ntp
timedatectl set-ntp 1
ntpd -q -g
chmod 1777 /tmp
