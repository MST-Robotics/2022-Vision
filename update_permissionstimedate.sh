systemctl enable ntp
timedatectl set-ntp 1
sudo /etc/init.d/ntp stop
ntpd -q -g
sudo /etc/init.d/ntp start
chmod 1777 /tmp
