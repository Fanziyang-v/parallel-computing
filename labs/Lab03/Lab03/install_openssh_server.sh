#!/bin/bash
# 注意此脚本必须以root权限运行

# shellcheck disable=SC2004
if (( ${EUID} != 0 )); then
    echo "[Failed] Please run this script as root"
    exit 1
fi

export ROOT_PASS="user1234"

apt -y install openssh-server
sed -i "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/g" /etc/ssh/sshd_config

echo "root:${ROOT_PASS}"|chpasswd

ipL=$(ip -o -4 addr | awk -F "inet |/" '!/127.0.0.1/ {print $2}' | head -n 1)

hostname ${ipL}

cp /etc/hostname /etc/hostname.bak
echo ${ipL} > /etc/hostname && chmod 644 /etc/hostname

reboot

# ssh-keygen -q -t rsa -N "" -f /root/.ssh/id_rsa


# mpirun --allow-run-as-root --hostfile hostfile --wd /home/lenovo/openmpi/Lab03/Lab03/lab3-code/ --prefix SumArrayCol
# mpirun --hostfile ./hostfile --wd /home/lenovo/openmpi/Lab03/Lab03/lab3-code/ ./SumArrayCol