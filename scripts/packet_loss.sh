#!/bin/bash

# Define the network interface
interface="ra0"

# Define the devices' MAC addresses and their corresponding packet loss rates
devices=(
  "5e:bb:f6:9e:ee:fa 30"
  "66:77:88:99:AA:BB 10"
  "CC:DD:EE:FF:00:11 15"
)

# Remove existing root qdisc and classes
tc qdisc del dev $interface root

# Add a new root qdisc
tc qdisc add dev $interface root handle 1: htb default 30

# Create classes for each device
classid=1
for device in "${devices[@]}"; do
  mac=$(echo $device | awk '{print $1}')
  loss=$(echo $device | awk '{print $2}')
  
  tc class add dev $interface parent 1: classid 1:$classid htb rate 10mbit
  tc qdisc add dev $interface parent 1:$classid netem loss $loss%
  
  classid=$((classid+1))
done

# Add filters to classify traffic based on MAC addresses
filterprio=1
for device in "${devices[@]}"; do
  mac=$(echo $device | awk '{print $1}')
  classid=$(echo $device | awk '{print $2}')
  
  tc filter add dev $interface parent 1: protocol ip prio $filterprio u32 match u32 0 0 flowid 1:$classid
  tc filter add dev $interface parent 1: protocol ip prio $filterprio u32 match u16 0x0800 0xFFFF at -2 match u32 0x${mac//:/} 0xFFFFFFFF at -12 flowid 1:$classid
  
  filterprio=$((filterprio+1))
done