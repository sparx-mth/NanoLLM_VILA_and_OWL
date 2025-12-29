#!/bin/bash

if [ -z "$1" ]; then
  echo "No argument provided! Map to user@192.168.131.22 (AGX1)"
  dsi_num=22
else
  dsi_num="$1"
fi

port_num="60${dsi_num}"
dsi_name="192.168.131.${dsi_num}"
remote_port_num="2222"

echo "Map port ${port_num} to ${dsi_name}:${remote_port_num}"

ssh -L "${port_num}:${dsi_name}:${remote_port_num}" "user@${dsi_name}"