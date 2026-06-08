@echo off
mode > ports_list.txt 2>&1
echo --- >> ports_list.txt
powershell -Command "[System.IO.Ports.SerialPort]::GetPortNames()" >> ports_list.txt 2>&1
echo --- >> ports_list.txt
wmic path Win32_SerialPort get DeviceID,Description >> ports_list.txt 2>&1
