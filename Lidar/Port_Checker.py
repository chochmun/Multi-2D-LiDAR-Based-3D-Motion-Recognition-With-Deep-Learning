import serial.tools.list_ports

def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    port_names = []

    for port in ports:
        port_names.append(port.device)
    
    return port_names

if __name__ == "__main__":
   
    """for idx, port_name in enumerate(port_names, start=1):
        print(f"port{idx} = '{port_name}'")"""
