import socket

server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
server.bind(("68:EC:C5:E1:49:15", 4))
server.listen(1)
print("====bt server open====")
print("wating client...")
client, addr= server.accept()
print("client accepted")
try:
    while True:
        message = input("Enter Message : ")
        client.send(message.encode("utf-8"))
        if message=="q":
            break
    
        """data= client.recv(1024)
        if not data:
            break
        print(f"Message: {data.decode('utf-8')}")"""

except OSError as e:
    print("error dectected")
    pass

client.close()
server.close()