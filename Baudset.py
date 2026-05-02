import serial
import time

PORT = "/dev/ttyACM0"   # leader port
CURRENT_BAUD = 115200

ser = serial.Serial(PORT, CURRENT_BAUD, timeout=0.1)
time.sleep(1)

def write_packet(motor_id, addr, value):
    packet = bytearray([
        0xFF, 0xFF,
        motor_id,
        4,
        0x03,      # WRITE
        addr,
        value
    ])
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)
    ser.write(packet)
    time.sleep(0.05)

def ping(motor_id):
    packet = bytearray([
        0xFF, 0xFF,
        motor_id,
        2,
        0x01
    ])
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)
    ser.write(packet)
    time.sleep(0.02)
    return ser.read(6)

for i in range(1, 7):
    resp = ping(i)
    if resp:
        print(f"✅ Motor {i} detected, setting baud → 1M")
        write_packet(i, 0x06, 0x00)  # 0x00 = 1M
    else:
        print(f"❌ Motor {i} NOT responding at 115200")

print("Done. Now re-check at 1M.")