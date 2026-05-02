import serial
import time

PORT = "/dev/ttyACM0"

# Common Feetech bauds
BAUDS = [1000000,115200]

def ping(ser, motor_id):
    packet = bytearray([
        0xFF, 0xFF,
        motor_id,
        2,
        0x01  # PING
    ])
    checksum = (~sum(packet[2:])) & 0xFF
    packet.append(checksum)

    ser.write(packet)
    time.sleep(0.02)

    return ser.read(6)

for baud in BAUDS:
    print(f"\n🔍 Testing baud: {baud}")
    try:
        ser = serial.Serial(PORT, baud, timeout=0.1)
        time.sleep(1)

        for motor_id in range(1, 10):
            resp = ping(ser, motor_id)
            if resp:
                print(f"✅ Found motor ID {motor_id} at baud {baud}")

        ser.close()

    except Exception as e:
        print("Error:", e)