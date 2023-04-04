# sudo apt-get install build-essential python-dev
# sudo pip3 install adafruit-circuitpython-dht
import time
import board
import adafruit_dht

# Initial the dht device, with data pin connected to:

def get_temperature_c():
    dhtDevice = adafruit_dht.DHT22(board.D2)
    temperature_c = 0.00
    try:
        # Print the values to the serial port
        temperature_c = round(float(dhtDevice.temperature),2)

    except Exception:
        dhtDevice.exit()
        raise Exception
    
    dhtDevice.exit()
    return temperature_c 

def get_temperature_f():
    dhtDevice = adafruit_dht.DHT22(board.D2)
    temperature_f = 0.00
    try:
        # Print the values to the serial port
        temperature_c = dhtDevice.temperature
        temperature_f = round(float(temperature_c * (9 / 5) + 32),2)

    except Exception:
        dhtDevice.exit()
        raise Exception

    dhtDevice.exit()
    return temperature_f

def get_humidity():
    humidity = 0.00
    dhtDevice = adafruit_dht.DHT22(board.D2)
    try:
        # Print the values to the serial port
        humidity = dhtDevice.humidity
        humidity = round(float(humidity),2)

    except Exception:
        dhtDevice.exit()
        raise Exception
        
    dhtDevice.exit()
    return humidity