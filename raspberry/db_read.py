import sqlite3

def get_db_temperature_c():
    TEMPERATURE = 0.0
    try:
        conn = sqlite3.connect('/home/pi/2022_19/sensor.db')
        cur = conn.cursor()
        cur.execute("SELECT DATA_TYPE, VALUE FROM 'SENSOR_DATA';")
        data = cur.fetchall()
        print(data)
        TEMPERATURE = data[0][1]
        HUMIDITY = data[1][1]
        print(TEMPERATURE, HUMIDITY)
        conn.close()
    except Exception as e:
        print("skipped reading",e)
    return TEMPERATURE 

def get_db_humidity():
    HUMIDITY = 0.0
    try:
        conn = sqlite3.connect('/home/pi/2022_19/sensor.db')
        cur = conn.cursor()
        cur.execute("SELECT DATA_TYPE, VALUE FROM 'SENSOR_DATA';")
        data = cur.fetchall()
        HUMIDITY = data[1][1]
        conn.close()
    except Exception as e:
        print("skipped reading",e)
    return HUMIDITY 