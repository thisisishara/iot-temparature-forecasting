import signal
import sqlite3
import sensor as DHT22
import time

conn = sqlite3.connect('/home/pi/2022_19/sensor.db')


conn.execute('''CREATE TABLE IF NOT EXISTS `SENSOR_DATA`
         (
        `DATA_TYPE` TEXT PRIMARY KEY NOT NULL,
        `VALUE` REAL NOT NULL
        );''')


conn.execute("INSERT OR IGNORE INTO `SENSOR_DATA` (`DATA_TYPE`,`VALUE`) VALUES( 'TEMPERATURE',0.0);")
conn.execute("INSERT OR IGNORE INTO `SENSOR_DATA` (`DATA_TYPE`,`VALUE`) VALUES( 'HUMIDITY',0.0);")



class DB_service:
    killer = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_servivice)
        signal.signal(signal.SIGTERM, self.exit_servivice)

    def exit_servivice(self, *args):
        self.killer = True
        exit(0)


if __name__ == '__main__':
    instance = DB_service()
    while not instance.killer:
        try:
            try:
                temperature = DHT22.get_temperature_c()
                humidity = DHT22.get_humidity()
                print(temperature,humidity)
                conn.execute("UPDATE OR IGNORE SENSOR_DATA SET VALUE = "+str(temperature)+" WHERE DATA_TYPE = 'TEMPERATURE';")
                conn.execute("UPDATE OR IGNORE SENSOR_DATA SET VALUE = "+str(humidity)+" WHERE DATA_TYPE = 'HUMIDITY';")
                conn.commit()
            except KeyboardInterrupt:
                print("Terminated by user")
                exit(0)

            time.sleep(1)
        except Exception as e:
            print("skipped reading")
            continue


conn.close()
