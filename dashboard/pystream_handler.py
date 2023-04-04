from typing import Optional
from datetime import datetime
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('PyStreamHandler\t%(asctime)s\t%(levelname)s\t%(name)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)


MQTT_STREAM_DIR = "./pypredict_data/temp_stream"
MQTT_STREAM_FILE_PATH = "./pypredict_data/temp_stream/temp_stream.csv"
MQTT_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
CSV_TIMESTAMP_FORMAT = "%Y-%m-%d"


class DataStreamHandler:
    @staticmethod
    def _generate_init_stream_csv() -> bool:
        try:
            with open(MQTT_STREAM_FILE_PATH, 'w+') as file_stream:
                file_stream.write("dt,AverageTemperature,count\n")
            return True
        except Exception as e:
            logger.error(f"Failed to create initial stream csv file. More Info: {e}")
            return False

    @staticmethod
    def _get_line_to_write(temp: float = None, timestamp: str = None) -> Optional[pd.DataFrame]:
        if not os.path.exists(MQTT_STREAM_FILE_PATH):
            logger.error("Error occurred while acquiring the stream csv file. A new stream file will be created.")
            DataStreamHandler._generate_init_stream_csv()

        current_streamed_df = pd.read_csv(MQTT_STREAM_FILE_PATH, date_parser=[0])
        current_columns = current_streamed_df.columns
        default_columns = ["dt", "AverageTemperature", "count"]
        for col in default_columns:
            if col not in current_columns:
                logger.error("Invalid column names were found in the current stream csv file.")
                return None

        current_streamed_df.dt = pd.to_datetime(current_streamed_df.dt)
        current_streamed_df.set_index(current_streamed_df.dt, inplace=True)
        current_streamed_df = current_streamed_df.drop(columns=['dt'])

        timestamp_datetime = datetime.strptime(timestamp, MQTT_TIMESTAMP_FORMAT)
        timestamp_month = datetime.strftime(
            timestamp_datetime.replace(day=1, hour=0, minute=0, second=0),
            CSV_TIMESTAMP_FORMAT
        )

        values_to_write = {"dt": timestamp_month, "temp": temp, "count": 1}

        if timestamp_month in current_streamed_df.index:
            previous_count = current_streamed_df.loc[timestamp_month, "count"]
            previous_average = current_streamed_df.loc[timestamp_month, "AverageTemperature"]

            new_count = previous_count + 1
            new_average = previous_average + ((temp - previous_average) / new_count)

            values_to_write["temp"] = new_average
            values_to_write["count"] = int(new_count)

        # # uncomment if returning a single string as the line to write to the csv file is required
        # line_to_write = f"{values_to_write['dt']},{values_to_write['temp']},{values_to_write['count']}\n"

        current_streamed_df.loc[timestamp_month, "count"] = values_to_write["count"]
        current_streamed_df.loc[timestamp_month, "AverageTemperature"] = values_to_write["temp"]

        # sorting by date
        current_streamed_df.sort_index(axis=0, inplace=True)

        return current_streamed_df

    @staticmethod
    def save_stream_to_csv(temp: float = None, timestamp: str = None, humidity: float = None) -> Optional[bool]:
        if not temp or not timestamp:
            return False

        try:
            # # uncomment if writing a string line is required
            # line_to_write = DataStreamHandler._get_line_to_write(temp, timestamp)
            # with open(MQTT_STREAM_FILE_PATH, 'a') as file_stream:
            #     file_stream.write(line_to_write)

            dataframe_to_write = DataStreamHandler._get_line_to_write(temp, timestamp)
            if dataframe_to_write is None:
                return False

            dataframe_to_write: pd.DataFrame = dataframe_to_write.reset_index()
            dataframe_to_write.to_csv(MQTT_STREAM_FILE_PATH, index=False)

            return True
        except FileNotFoundError:
            logger.error("Could not find the temp_stream.csv, failed to save the incoming data stream.")
        except Exception as e:
            logger.error(f"Unknown exception occurred while saving incoming data stream. More Info: {e}")
