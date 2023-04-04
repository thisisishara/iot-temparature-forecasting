import json
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pmdarima.arima.arima import ARIMA
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import joblib
import glob
import os
import re

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('PyARIMA\t%(asctime)s\t%(levelname)s\t%(name)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
TIMESTAMP_DATE_ONLY_FORMAT = "%Y%m%d"
MODELS_DIR = "./pypredict_data/models"
DATASET_DIR = "./pypredict_data/datasets"
DEFAULT_DATASET_PATH = "./pypredict_data/datasets/temp.csv"
DEFAULT_CLEANED_DATASET_PATH = "./pypredict_data/datasets/temp_cleaned.csv"
DEFAULT_LATEST_DATASET_PATH = "./pypredict_data/datasets/temp_cleaned_latest.csv"
STREAM_DIR = "./pypredict_data/temp_stream"
STREAM_FILE_PATH = "./pypredict_data/temp_stream/temp_stream.csv"
MODEL_NAME_PATTERN = re.compile("^([0-9]{8})_([0-9]{6}).pkl$")
TIMESTAMP_FORMAT_PATTERN = re.compile("^([0-9]{8})_([0-9]{6})$")


class PredictionStartPoint:
    MODEL_END = 1
    UPTODATE = 2


class PredictionsType:
    RAW = 1
    JSON = 2


class IOTArimaTemperature:
    @staticmethod
    def get_monthly_average_temp(df: pd.DataFrame = None) -> Optional[list]:
        monthly_temp_avg_list = list()
        for i in range(12):
            m_avg = df[pd.DatetimeIndex(df.dt).month == (i + 1)].mean(numeric_only=True)[0]
            monthly_temp_avg_list.append(m_avg)
        return monthly_temp_avg_list

    @staticmethod
    def temperature_fill_na_monthly_average(df: pd.DataFrame = None) -> Optional[pd.Series]:
        monthly_temp_avg_list = IOTArimaTemperature.get_monthly_average_temp(df)

        for i in range(12):
            df.loc[
                (df.AverageTemperature.isna()) & (pd.DatetimeIndex(df.dt).month == (i + 1)), "AverageTemperature"] = \
                monthly_temp_avg_list[i]

        df.set_index(df.dt, inplace=True)
        df = df.drop(columns=['dt'])
        return df.AverageTemperature

    @staticmethod
    def is_stationary_ADF(df: pd.Series = None, alpha: float = 0.05) -> bool:
        if df is None:
            return False

        if not isinstance(df, pd.Series):
            return False

        vals = df.values
        result = adfuller(vals)
        # adf_stat = result[0]
        # critical_vals = result[4]
        p_val = result[1]

        if p_val > alpha:
            return False
        return True

    @staticmethod
    def train_initial_model() -> Optional[str]:
        df = pd.read_csv(DEFAULT_DATASET_PATH, date_parser=[0])
        df.dt = pd.to_datetime(df.dt)
        df_clean = IOTArimaTemperature.temperature_fill_na_monthly_average(df)

        # terminate if not stationary
        if not IOTArimaTemperature.is_stationary_ADF(df_clean):
            logger.error("Dataset is not stationary. A stationary dataset is expected.")
            return

        # # Uncomment to see graphs
        # df_clean.plot()
        # plt.show()
        # plot_acf(df_clean);
        # plot_pacf(df_clean, method='ywm');

        seasonal_model = pm.auto_arima(
            df_clean,
            start_p=0, start_q=0,
            test='adf',
            max_p=4, max_q=4,
            m=12,
            start_P=0, seasonal=True,
            d=None, D=1, trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        logger.info(f"Seasonal ARIMA model summary: {seasonal_model.summary()}")

        # saving the initial model
        model_last_timestamp: pd.Timestamp = df_clean.index[-1]
        model_name = IOTArimaTemperature.generate_model_name_from_date(model_last_timestamp)
        model_path = os.path.join(MODELS_DIR, model_name)

        if os.path.exists(model_path):
            logger.warning("Found an existing model with the same timestamp. The new model will replace the old model.")
            os.remove(model_path)

        if os.path.exists(DEFAULT_LATEST_DATASET_PATH):
            logger.warning("Cleaning up the datasets directory.")
            os.remove(DEFAULT_LATEST_DATASET_PATH)

        all_model_paths = [path for path in glob.glob(os.path.join(MODELS_DIR, '*.pkl'))]
        if all_model_paths:
            logger.warning("Cleaning up the models directory.")
        for path in all_model_paths:
            if os.path.exists(path):
                os.remove(path)

        joblib.dump(seasonal_model, model_path, compress=True)

        # saving the cleaned dataset for future reference
        df_to_save = df_clean.copy()
        df_to_save: pd.DataFrame = df_to_save.reset_index()
        df_to_save.to_csv(os.path.join(DATASET_DIR, "temp_cleaned.csv"), index=False)
        df_to_save.to_csv(os.path.join(DATASET_DIR, "temp_cleaned_latest.csv"), index=False)

        return model_path

    @staticmethod
    def get_date_from_model_name(model_name: str = None) -> Optional[datetime]:
        if not MODEL_NAME_PATTERN.match(model_name):
            return None

        model_name_date_str = re.sub(".pkl", "", model_name)
        if not TIMESTAMP_FORMAT_PATTERN.match(model_name_date_str):
            return None

        date = datetime.strptime(model_name_date_str, TIMESTAMP_FORMAT)
        return date

    @staticmethod
    def fill_na_using_latest_monthly_average(df_to_fill: pd.DataFrame = None) -> Optional[pd.Series]:
        try:
            df_latest = pd.read_csv(DEFAULT_LATEST_DATASET_PATH, date_parser=[0])
        except Exception as e:
            logger.error("Could not find the latest cleaned temp dataset. Loading the initial dataset to calculate"
                         f" the monthly averages. More Info: {e}")
            df_latest = pd.read_csv(DEFAULT_DATASET_PATH, date_parser=[0])

        df_latest.dt = pd.to_datetime(df_latest.dt)
        monthly_temp_avg_latest = IOTArimaTemperature.get_monthly_average_temp(df_latest)

        for month in range(12):
            df_to_fill.loc[
                (df_to_fill.AverageTemperature.isna()) & (pd.DatetimeIndex(df_to_fill.dt).month == (month + 1)),
                "AverageTemperature"] = monthly_temp_avg_latest[month]

        df_to_fill.set_index(df_to_fill.dt, inplace=True)
        df = df_to_fill.drop(columns=['dt'])
        return df.AverageTemperature

    @staticmethod
    def update_model(new_data: pd.Series = None, model_name: str = None) -> Optional[str]:
        if not model_name:
            model_name = IOTArimaTemperature.get_latest_model_name()
            logger.warning(f"A model name has not been specified. Updating the latest model: {model_name}")
        else:
            list_of_models = IOTArimaTemperature.get_all_existing_model_names()
            if model_name not in list_of_models:
                logger.error("The given model does not exist. Please retry with a valid model name.")
                return None
        latest_valid_model_name = IOTArimaTemperature.generate_latest_model_name()

        if model_name == latest_valid_model_name:
            logger.error("The ARIMA model is already up-to-date.")
            return None

        if new_data is None:
            logger.error("An explicit dataset has not been specified. Latest streaming dataset will be used "
                         "to update the specified model.")
            new_data_df = pd.read_csv(STREAM_FILE_PATH, date_parser=[0])
            new_data_df.dt = pd.to_datetime(new_data_df.dt)
            new_data_df.set_index(new_data_df.dt, inplace=True)
            new_data = new_data_df.AverageTemperature

        if new_data is None or len(new_data) <= 0:
            logger.error("The provided series is empty. Updating process terminated.")
            return None

        # sorting the data series by date
        new_data.sort_index(axis=0, inplace=True)

        model_end_date = IOTArimaTemperature.get_date_from_model_name(model_name)
        new_start_index = model_end_date + pd.DateOffset(months=1)
        df_start_index = new_data.index[0]
        df_end_index = new_data.index[-1]

        if not isinstance(df_start_index, datetime) or not isinstance(df_end_index, datetime):
            logger.error("Invalid indexes in the new dataset. Index values should be given in datetime format.")
            return None

        df_end_boundary = \
            datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + pd.DateOffset(months=-1)

        for idx in new_data.index[::-1]:
            if idx <= df_end_boundary:
                df_end_index = idx
                break

        if df_end_index <= model_end_date:
            logger.warning("Model contains up-to-date data temperature readings.")
            return None

        # removing the overlap
        if df_start_index <= model_end_date:
            logger.warning("New dataset overlaps with the model. Overlapping data will be discarded.")
            new_data = new_data[new_start_index:]

        # Adjusting the end index
        new_data = new_data[:df_end_index]

        # filling the gaps
        if df_start_index > new_start_index:
            logger.warning("There is a gap between the new dataset and the last model update date. "
                           "The gap will be filled using the monthly average temperature values prior to model update.")
            # gap_end_date = df_start_index + pd.DateOffset(months=-1)
            new_indexes = pd.date_range(start=new_start_index, end=df_end_index, freq='MS')
            new_data_df = pd.DataFrame(np.nan, index=new_indexes, columns=['AverageTemperature'])

            for date_index in new_data.index:
                new_data_df.loc[date_index]['AverageTemperature'] = new_data[date_index]

            new_data_df.index.rename("dt", inplace=True)
            new_data_df = new_data_df.reset_index()

            new_data_cleaned = IOTArimaTemperature.fill_na_using_latest_monthly_average(new_data_df)
            new_data = new_data_cleaned.copy()

        model: ARIMA = IOTArimaTemperature.get_model_by_name(model_name)

        if not model:
            logger.error(f"Could not load the model: {model_name}")
            return None

        updated_model = model.update(new_data)

        # saving the updated model
        model_date: pd.Timestamp = new_data.index[-1]
        updated_model_name = IOTArimaTemperature.generate_model_name_from_date(model_date)
        updated_model_path = os.path.join(MODELS_DIR, updated_model_name)

        if os.path.exists(updated_model_path):
            logger.warning("Found an existing model with the same timestamp. The new model will replace the old model.")
            os.remove(updated_model_path)

        joblib.dump(updated_model, updated_model_path, compress=True)

        # saving the cleaned dataset for future reference
        new_data_copy = new_data.copy()
        previous_dataset = pd.read_csv(DEFAULT_LATEST_DATASET_PATH, date_parser=[0])
        previous_dataset.dt = pd.to_datetime(previous_dataset.dt)
        previous_dataset.set_index(previous_dataset.dt, inplace=True)
        previous_dataset = previous_dataset.AverageTemperature
        df_to_save = previous_dataset.append(new_data_copy)
        df_to_save.sort_index(axis=0, inplace=True)
        df_to_save: pd.DataFrame = df_to_save.reset_index()
        df_to_save.to_csv(os.path.join(DATASET_DIR, "temp_cleaned_latest.csv"), index=False)

        logger.info(f"Model {model_name} updated successfully. The updated model is {updated_model_name}")
        return updated_model_path

    @staticmethod
    def generate_latest_model_name() -> str:
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        previous_month = current_month + pd.DateOffset(months=-1)
        return previous_month.strftime(TIMESTAMP_FORMAT) + ".pkl"

    @staticmethod
    def generate_model_name_from_date(date: pd.Timestamp = None) -> Optional[str]:
        if date is None:
            return None
        return date.to_pydatetime().replace(day=1, hour=0, minute=0, second=0).strftime(TIMESTAMP_FORMAT) + ".pkl"

    @staticmethod
    def get_latest_model() -> Optional[ARIMA]:
        latest_model_name = IOTArimaTemperature.get_latest_model_name()
        if latest_model_name:
            latest_model_path = os.path.join(MODELS_DIR, latest_model_name)
            latest_model = joblib.load(latest_model_path)
            return latest_model
        return None

    @staticmethod
    def get_model_by_name(model_name: str = None) -> Optional[ARIMA]:
        if not model_name:
            return None
        model_path = os.path.join(MODELS_DIR, model_name)
        model = joblib.load(model_path)
        return model

    @staticmethod
    def get_model_by_path(model_path: str = None) -> Optional[ARIMA]:
        if not model_path:
            return None
        if not os.path.exists(model_path):
            return None
        model = joblib.load(model_path)
        return model

    @staticmethod
    def get_latest_model_name() -> Optional[str]:
        all_model_names = [os.path.split(path)[-1] for path in glob.glob(os.path.join(MODELS_DIR, '*.pkl'))]
        # all_model_paths = glob.glob(os.path.join(MODELS_DIR, '*.pkl'))

        if not all_model_names:
            return None
        valid_model_names = sorted([name for name in all_model_names if MODEL_NAME_PATTERN.match(name)])
        if not valid_model_names:
            return None
        return valid_model_names[-1]

    @staticmethod
    def get_all_existing_model_names() -> Optional[list]:
        all_models = [path.split("\\")[-1] for path in glob.glob(os.path.join(MODELS_DIR, '*.pkl'))]
        if not all_models:
            return None
        valid_model_names = sorted([name for name in all_models if MODEL_NAME_PATTERN.match(name)])
        return valid_model_names

    @staticmethod
    def get_predictions_from_latest_model(
            n_past: int = 12,
            n_future: int = 12,
            mid_point: int = PredictionStartPoint.UPTODATE,
            pred_type: int = PredictionsType.RAW,
    ) -> Optional[list]:
        n_past_original = n_past
        n_future_original = n_future

        latest_model_name = IOTArimaTemperature.get_latest_model_name()
        model_end_date = IOTArimaTemperature.get_date_from_model_name(latest_model_name)
        latest_df = pd.read_csv(DEFAULT_LATEST_DATASET_PATH, date_parser=[0])

        if mid_point == PredictionStartPoint.UPTODATE:
            mid_point_date = \
                datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) + pd.DateOffset(months=-1)

            if model_end_date < mid_point_date:
                n_gap = int((mid_point_date - model_end_date) / np.timedelta64(1, 'M'))
                if n_past > n_gap > 0:
                    n_past = n_past - n_gap
                    n_future = n_future + n_gap
                elif n_gap > n_past:
                    n_past = 0
                    n_future = n_future + n_gap

        if n_past != n_past_original:
            logger.warning("The latest model is not up-to-date. There is a gap between the last model update and "
                           "the mid-point in consideration for the predictions. You will only get future predictions "
                           "however the accuracy might be off due to using a non up-to-date model.")

        if n_past > 0:
            past_pred_df = latest_df[latest_df.index[-1 * n_past]:].copy()
            past_pred_df.reset_index(inplace=True)
            past_pred_df.dt = pd.to_datetime(past_pred_df.dt)
            past_pred_df.set_index(past_pred_df.dt, inplace=True)
            past_pred = past_pred_df.AverageTemperature
        else:
            past_pred = pd.Series(dtype='float64')

        latest_model = IOTArimaTemperature.get_model_by_name(latest_model_name)
        future_pred = latest_model.predict(n_periods=n_future)
        future_pred_start_index = model_end_date + pd.DateOffset(months=1)
        future_pred_index = pd.date_range(start=future_pred_start_index, periods=n_future, freq='MS')
        future_pred = pd.Series(future_pred, index=future_pred_index, name="AverageTemperature")
        future_pred = future_pred[-1 * (n_past_original + n_future_original - n_past):]

        if pred_type == PredictionsType.JSON:
            pred_dict = dict()
            past_pred_list = list()
            future_pred_list = list()

            if n_past != 0:
                for past_idx in past_pred.index:
                    past_pred_list.append({"x": past_idx.strftime('%Y-%m-%d %H:%M:%S'), "y": past_pred[past_idx]})

            for future_idx in future_pred.index:
                future_pred_list.append({"x": future_idx.strftime('%Y-%m-%d %H:%M:%S'), "y": future_pred[future_idx]})

            pred_dict["series"] = ["past", "future"]
            pred_dict["data"] = [past_pred_list, future_pred_list]
            pred_dict["labels"] = [""]
            return [json.dumps(pred_dict)]
        return [past_pred, future_pred]

    @staticmethod
    def get_predictions_from_init_model(
            n_past: int = 12,
            n_future: int = 12,
            pred_type: int = PredictionsType.RAW,
    ) -> Optional[list]:
        init_model = IOTArimaTemperature.get_model_by_name("20130901_000000.pkl")
        init_df = pd.read_csv(DEFAULT_CLEANED_DATASET_PATH, date_parser=[0])

        past_pred_df = init_df[init_df.index[-1 * n_past]:].copy()
        past_pred_df.reset_index(inplace=True)
        past_pred_df.dt = pd.to_datetime(past_pred_df.dt)
        past_pred_df.set_index(past_pred_df.dt, inplace=True)
        past_pred = past_pred_df.AverageTemperature

        future_pred = init_model.predict(n_periods=n_future)
        model_end_date = IOTArimaTemperature.get_date_from_model_name("20130901_000000.pkl")
        future_pred_start_index = model_end_date + pd.DateOffset(months=1)
        future_pred_index = pd.date_range(start=future_pred_start_index, periods=n_future, freq='MS')
        future_pred = pd.Series(future_pred, index=future_pred_index, name="AverageTemperature")

        if pred_type == PredictionsType.JSON:
            pred_dict = dict()
            past_pred_list = list()
            future_pred_list = list()

            for past_idx in past_pred.index:
                past_pred_list.append({"x": past_idx.strftime('%Y-%m-%d %H:%M:%S'), "y": past_pred[past_idx]})

            for future_idx in future_pred.index:
                future_pred_list.append({"x": future_idx.strftime('%Y-%m-%d %H:%M:%S'), "y": future_pred[future_idx]})

            pred_dict["series"] = ["past", "future"]
            pred_dict["data"] = [past_pred_list, future_pred_list]
            pred_dict["labels"] = [""]
            return [json.dumps(pred_dict)]
        return [past_pred, future_pred]


if __name__ == "__main__":
    logger.info("PyARIMA script is executed directly from source.")

    # # PREDICTIONS FROM THE LATEST MODEL AVAILABLE
    # print(IOTArimaTemperature.get_latest_model().predict(n_periods=5))

    # # TRAIN THE INITIAL MODEL WITH A CUSTOM DATASET
    # print(IOTArimaTemperature.train_new_model("./pypredict_data/datasets/temp.csv"))

    # # TRAIN THE INITIAL MODEL
    # path_ = IOTArimaTemperature.train_initial_model()
    # print(IOTArimaTemperature.get_latest_model_name(), path_)

    # # UPDATE THE LATEST MODEL WITH CUSTOM DATASET
    # new_idx = pd.date_range(start='2022-01-01', periods=4, freq='MS')
    # path_ = IOTArimaTemperature.update_model(pd.Series([100, 220.667, None, 3100.332], index=new_idx))
    # mod = IOTArimaTemperature.get_model_by_path(path_)
    # pred = mod.predict(n_periods=12)
    # print(pred)

    # # UPDATING THE LATEST MODEL
    # path_ = IOTArimaTemperature.update_model()
    # print(IOTArimaTemperature.get_latest_model_name(), path_)

    # # GET PREDICTIONS FROM A SPECIFIC MODEL
    # old_m = IOTArimaTemperature.get_model_by_name("20130901_000000.pkl")
    # old_p = old_m.predict(n_periods=12)
    # print(old_p)

    # # GET PREDICTIONS FROM THE INIT MODEL
    # pred = IOTArimaTemperature.get_predictions_from_init_model(pred_type=PredictionsType.RAW)
    # print("Temperature in the past 12 months from model end date:")
    # print(pred[0])
    # print("Temperature predictions for 12 months from model end date:")
    # print(pred[1])

    # # GET RAW PREDICTIONS FROM THE LATEST MODEL AVAILABLE
    # pred = IOTArimaTemperature.get_predictions_from_latest_model(
    #     pred_type=PredictionsType.RAW,
    #     mid_point=PredictionStartPoint.UPTODATE,
    # )
    # print("Temperature in the past 12 months from previous month:")
    # print(pred[0])
    # print("Temperature predictions for 12 months from previous month:")
    # print(pred[1])

    # # GET JSON PREDICTIONS FROM THE LATEST MODEL AVAILABLE
    # pred = IOTArimaTemperature.get_predictions_from_latest_model(
    #     pred_type=PredictionsType.JSON,
    #     mid_point=PredictionStartPoint.UPTODATE,
    # )
    # print("Predicted Values")
    # print(pred[0])
    pass
