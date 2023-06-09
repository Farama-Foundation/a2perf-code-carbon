"""
Contains implementations of the Public facing API: EmissionsTracker,
OfflineEmissionsTracker and @track_emissions
"""
import dataclasses
import os
import platform
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from functools import wraps
from typing import Callable, List, Optional, Union

import gin
from absl import logging
from ._version import __version__
from .core import cpu, gpu
from .core.config import get_hierarchical_config, parse_gpu_ids
from .core.emissions import Emissions
from .core.units import Energy, Power, Time
from .core.util import count_cpus, suppress
from .external.geography import CloudMetadata, GeoMetadata
from .external.hardware import CPU, GPU, RAM
from .external.logger import logger, set_logger_format, set_logger_level
from .external.scheduler import PeriodicScheduler
from .input import DataSource
from .output import (
    BaseOutput,
    CodeCarbonAPIOutput,
    EmissionsData,
    FileOutput,
    HTTPOutput,
    LoggerOutput,
)


# /!\ Warning: current implementation prevents the user from setting any value to None
# from the script call
# Imagine:
#   1/ emissions_endpoint=localhost:8000 in ~/.codecarbon.config
#   2/ Inside the script, the user cannot disable emissions_endpoint with
#   EmissionsTracker(emissions_endpoint=None) since the config logic will use the one in
#   the config file.
#
# Alternative: EmissionsTracker(emissions_endpoint=False) would work
# TODO: document this
#
# To fix this, a complex move would be to have default values set to the sentinel:
# _sentinel = object()
# see: https://stackoverflow.com/questions/67202314/
#      python-distinguish-default-argument-and-argument-provided-with-default-value


class BaseEmissionsTracker(ABC):
    """
    Primary abstraction with Emissions Tracking functionality.
    Has two abstract methods, `_get_geo_metadata` and `_get_cloud_metadata`
    that are implemented by two concrete classes: `OfflineCarbonTracker`
    and `CarbonTracker.`
    """

    def __init__(
            self,
            project_name: Optional[str] = None,
            measure_power_secs: Optional[int] = None,
            api_call_interval: Optional[int] = None,
            api_endpoint: Optional[str] = None,
            api_key: Optional[str] = None,
            output_dir: Optional[str] = None,
            output_file: Optional[str] = None,
            save_to_file: Optional[bool] = None,
            save_to_api: Optional[bool] = None,
            save_to_logger: Optional[bool] = None,
            logging_logger: Optional[LoggerOutput] = None,
            gpu_ids: Optional[List] = None,
            emissions_endpoint: Optional[str] = None,
            experiment_id: Optional[str] = None,
            co2_signal_api_token: Optional[str] = None,
            tracking_mode: Optional[str] = None,
            log_level: Optional[Union[int, str]] = None,
            on_csv_write: Optional[str] = None,
            logger_preamble: Optional[str] = None,
            default_cpu_power: Optional[int] = None,
    ):

        """
        :param project_name: Project name for current experiment run, default name
                             is "codecarbon".
        :param measure_power_secs: Interval (in seconds) to measure hardware power
                                   usage, defaults to 15.
        :param api_call_interval: Occurrence to wait before calling API :
                            -1 : only call api on flush() and at the end.
                            1 : at every measure
                            2 : every 2 measure, etc...
        :param api_endpoint: Optional URL of Code Carbon API endpoint for sending
                             emissions data.
        :param api_key: API key for Code Carbon API (mandatory!).
        :param output_dir: Directory path to which the experiment details are logged,
                           defaults to current directory.
        :param output_file: Name of the output CSV file, defaults to `emissions.csv`.
        :param save_to_file: Indicates if the emission artifacts should be logged to a
                             file, defaults to True.
        :param save_to_api: Indicates if the emission artifacts should be sent to the
                            CodeCarbon API, defaults to False.
        :param save_to_logger: Indicates if the emission artifacts should be written
                            to a dedicated logger, defaults to False.
        :param logging_logger: LoggerOutput object encapsulating a logging.logger
                            or a Google Cloud logger.
        :param gpu_ids: User-specified known gpu ids to track, defaults to None.
        :param emissions_endpoint: Optional URL of http endpoint for sending emissions
                                   data.
        :param experiment_id: Id of the experiment.
        :param co2_signal_api_token: API token for co2signal.com (requires sign-up for
                                     free beta)
        :param tracking_mode: One of "process" or "machine" in order to measure the
                              power consumption due to the entire machine or to try and
                              isolate the tracked processe's in isolation.
                              Defaults to "machine".
        :param log_level: Global codecarbon log level. Accepts one of:
                            {"debug", "info", "warning", "error", "critical"}.
                          Defaults to "info".
        :param on_csv_write: "append" or "update". Whether to always append a new line
                             to the csv when writing or to update the existing `run_id`
                             row (useful when calling`tracker.flush()` manually).
                             Accepts one of "append" or "update". Default is "append".
        :param logger_preamble: String to systematically include in the logger.
                                messages. Defaults to "".
        :param default_cpu_power: cpu power to be used as default if the cpu is not known
        """

        # logger.info("base tracker init")
        # self._external_conf = get_hierarchical_config()

        self._api_call_interval = api_call_interval
        self._api_endpoint = api_endpoint
        self._co2_signal_api_token = co2_signal_api_token
        self._emissions_endpoint = emissions_endpoint
        self._gpu_ids = gpu_ids
        self._log_level = log_level
        self._measure_power_secs = measure_power_secs
        self._output_dir = output_dir
        self._output_file = output_file
        self._project_name = project_name
        self._save_to_api = save_to_api
        self._save_to_file = save_to_file
        self._save_to_logger = save_to_logger
        self._logging_logger = logging_logger
        self._tracking_mode = tracking_mode
        self._on_csv_write = on_csv_write
        self._logger_preamble = logger_preamble
        self._default_cpu_power = default_cpu_power

        assert self._tracking_mode in ["machine", "process"]
        set_logger_level(self._log_level)
        set_logger_format(self._logger_preamble)
        self._codecarbon_version = __version__
        self._start_time: Optional[float] = None
        self._last_measured_time: float = time.time()
        self._total_energy: Energy = Energy.from_energy(kWh=0)
        self._total_cpu_energy: Energy = Energy.from_energy(kWh=0)
        self._total_gpu_energy: Energy = Energy.from_energy(kWh=0)
        self._total_ram_energy: Energy = Energy.from_energy(kWh=0)
        self._cpu_power: Power = Power.from_watts(watts=0)
        self._gpu_power: Power = Power.from_watts(watts=0)
        self._ram_power: Power = Power.from_watts(watts=0)
        self._cc_api__out = None
        self._measure_occurrence: int = 0
        self._cloud = None
        self._previous_emissions = None
        self._os = platform.platform()
        self._python_version = platform.python_version()
        self._cpu_count = count_cpus()
        self._geo = None

        if isinstance(self._gpu_ids, str):
            self._gpu_ids: List[int] = parse_gpu_ids(self._gpu_ids)
            self._gpu_count = len(self._gpu_ids)

        logger.info("[setup] RAM Tracking...")
        ram = RAM(tracking_mode=self._tracking_mode)
        self._ram_total_size = ram.machine_memory_GB
        self._ram_process = ram.process_memory_GB
        self._hardware: List[Union[RAM, CPU, GPU]] = [ram]

        # Hardware detection
        logger.info("[setup] GPU Tracking...")
        if gpu.is_gpu_details_available():
            logger.info("Tracking Nvidia GPU via pynvml")
            self._hardware.append(GPU.from_utils(self._gpu_ids))
            gpu_names = [n["name"] for n in gpu.get_gpu_static_info()]
            gpu_names_dict = Counter(gpu_names)
            self._gpu_model = "".join(
                [f"{i} x {name}" for name, i in gpu_names_dict.items()]
            )
            self._gpu_count = len(gpu.get_gpu_static_info())
        else:
            logger.info("No GPU found.")
            self._gpu_count = 0
            self._gpu_model = "None"

        logger.info("[setup] CPU Tracking...")
        if cpu.is_powergadget_available():
            logger.info("Tracking Intel CPU via Power Gadget")
            hardware = CPU.from_utils(self._output_dir, "intel_power_gadget")
            self._hardware.append(hardware)
            self._cpu_model = hardware.get_model()
        elif cpu.is_rapl_available():
            logger.info("Tracking Intel CPU via RAPL interface")
            hardware = CPU.from_utils(self._output_dir, "intel_rapl")
            self._hardware.append(hardware)
            self._cpu_model = hardware.get_model()
        else:
            logger.warning("No CPU tracking mode found. Falling back on CPU constant mode.")
            tdp = cpu.TDP()
            power = tdp.tdp
            model = tdp.model
            if (power is None) and self._default_cpu_power:
                # We haven't been able to calculate CPU power but user has input a default one. We use it
                user_input_power = self._default_cpu_power
                logger.debug(f"Using user input TDP: {user_input_power} W")
                power = user_input_power
            logger.info(f"CPU Model on constant consumption mode: {model}")
            self._cpu_model = model
            if power is not None and model is not None:
                hardware = CPU.from_utils(self._output_dir, "constant", model, power)
                self._hardware.append(hardware)
            else:
                logger.warning(
                    "Failed to match CPU TDP constant. Falling back on a global constant."
                )
                hardware = CPU.from_utils(self._output_dir, "constant")
                self._hardware.append(hardware)

        # self._hardware = list(map(lambda x: x.description(), self._hardware))

        logger.info(">>> Tracker's metadata:")
        logger.info(f"  Platform system: {self._os}")
        logger.info(f"  Python version: {self._python_version}")
        logger.info(f"  CodeCarbon version: {self._codecarbon_version}")
        logger.info(f"  Available RAM : {self._ram_total_size:.3f} GB")
        logger.info(f"  CPU count: {self._cpu_count}")
        logger.info(f"  CPU model: {self._cpu_model}")
        logger.info(f"  GPU count: {self._gpu_count}")
        logger.info(f"  GPU model: {self._gpu_model}")

        # Run `self._measure_power` every `measure_power_secs` seconds in a
        # background thread
        self._scheduler = PeriodicScheduler(
            # function=self._measure_power_and_energy,
            function=self.flush,
            interval=self._measure_power_secs,
        )

        self._data_source = DataSource()

        cloud: CloudMetadata = self._get_cloud_metadata()

        if cloud.is_on_private_infra:
            self._geo = self._get_geo_metadata()
            self._longitude = self._geo.longitude
            self._latitude = self._geo.latitude
            self._region = cloud.region
            self._provider = cloud.provider
        else:
            self._region = cloud.region
            self._provider = cloud.provider

        self._emissions: Emissions = Emissions(
            self._data_source, self._co2_signal_api_token
        )
        self.persistence_objs: List[BaseOutput] = list()

        if self._save_to_file:
            self.persistence_objs.append(
                FileOutput(
                    os.path.join(self._output_dir, self._output_file),
                    self._on_csv_write,
                )
            )

        if self._save_to_logger:
            self.persistence_objs.append(self._logging_logger)

        if self._emissions_endpoint:
            self.persistence_objs.append(HTTPOutput(emissions_endpoint))

        # if self._save_to_api:
        #     experiment_id = self._set_from_conf(
        #         experiment_id, "experiment_id", "5b0fa12a-3dd7-45bb-9766-cc326314d9f1"
        #     )
        #     self._cc_api__out = CodeCarbonAPIOutput(
        #         endpoint_url=self._api_endpoint,
        #         experiment_id=experiment_id,
        #         api_key=api_key,
        #         conf=self._conf,
        #     )
        #     self.run_id = self._cc_api__out.run_id
        #     self.persistence_objs.append(self._cc_api__out)
        #
        # else:
        self.run_id = uuid.uuid4()

    @suppress(Exception)
    def start(self) -> None:
        """
        Starts tracking the experiment.
        Currently, Nvidia GPUs are supported.
        :return: None
        """
        if self._start_time is not None:
            logger.warning("Already started tracking")
            return

        self._last_measured_time = self._start_time = time.time()
        # Read initial energy for hardware
        for hardware in self._hardware:
            hardware.start()

        self._scheduler.start()

    @suppress(Exception)
    def flush(self) -> Optional[float]:
        """
        Write the emissions to disk or call the API depending on the configuration,
        but keep running the experiment.
        :return: CO2 emissions in kgs
        """
        if self._start_time is None:
            logger.error("You first need to start the tracker.")
            return None

        # Run to calculate the power used from last
        # scheduled measurement to shutdown
        self._measure_power_and_energy()

        emissions_data = self._prepare_emissions_data(delta=True)
        for persistence in self.persistence_objs:
            if isinstance(persistence, CodeCarbonAPIOutput):
                emissions_data = self._prepare_emissions_data(delta=True)
            persistence.out(emissions_data)

        return emissions_data.emissions

    @suppress(Exception)
    def stop(self) -> Optional[float]:
        """
        Stops tracking the experiment
        :return: CO2 emissions in kgs
        """
        if self._start_time is None:
            logger.error("You first need to start the tracker.")
            return None

        if self._scheduler:
            self._scheduler.stop()
            self._scheduler = None
            # Run to calculate the power used from last
            # scheduled measurement to shutdown
            self._measure_power_and_energy()
        else:
            logger.warning("Tracker already stopped !")

        emissions_data = self._prepare_emissions_data(delta=True)

        for persistence in self.persistence_objs:
            if isinstance(persistence, CodeCarbonAPIOutput):
                emissions_data = self._prepare_emissions_data(delta=True)

            persistence.out(emissions_data)

        self.final_emissions_data = emissions_data
        self.final_emissions = emissions_data.emissions
        return emissions_data.emissions

    def _prepare_emissions_data(self, delta=False) -> EmissionsData:
        """
        :delta: If 'True', return only the delta comsumption since the last call.
        """
        cloud: CloudMetadata = self._get_cloud_metadata()
        duration: Time = Time.from_seconds(time.time() - self._start_time)

        if cloud.is_on_private_infra:
            emissions = self._emissions.get_private_infra_emissions(
                self._total_energy, self._geo
            )  # float: kg co2_eq
            country_name = self._geo.country_name
            country_iso_code = self._geo.country_iso_code
            region = self._geo.region
            on_cloud = "N"
            cloud_provider = ""
            cloud_region = ""
        else:
            emissions = self._emissions.get_cloud_emissions(self._total_energy, cloud)
            country_name = self._emissions.get_cloud_country_name(cloud)
            country_iso_code = self._emissions.get_cloud_country_iso_code(cloud)
            region = self._emissions.get_cloud_geo_region(cloud)
            on_cloud = "Y"
            cloud_provider = cloud.provider
            cloud_region = cloud.region
        total_emissions = EmissionsData(
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            project_name=self._project_name,
            run_id=str(self.run_id),
            duration=duration.seconds,
            emissions=emissions,  # kg
            emissions_rate=emissions / duration.seconds,  # kg/s
            cpu_power=self._cpu_power.W,
            gpu_power=self._gpu_power.W,
            ram_power=self._ram_power.W,
            cpu_energy=self._total_cpu_energy.kWh,
            gpu_energy=self._total_gpu_energy.kWh,
            ram_energy=self._total_ram_energy.kWh,
            energy_consumed=self._total_energy.kWh,
            country_name=country_name,
            country_iso_code=country_iso_code,
            region=self._region,
            on_cloud=on_cloud,
            cloud_provider=self._provider,
            cloud_region=self._region,
            os=self._os,
            python_version=self._python_version,
            codecarbon_version=self._codecarbon_version,
            gpu_count=self._gpu_count,
            gpu_model=self._gpu_model,
            cpu_count=self._cpu_count,
            cpu_model=self._cpu_model,
            longitude=self._longitude,
            latitude=self._latitude,
            ram_total_size=self._ram_total_size,
            tracking_mode=self._tracking_mode,
            ram_process=self._ram_process,
        )

        if delta:
            if self._previous_emissions is None:
                self._previous_emissions = total_emissions
            else:
                # Create a copy
                delta_emissions = dataclasses.replace(total_emissions)
                # Compute emissions rate from delta
                delta_emissions.compute_delta_emission(self._previous_emissions)
                # TODO : find a way to store _previous_emissions only when
                # TODO : the API call succeeded
                self._previous_emissions = total_emissions
                total_emissions = delta_emissions
        logger.debug(total_emissions)
        return total_emissions

    @abstractmethod
    def _get_geo_metadata(self) -> GeoMetadata:
        """
        :return: Metadata containing geographical info
        """
        pass

    @abstractmethod
    def _get_cloud_metadata(self) -> CloudMetadata:
        """
        :return: Metadata containing cloud info
        """
        pass

    def _measure_power_and_energy(self) -> None:
        """
        A function that is periodically run by the `BackgroundScheduler`
        every `self._measure_power_secs` seconds.
        :return: None
        """
        last_duration = time.time() - self._last_measured_time

        warning_duration = self._measure_power_secs * 3
        if last_duration > warning_duration:
            warn_msg = (
                    "Background scheduler didn't run for a long period"
                    + " (%ds), results might be inaccurate"
            )
            logger.warning(warn_msg, last_duration)

        for hardware in self._hardware:
            h_time = time.time()
            # Compute last_duration again for more accuracy
            last_duration = time.time() - self._last_measured_time
            power, energy = hardware.measure_power_and_energy(
                last_duration=last_duration
            )
            self._total_energy += energy
            if isinstance(hardware, CPU):
                self._total_cpu_energy += energy
                self._cpu_power = power
                logger.info(
                    f"Energy consumed for all CPUs : {self._total_cpu_energy.kWh:.6f} kWh"
                    + f". Total CPU Power : {self._cpu_power.W} W"
                )
            elif isinstance(hardware, GPU):
                self._total_gpu_energy += energy
                self._gpu_power = power
                logger.info(
                    f"Energy consumed for all GPUs : {self._total_gpu_energy.kWh:.6f} kWh"
                    + f". Total GPU Power : {self._gpu_power.W} W"
                )
            elif isinstance(hardware, RAM):
                self._total_ram_energy += energy
                self._ram_power = power
                logger.info(
                    f"Energy consumed for RAM : {self._total_ram_energy.kWh:.6f} kWh"
                    + f". RAM Power : {self._ram_power.W} W"
                )
            else:
                logger.error(f"Unknown hardware type: {hardware} ({type(hardware)})")
            h_time = time.time() - h_time
            logger.debug(
                f"{hardware.__class__.__name__} : {hardware.total_power().W:,.2f} "
                + f"W during {last_duration:,.2f} s [measurement time: {h_time:,.4f}]"
            )
        logger.info(
            f"{self._total_energy.kWh:.6f} kWh of electricity used since the beginning."
        )
        self._last_measured_time = time.time()
        self._measure_occurrence += 1
        if self._cc_api__out is not None and self._api_call_interval != -1:
            if self._measure_occurrence >= self._api_call_interval:
                emissions = self._prepare_emissions_data(delta=True)
                logger.info(
                    f"{emissions.emissions_rate * 1000:.6f} g.CO2eq/s mean an estimation of "
                    + f"{emissions.emissions_rate * 3600 * 24 * 365:,} kg.CO2eq/year"
                )
                self._cc_api__out.out(emissions)
                self._measure_occurrence = 0
        logger.debug(f"last_duration={last_duration}\n------------------------")
        # self.flush()  # force write to disk

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.stop()


@gin.configurable
class OfflineEmissionsTracker(BaseEmissionsTracker):
    """
    Offline implementation of the `EmissionsTracker`
    In addition to the standard arguments, the following are required.
    """

    @suppress(Exception)
    def __init__(
            self,
            *args,
            country_iso_code: Optional[str] = None,
            region: Optional[str] = None,
            cloud_provider: Optional[str] = None,
            cloud_region: Optional[str] = None,
            country_2letter_iso_code: Optional[str] = None,
            **kwargs,
    ):
        """
        :param country_iso_code: 3 letter ISO Code of the country where the
                                 experiment is being run
        :param region: The province or region (e.g. California in the US).
                       Currently, this only affects calculations for the United States
                       and Canada
        :param cloud_provider: The cloud provider specified for estimating emissions
                               intensity, defaults to None.
                               See https://github.com/mlco2/codecarbon/
                                        blob/master/codecarbon/data/cloud/impact.csv
                               for a list of cloud providers
        :param cloud_region: The region of the cloud data center, defaults to None.
                             See https://github.com/mlco2/codecarbon/
                                        blob/master/codecarbon/data/cloud/impact.csv
                             for a list of cloud regions.
        :param country_2letter_iso_code: For use with the CO2Signal emissions API.
                                         See http://api.electricitymap.org/v3/zones for
                                         a list of codes and their corresponding
                                         locations.
        """
        self._external_conf = get_hierarchical_config()
        self._cloud_provider = cloud_provider
        self._cloud_region = cloud_region
        self._country_2letter_iso_code = country_2letter_iso_code
        self._country_iso_code = country_iso_code
        self._region = region

        logger.info("offline tracker init")

        if self._region is not None:
            assert isinstance(self._region, str)
            self._region: str = self._region.lower()

        if self._cloud_provider:
            if self._cloud_region is None:
                logger.error(
                    "Cloud Region must be provided " + " if cloud provider is set"
                )

            df = DataSource().get_cloud_emissions_data()
            if (
                    len(
                        df.loc[
                            (df["provider"] == self._cloud_provider)
                            & (df["region"] == self._cloud_region)
                        ]
                    )
                    == 0
            ):
                logger.error(
                    "Cloud Provider/Region "
                    f"{self._cloud_provider} {self._cloud_region} "
                    "not found in cloud emissions data."
                )
        if self._country_iso_code:
            try:
                self._country_name: str = DataSource().get_global_energy_mix_data()[
                    self._country_iso_code
                ]["country_name"]
            except KeyError as e:
                logger.error(
                    "Does not support country"
                    + f" with ISO code {self._country_iso_code} "
                      f"Exception occurred {e}"
                )

        if self._country_2letter_iso_code:
            assert isinstance(self._country_2letter_iso_code, str)
            self._country_2letter_iso_code: str = self._country_2letter_iso_code.upper()

        super().__init__(*args, **kwargs)

    def _get_geo_metadata(self) -> GeoMetadata:
        return GeoMetadata(
            country_iso_code=self._country_iso_code,
            country_name=self._country_name,
            region=self._region,
            country_2letter_iso_code=self._country_2letter_iso_code,
        )

    def _get_cloud_metadata(self) -> CloudMetadata:
        if self._cloud is None:
            self._cloud = CloudMetadata(
                provider=self._cloud_provider, region=self._cloud_region
            )
        return self._cloud


class EmissionsTracker(BaseEmissionsTracker):
    """
    An online emissions tracker that auto infers geographical location,
    using the `geojs` API
    """

    def _get_geo_metadata(self) -> GeoMetadata:
        return GeoMetadata.from_geo_js(self._data_source.geo_js_url)

    def _get_cloud_metadata(self) -> CloudMetadata:
        if self._cloud is None:
            self._cloud = CloudMetadata.from_utils()
        return self._cloud


@gin.configurable
def track_emissions(
        fn: Callable = None,
        project_name: Optional[str] = None,
        measure_power_secs: Optional[float] = None,
        api_call_interval: int = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        save_to_file: Optional[bool] = None,
        save_to_api: Optional[bool] = None,
        save_to_logger: Optional[bool] = None,
        logging_logger: Optional[LoggerOutput] = None,
        offline: Optional[bool] = None,
        emissions_endpoint: Optional[str] = None,
        experiment_id: Optional[str] = None,
        country_iso_code: Optional[str] = None,
        region: Optional[str] = None,
        cloud_provider: Optional[str] = None,
        cloud_region: Optional[str] = None,
        gpu_ids: Optional[List] = None,
        co2_signal_api_token: Optional[str] = None,
        log_level: Optional[Union[int, str]] = None,
        tracking_mode: Optional[str] = None,
        baseline_measure_sec: Optional[float] = None,
):
    """
    Decorator that supports both `EmissionsTracker` and `OfflineEmissionsTracker`
    :param fn: Function to be decorated
    :param project_name: Project name for current experiment run,
                         default name is "codecarbon".
    :param measure_power_secs: Interval (in seconds) to measure hardware power usage,
                               defaults to 15.
    :api_call_interval: Number of measure to make before calling the Code Carbon API.
    :param output_dir: Directory path to which the experiment details are logged,
                       defaults to current directory.
    :param output_file: Name of output CSV file, defaults to `emissions.csv`
    :param save_to_file: Indicates if the emission artifacts should be logged to a file,
                         defaults to True.
    :param save_to_api: Indicates if the emission artifacts should be send to the
                        CodeCarbon API, defaults to False.
    :param save_to_logger: Indicates if the emission artifacts should be written
                        to a dedicated logger, defaults to False.
    :param logging_logger: LoggerOutput object encapsulating a logging.logger
                        or a Google Cloud logger.
    :param offline: Indicates if the tracker should be run in offline mode.
    :param country_iso_code: 3 letter ISO Code of the country where the experiment is
                             being run, required if `offline=True`
    :param region: The provice or region (e.g. California in the US).
                   Currently, this only affects calculations for the United States.
    :param cloud_provider: The cloud provider specified for estimating emissions
                           intensity, defaults to None.
                           See https://github.com/mlco2/codecarbon/
                                            blob/master/codecarbon/data/cloud/impact.csv
                           for a list of cloud providers.
    :param cloud_region: The region of the cloud data center, defaults to None.
                         See https://github.com/mlco2/codecarbon/
                                            blob/master/codecarbon/data/cloud/impact.csv
                         for a list of cloud regions.
    :param gpu_ids: User-specified known gpu ids to track, defaults to None
    :param log_level: Global codecarbon log level. Accepts one of:
                        {"debug", "info", "warning", "error", "critical"}.
                      Defaults to "info".
    :param tracking_mode: Tracking mode for the emissions tracker.
    :param baseline_measure_sec: Interval (in seconds) to measure baseline power usage.

    :return: The decorated function
    """

    def _decorate(fn: Callable):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            fn_result = None

            if (country_iso_code is None) and (cloud_provider is None):
                raise Exception("Needs ISO Code of the Country for Offline mode")
            baseline_output_file = "baseline_emissions.csv"
            baseline_tracker = OfflineEmissionsTracker(
                project_name=f"{project_name}_baseline",
                measure_power_secs=measure_power_secs,
                output_dir=output_dir,
                output_file=baseline_output_file,
                save_to_file=save_to_file,
                save_to_logger=save_to_logger,
                logging_logger=logging_logger,
                country_iso_code=country_iso_code,
                region=region,
                cloud_provider=cloud_provider,
                cloud_region=cloud_region,
                gpu_ids=gpu_ids,
                log_level=log_level,
                co2_signal_api_token=co2_signal_api_token,
                tracking_mode=tracking_mode,
            )

            tracker = OfflineEmissionsTracker(
                project_name=project_name,
                measure_power_secs=measure_power_secs,
                output_dir=output_dir,
                output_file=output_file,
                save_to_file=save_to_file,
                save_to_logger=save_to_logger,
                logging_logger=logging_logger,
                country_iso_code=country_iso_code,
                region=region,
                cloud_provider=cloud_provider,
                cloud_region=cloud_region,
                gpu_ids=gpu_ids,
                log_level=log_level,
                co2_signal_api_token=co2_signal_api_token,
                tracking_mode=tracking_mode,
            )

            # Measure baseline power usage
            logging.info(
                "Measuring baseline power usage for {} seconds...".format(
                    baseline_measure_sec
                )
            )
            if baseline_measure_sec is not None:
                baseline_tracker.start()
                time.sleep(baseline_measure_sec)
                baseline_tracker.stop()
            del baseline_tracker
            logging.info('Starting Code Carbon emissions tracking for participant')
            tracker.start()
            try:
                fn_result = fn(*args, **kwargs)
            finally:
                logger.info(
                    "\nGraceful stopping: collecting and writing information.\n"
                    + "Please wait a few seconds..."
                )
                tracker.stop()
                logger.info("Done!\n")
            return fn_result

        return wrapped_fn

    if fn:
        return _decorate(fn)
    return _decorate
