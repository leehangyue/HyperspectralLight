import numpy as np
import time
from os.path import join, split, isdir
from os import makedirs
from torch_bearer_api import PJGSpectrometer
from util_pca9685 import PCA9685, list_serial_ports


def time_to_str(time_epoch: float):
    # Convert to struct_time in local time (or use time.gmtime() for UTC)
    struct_time = time.localtime(time_epoch)

    # Format the time (note: time.strftime doesn't support milliseconds)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", struct_time)

    # Add milliseconds (timestamp has fractional seconds)
    milliseconds = int((time_epoch - int(time_epoch)) * 1000)
    time_str_with_ms = f"{time_str}.{milliseconds:03d}"

    # Get timezone (e.g., UTC offset like +0800)
    timezone = time.strftime("%z", struct_time)  # e.g., "+0800" or "-0500"

    # Combine into final string
    formatted_time = f"{time_str_with_ms} UTC{timezone}"
    return formatted_time


def lights_off():
    pca9685 = PCA9685()
    pca9685.i2c_address = int(0x43)
    pca9685.set_channels(channel_flux_ratios=np.zeros(16))
    pca9685.i2c_address = int(0x42)
    pca9685.set_channels(channel_flux_ratios=np.zeros(16))
    pca9685.sender.stop()


def save_calibration_data(spectrometer_port: str, n_channels: int = 30, 
                          set_flux_ratios: list = [1., 0.5, 0.2, 0.1, 0.05],
                          save_folder: str = "channel_calibration_data"):
    """
    获取校准数据并保存
    """
    # 创建光谱仪实例
    spectrometer = PJGSpectrometer(spectrometer_port)

    # 获取波长范围
    wavelength_range = spectrometer.get_wavelength_range()
    print(f"波长范围: {wavelength_range['start']}nm - {wavelength_range['end']}nm")
    
    # 设置曝光模式为自动
    if spectrometer.set_exposure_mode(auto=True):
        print("曝光模式已设置为自动")

    # 设置最大曝光时间
    if spectrometer.set_max_exposure_time(5000000):  # 5秒
        print("最大曝光时间已设置")

    ctrl1_channels = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    ctrl2_channels = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29]

    wavelengths = None

    for ch_idx in range(n_channels):
        lights_off()
        time.sleep(0.2)
        # _ = input(f"请将光谱仪对准通道{ch_idx}，按回车开始测量\n")
        for flux_ratio in set_flux_ratios:
            print(f"正在测量通道{ch_idx}，亮度{flux_ratio*100}%...")
            channel_flux_ratios = np.zeros(n_channels)
            channel_flux_ratios[ch_idx] = flux_ratio

            pca9685 = PCA9685()

            channel_flux_ratios_1 = np.append(channel_flux_ratios[ctrl1_channels], 0.)
            pca9685.i2c_address = int(0x43)
            pca9685.set_channels(channel_flux_ratios=channel_flux_ratios_1)

            channel_flux_ratios_2 = np.append(channel_flux_ratios[ctrl2_channels], 0.)
            pca9685.i2c_address = int(0x42)
            pca9685.set_channels(channel_flux_ratios=channel_flux_ratios_2)

            pca9685.sender.stop()

            print(f"已设置通道{ch_idx}亮度为{flux_ratio*100}%")
            
            time.sleep(2)  # 等待2秒以确保光源稳定

            # 设置曝光时间
            if spectrometer.set_exposure_time(int(0.5 + 50 * 1000 / flux_ratio)):
                print("成功设置曝光时间")

            # 连续采集多帧光谱
            spectra = spectrometer.acquire_continuous_spectra(20)
            print(f"成功采集 {len(spectra)} 帧光谱数据")

            if wavelengths is None:
                wavelengths = np.array(spectra[0]["wavelengths"])
            else:
                assert np.max(np.abs(wavelengths - np.array(spectra[0]["wavelengths"]))) < 1e-3
            spectra_data = np.array([spec["spectrum"] for spec in spectra])
            exposure_time_ms = [spec["exposure_time_us"] * 1e-3 for spec in spectra]
            time_strs = [time_to_str(spec["measure_time_epoch"]) for spec in spectra]

            delim = ","
            titles = ["Sample index"] + [str(smpl_idx + 0) for smpl_idx in range(len(spectra))]
            fname_save = f"ch{ch_idx:02d}_{int(flux_ratio*100+0.5):03d}.csv"
            save_dir = join(split(__file__)[0], save_folder)
            if not isdir(save_dir):
                makedirs(save_dir)
            fname_save = join(save_dir, fname_save)
            col0 = ["Measure time", "Exposure time [ms]"] + wavelengths.tolist()
            X = np.vstack([col0, np.vstack([time_strs, exposure_time_ms, spectra_data.T]).T]).T
            np.savetxt(fname=fname_save, X=X, header=delim.join(titles), delimiter=delim, 
                       comments="", encoding="utf-8", fmt="%s")
            print(f"已保存通道{ch_idx}亮度{flux_ratio*100}%的光谱数据到 {fname_save}")

            if np.max(spectra_data) == 0.:
                break  # if the channel does not shine at all, skip all remaining dim levels


def main():
    lights_off()
    try:
        save_calibration_data("/dev/cu.usbmodem57190190851")
    finally:
        lights_off()


if __name__ == "__main__":
    print("\n".join(list_serial_ports()))  # Uncomment to list serial ports if needed
    main()
