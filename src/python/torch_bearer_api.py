import serial
import struct
import time
from typing import Dict, List, Tuple, Union

class PJGSpectrometer:
    """
    PJG光谱仪API类
    串口设置: 115200 bps, 8数据位, 1停止位, 无校验, 无流控制
    """
    
    # 常量定义
    CMD_HEADER = bytes.fromhex("CC01")
    RESP_HEADER = bytes.fromhex("CC81")
    END_MARKER = bytes.fromhex("0D0A")
    
    # 命令类型映射
    CMD_TYPES = {
        "get_wavelength_range": 0x0F,
        "single_spectrum": 0x32,
        "start_continuous": 0x33,
        "stop_continuous": 0x04,
        "get_device_info": 0x08,
        "set_exposure_mode": 0x0A,
        "get_exposure_mode": 0x0B,
        "set_exposure_time": 0x0C,
        "get_exposure_time": 0x0D,
        "set_max_exposure": 0x13,
        "get_max_exposure": 0x14,
        "set_cie_mode": 0x36,
        "get_cie_mode": 0x37,
        "start_efficiency_curve": 0x23,
        "validate_efficiency_curve": 0x27,
        "reset_efficiency_curve": 0x25,
    }
    
    # CIE模式映射
    CIE_MODES = {
        0x00: "CIE1931_2deg",
        0x01: "CIE1964_10deg",
        0x02: "CIE2015_2deg",
        0x03: "CIE2015_10deg",
    }
    
    # 曝光状态映射
    EXPOSURE_STATUS = {
        0x00: "正常",
        0x01: "过曝",
        0x02: "欠曝",
    }
    
    # 光度学参数名称
    PHOTOMETRIC_PARAMS = [
        "X", "Y", "Z", "x", "y", "u", "v", "u'", "v'", "CCT", "Nit", 
        "r_ratio", "g_ratio", "b_ratio", "DUV", "Ra", "R1", "R2", "R3", 
        "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", 
        "R14", "R15", "Lp", "HW", "Ld", "purity", "SP", "SDCM", "k", 
        "lux", "Ee", "fc", "CQS", "GAI_EES", "GAI_BB_8", "GAI_BB_15", 
        "EML", "M_EDI"
    ]
    
    # 近红外参数名称
    NIR_PARAMS = ["Red_Ee", "Nir_EeA", "Nir_EeB"]
    
    def __init__(self, port: str):
        """
        初始化光谱仪连接
        :param port: 串口号 (如 'COM3' 或 '/dev/ttyUSB0')
        """
        self.ser = serial.Serial(
            port=port,
            baudrate=921600,  # 波特率
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=2  # 读取超时时间(秒)
        )
    
    def _send_command(self, cmd_type: int, data: bytes = b"", no_response: bool = False) -> bytes:
        """
        发送命令并接收响应(内部方法)
        :param cmd_type: 命令类型字节
        :param data: 命令数据
        :param no_response: 是否预期没有响应数据
        :return: 响应数据包
        """
        # 构建完整命令包
        total_length = 6 + len(data) + 1 + 2  # 头2B + 长度3B + 类型1B + 数据 + 校验1B + 结束2B
        length_bytes = struct.pack("<I", total_length)[:3]  # 小端序3字节
        
        # 计算校验和(不包括结束标识)
        cmd_data = self.CMD_HEADER + length_bytes + bytes([cmd_type]) + data
        checksum = sum(cmd_data) & 0xFF
        
        # 完整命令包
        full_cmd = cmd_data + bytes([checksum]) + self.END_MARKER
        self.ser.flush()  # 清空串口缓冲区
        self.ser.write(full_cmd)
        
        if no_response:
            return b""

        # 读取响应头
        header = self.ser.read(2)
        if header != self.RESP_HEADER:
            raise ValueError("无效的响应头")
        
        # 读取包长度(小端3字节)
        len_bytes = self.ser.read(3)
        total_len = struct.unpack("<I", len_bytes + b"\x00")[0]
        
        # 读取剩余数据
        remaining_len = total_len - 5  # 减去已读的5字节(头2B+长度3B)
        if remaining_len < 4:  # 至少应有类型1B+校验1B+结束2B
            raise ValueError("无效的包长度")
        
        response_data = self.ser.read(remaining_len)
        
        # 验证结束标识
        if response_data[-2:] != self.END_MARKER:
            raise ValueError("无效的结束标识")
        
        # 验证校验和
        received_checksum = response_data[-3]
        calculated_checksum = (sum(self.RESP_HEADER) + 
                               sum(len_bytes) + 
                               sum(response_data[:-3])) & 0xFF
        
        if received_checksum != calculated_checksum:
            raise ValueError("校验和错误")
        
        # 返回有效数据(不包括类型、校验和和结束标识)
        return response_data[1:-3]  # 跳过类型字节，去掉校验和和结束标识
    
    def _parse_float_data(self, data: bytes, count: int) -> List[float]:
        """解析浮点数数组"""
        return list(struct.unpack(f"<{count}f", data[:4*count]))
    
    def _parse_uint16_data(self, data: bytes, count: int) -> List[int]:
        """解析uint16数组(小端)"""
        return list(struct.unpack(f"<{count}H", data[:2*count]))
    
    def get_wavelength_range(self) -> Dict[str, float]:
        """
        获取光谱起始和终止波长
        返回: {'start': 起始波长, 'end': 终止波长}
        """
        resp = self._send_command(self.CMD_TYPES["get_wavelength_range"])
        
        # 解析波长数据(小端16位整数)
        start_nm = struct.unpack("<H", resp[:2])[0]
        end_nm = struct.unpack("<H", resp[2:4])[0]
        
        return {"start": start_nm, "end": end_nm}
    
    def _parse_spectrum_data(self, data: bytes, wavelengths: Tuple[int, int]) -> Dict:
        """
        解析光谱数据(内部方法)
        :param data: 原始数据
        :param wavelengths: (起始波长, 终止波长)
        :return: 解析后的光谱数据字典
        """
        # 计算光谱点数(每nm一个点)
        start, end = wavelengths
        num_points = end - start + 1
        
        # 解析基本参数
        exposure_status = data[0]
        exposure_time = struct.unpack("<I", data[1:5])[0]
        
        # 解析光度学参数(47个浮点数)
        photometric_offset = 5
        photometric_data = data[photometric_offset:photometric_offset+47 * 4]
        photometric_values = self._parse_float_data(photometric_data, 47)
        
        # 解析近红外参数(3个浮点数)
        nir_offset = photometric_offset + 47 * 4
        nir_data = data[nir_offset:nir_offset+3 * 4]
        nir_values = self._parse_float_data(nir_data, 3)
        
        # 解析光谱系数
        coeff_offset = nir_offset + 3 * 4
        spectrum_coeff = struct.unpack("<h", data[coeff_offset:coeff_offset+2])[0]
        scale_factor = 10 ** spectrum_coeff
        
        # 解析光谱数据
        spectrum_offset = coeff_offset + 2
        spectrum_data = data[spectrum_offset:spectrum_offset+num_points*2]
        raw_spectrum = self._parse_uint16_data(spectrum_data, num_points)
        scaled_spectrum = [val / scale_factor for val in raw_spectrum]
        
        # 构建波长轴
        wavelengths = list(range(start, end + 1))

        return {
            "measure_time_epoch": time.time(),
            "exposure_status": self.EXPOSURE_STATUS.get(exposure_status, "未知"),
            "exposure_time_us": exposure_time,
            "photometric": dict(zip(self.PHOTOMETRIC_PARAMS, photometric_values)),
            "nir": dict(zip(self.NIR_PARAMS, nir_values)),
            "spectrum_coeff": spectrum_coeff,
            "wavelengths": wavelengths,
            "spectrum": scaled_spectrum,
        }
    
    def get_single_spectrum(self) -> Dict:
        """获取单帧光谱数据"""
        # 先获取波长范围
        wavelength_range = self.get_wavelength_range()
        start, end = wavelength_range["start"], wavelength_range["end"]
        num_points = end - start + 1
        
        # 发送单帧命令
        resp = self._send_command(self.CMD_TYPES["single_spectrum"])
        
        # 解析光谱数据
        return self._parse_spectrum_data(resp, (start, end))
    
    def start_continuous_acquisition(self):
        """开始连续获取光谱数据"""
        self._send_command(self.CMD_TYPES["start_continuous"])
    
    def stop_continuous_acquisition(self):
        """停止连续获取光谱数据"""
        self._send_command(self.CMD_TYPES["stop_continuous"], no_response=True)
    
    def acquire_continuous_spectra(self, num_frames: int, discard_bad_exposure: bool = True, 
                                   discard_initial_frames: int = 5, 
                                   max_discard_frames: int = 10) -> List[Dict]:
        """
        连续采集指定帧数的光谱数据
        :param num_frames: 要采集的帧数
        :param discard_bad_exposure: 是否丢弃曝光不良的帧
        :param discard_initial_frames: 丢弃的初始帧数
        :param max_discard_frames: 最大丢弃帧数
        :return: 光谱数据字典列表
        """
        # 获取波长范围
        wavelength_range = self.get_wavelength_range()
        start, end = wavelength_range["start"], wavelength_range["end"]
        # num_points = end - start + 1
        
        # # 计算预期数据包大小
        # frame_size = 6 + 1 + 4 + 47 * 4 + 3 * 4 + 2 + num_points*2 + 3
        
        # 开始连续采集
        self.start_continuous_acquisition()
        
        spectra = []
        frame_count = 0
        discarded_valid_frames = 0
        try:
            while frame_count < num_frames:
                # assert self.ser.in_waiting == frame_size
                time.sleep(0.03)
                # 读取完整帧
                header = self.ser.read(2)
                if header != self.RESP_HEADER:
                    self.ser.reset_input_buffer()  # 清空缓冲区
                    continue
                
                # 读取包长度(小端3字节)
                len_bytes = self.ser.read(3)
                total_len = struct.unpack("<I", len_bytes + b"\x00")[0]
                
                # 读取剩余数据
                remaining = total_len - 5
                resp_data = self.ser.read(remaining)
                
                # 验证结束标识和校验和
                if resp_data[-2:] != self.END_MARKER:
                    continue  # 跳过无效帧

                if discarded_valid_frames < discard_initial_frames:
                    discarded_valid_frames += 1
                    continue  # 跳过初始丢弃帧

                # 提取有效数据(跳过类型字节)
                valid_data = resp_data[1:-3]
                
                # 解析光谱数据
                spectrum_data = self._parse_spectrum_data(valid_data, (start, end))
                time.sleep(spectrum_data["exposure_time_us"] * 1e-6)  # 等待曝光时间（预期下一帧曝光时间相同）
                if spectrum_data["exposure_status"] != "正常" and discard_bad_exposure:
                    if discarded_valid_frames < max_discard_frames:
                        discarded_valid_frames += 1
                        continue  # 跳过曝光不良的帧
                spectra.append(spectrum_data)
                frame_count += 1
        
        finally:
            # 确保停止采集
            self.stop_continuous_acquisition()
            time.sleep(0.03)
        
        return spectra
    
    def get_device_info(self, expected_info_length: int = 24) -> str:
        """获取设备信息"""
        resp = self._send_command(self.CMD_TYPES["get_device_info"], data=expected_info_length.to_bytes(1))
        return resp.decode("ascii").strip()
    
    def set_exposure_mode(self, auto: bool) -> bool:
        """
        设置曝光模式
        :param auto: True=自动, False=手动
        :return: 是否设置成功
        """
        mode = 0x01 if auto else 0x00
        resp = self._send_command(self.CMD_TYPES["set_exposure_mode"], bytes([mode]))
        return resp[0] == 0x00  # 检查返回状态
    
    def get_exposure_mode(self) -> str:
        """获取当前曝光模式"""
        resp = self._send_command(self.CMD_TYPES["get_exposure_mode"])
        return "自动" if resp[0] == 0x01 else "手动"
    
    def set_exposure_time(self, time_us: int) -> bool:
        """
        设置曝光时间(手动模式)
        :param time_us: 曝光时间(微秒)
        :return: 是否设置成功
        """
        data = struct.pack("<I", time_us)
        resp = self._send_command(self.CMD_TYPES["set_exposure_time"], data)
        return resp[0] == 0x00  # 检查返回状态
    
    def get_exposure_time(self) -> int:
        """获取当前曝光时间"""
        resp = self._send_command(self.CMD_TYPES["get_exposure_time"])
        if len(resp) != 4:
            return float('nan')  # 如果响应长度不正确，返回NaN
        return struct.unpack("<I", resp)[0]
    
    def set_max_exposure_time(self, max_time_us: int) -> bool:
        """
        设置最大曝光时间
        :param max_time_us: 最大曝光时间(微秒)
        :return: 是否设置成功
        """
        data = struct.pack("<I", max_time_us)
        resp = self._send_command(self.CMD_TYPES["set_max_exposure"], data)
        return resp[0] == 0x00  # 检查返回状态
    
    def get_max_exposure_time(self) -> int:
        """获取最大曝光时间"""
        resp = self._send_command(self.CMD_TYPES["get_max_exposure"])
        return struct.unpack("<I", resp)[0]
    
    def set_cie_mode(self, mode: str) -> bool:
        """
        设置CIE模式
        :param mode: 模式字符串 
            ('CIE1931_2deg', 'CIE1964_10deg', 'CIE2015_2deg', 'CIE2015_10deg')
        :return: 是否设置成功
        """
        mode_map = {v: k for k, v in self.CIE_MODES.items()}
        if mode not in mode_map:
            raise ValueError("无效的CIE模式")
        
        data = bytes([mode_map[mode]])
        resp = self._send_command(self.CMD_TYPES["set_cie_mode"], data)
        return resp[0] == 0x00  # 检查返回状态
    
    def get_cie_mode(self) -> str:
        """获取当前CIE模式"""
        resp = self._send_command(self.CMD_TYPES["get_cie_mode"])
        return self.CIE_MODES.get(resp[0], "未知")
    
    def start_efficiency_curve(self, ratios: List[float]) -> None:
        """
        发送效率曲线修正比值
        :param ratios: 效率曲线修正比值列表，每个元素是一个浮点数
        """
        # 1. 发送起始数据包
        self._send_command(self.CMD_TYPES["start_efficiency_curve"], bytes([0x04]))
        
        # 2. 将浮点数列表转换为字节数据
        ratio_bytes = b""
        for ratio in ratios:
            # 将浮点数转换为小端字节序的4字节表示
            ratio_bytes += struct.pack("<f", ratio)
        
        # 3. 计算每个包的最大数据长度(不超过999字节)
        max_data_per_packet = 999 - 9  # 减去包头(2B) + 包长(3B) + 命令类型(1B) + 校验(1B) + 结束符(2B)
        num_packets = (len(ratio_bytes) + max_data_per_packet - 1) // max_data_per_packet
        
        # 4. 分多个包发送效率曲线数据
        for i in range(num_packets):
            start_idx = i * max_data_per_packet
            end_idx = min((i + 1) * max_data_per_packet, len(ratio_bytes))
            packet_data = ratio_bytes[start_idx:end_idx]
            
            # 计算总包长(包括包头+包长+命令类型+数据+校验+结束符)
            total_length = 2 + 3 + 1 + len(packet_data) + 1 + 2
            length_bytes = struct.pack("<I", total_length)[:3]  # 小端序3字节
            
            # 构建完整命令包
            cmd_data = self.CMD_HEADER + length_bytes + bytes([self.CMD_TYPES["start_efficiency_curve"]]) + packet_data
            checksum = sum(cmd_data) & 0xFF
            full_cmd = cmd_data + bytes([checksum]) + self.END_MARKER
            
            # 发送包
            self.ser.write(full_cmd)
            
            # 添加短暂延迟防止缓冲区溢出
            time.sleep(0.03)
    
    def validate_efficiency_curve(self) -> bool:
        """校验效率曲线并计算"""
        resp = self._send_command(self.CMD_TYPES["validate_efficiency_curve"])
        return resp[0] == 0x00  # 检查返回状态
    
    def reset_efficiency_curve(self) -> bool:
        """恢复出厂效率曲线设置"""
        resp = self._send_command(self.CMD_TYPES["reset_efficiency_curve"])
        return resp[0] == 0x00  # 检查返回状态
    
    def close(self):
        """关闭串口连接"""
        self.ser.close()

# 使用示例
if __name__ == "__main__":
    # 初始化设备连接
    spectrometer = PJGSpectrometer("/dev/cu.usbmodem57190190851")  # 替换为实际串口号
    
    try:
        # 获取波长范围
        wavelength_range = spectrometer.get_wavelength_range()
        print(f"波长范围: {wavelength_range['start']}nm - {wavelength_range['end']}nm")
        
        # # 设置曝光模式为手动
        # if spectrometer.set_exposure_mode(auto=False):
        #     print("曝光模式已设置为手动")

        # 设置曝光模式为自动
        if spectrometer.set_exposure_mode(auto=True):
            print("曝光模式已设置为自动")

        # 设置曝光时间
        if spectrometer.set_exposure_time(10000):
            print("成功设置曝光时间")

        # 获取单帧光谱
        single_spectrum = spectrometer.get_single_spectrum()
        print(f"单帧曝光时间: {single_spectrum['exposure_time_us'] * 1e-3} ms")
        print(f"CCT: {single_spectrum['photometric']['CCT']}K")
        
        # 连续采集多帧光谱
        print("开始连续采集多帧光谱...")
        spectra = spectrometer.acquire_continuous_spectra(5)
        print(f"成功采集 {len(spectra)} 帧光谱数据")
        
        # # 设置效率曲线
        # print("发送效率曲线修正比值...")
        # ratios = [1.5] * 661  # 创建661个1.5的比值列表
        # spectrometer.start_efficiency_curve(ratios)
        
        # # 校验效率曲线
        # if spectrometer.validate_efficiency_curve():
        #     print("效率曲线校验成功")
        # else:
        #     print("效率曲线校验失败")
        
        # # 恢复出厂效率曲线设置
        # if spectrometer.reset_efficiency_curve():
        #     print("效率曲线已恢复出厂设置")
        
        # 获取设备信息
        device_info = spectrometer.get_device_info()
        print(f"设备信息: {device_info}")
        
    finally:
        # 确保关闭连接
        spectrometer.close()
