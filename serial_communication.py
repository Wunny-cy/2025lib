import serial
import time

class SerialCommunication:
    def __init__(self, port='COM22', baudrate=115200):
        """
        初始化串口通信
        Args:
            port: 串口名称
            baudrate: 波特率
        """
        try:
            self.serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"成功连接到串口 {port}")
            
            # 执行初始化握手
            if not self._handshake():
                raise Exception("与STM32握手失败")
            
                
        except Exception as e:
            raise Exception(f"串口连接失败: {str(e)}")
    
    
    def _handshake(self):
        """
        与STM32进行握手确认
        Returns:
            bool: 握手是否成功
        """
        try:
            # 清空接收缓冲区
            self.serial.reset_input_buffer()
            
            # 发送握手消息
            self.send_command("HELLO STM32\r\n")
            
            # 等待接收响应
            start_time = time.time()
            while time.time() - start_time < 10:  # 最多等待X秒
                response = self.read_response()
                if response == "HAL_OK!":
                    print("与STM32握手成功")
                    return True
                time.sleep(0.1)
            
            print("等待STM32响应超时")
            return False
            
        except Exception as e:
            print(f"握手过程出错: {str(e)}")
            return False
    
    def send_command(self, command):
        """
        发送命令到机器人
        Args:
            command: 要发送的命令字符串
        """
        try:
            # 添加换行符确保命令完整
            command = command + '\r\n'
            self.serial.write(command.encode())
            time.sleep(0.1)  # 等待命令执行
        except Exception as e:
            raise Exception(f"发送命令失败: {str(e)}")

    def check_ok(self):
        while True:
            response = self.read_response()
            print(response)
            if response != None and "ok" in response.lower():
                # print(222)
                break
            time.sleep(0.1)

    def read_response(self):
        """
        读取机器人返回的数据
        Returns:
        str: 返回的数据
        """
        max_retries = 3  # 最大重试次数
        retry_count = 0  # 当前重试次数

        while retry_count < max_retries:
            if self.serial.in_waiting:
                response = self.serial.readline()
                try:
                    # 尝试解码响应
                    decoded_response = response.decode('utf-8').strip()
                    print(f"解码成功: {decoded_response}")
                    return decoded_response
                except UnicodeDecodeError as e:
                    print(f"解码错误: {e}")
                    retry_count += 1
                    print(f"重试次数: {retry_count}")
                    # # 重新发送数据
                    # self.serial.write(b'requestdata')  # 假设发送请求数据的命令
                    self.send_command("GD")#风险
                    time.sleep(0.1)  # 等待一段时间再重试
            else:
                return None

        print("达到最大重试次数，读取失败")
        return None

    def close(self):
        """
        关闭串口连接
        """
        if self.serial.is_open:
            self.serial.close()
            print("串口连接已关闭") 