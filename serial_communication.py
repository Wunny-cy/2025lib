import serial
import time

class SerialCommunication:
    def __init__(self, port='COM13', baudrate=115200):
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
            self.send_command("收到")
            
            # 等待接收响应
            start_time = time.time()
            while time.time() - start_time < 5:  # 最多等待5秒
                response = self.read_response()
                if response == "收到":
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
            command = command + '\n'
            self.serial.write(command.encode())
            time.sleep(0.1)  # 等待命令执行
        except Exception as e:
            raise Exception(f"发送命令失败: {str(e)}")
    
    def read_response(self):
        """
        读取机器人返回的数据
        Returns:
            str: 返回的数据
        """
        try:
            if self.serial.in_waiting:
                response = self.serial.readline().decode().strip()
                return response
            return None
        except Exception as e:
            raise Exception(f"读取响应失败: {str(e)}")

    def close(self):
        """
        关闭串口连接
        """
        if self.serial.is_open:
            self.serial.close()
            print("串口连接已关闭") 