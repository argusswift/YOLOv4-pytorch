import logging


class Logger(object):
    def __init__(self,log_file_name,log_level,logger_name):
        #第一步，创建一个logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        #第二步，创建一个handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        #第三步,定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s '
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        #第四步,将Hander添加到logger中
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger