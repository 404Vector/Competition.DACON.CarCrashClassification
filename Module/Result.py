import typing
import numpy as np

__all__ = [
    "Result"
]

class Result:
    '''
    13가지의 차량 충돌 상황 Class의 세부 정보
    
    ego-Involve, weather, timing의 정보는 '차량 충돌 사고'가 일어난 경우에만 분석합니다.
    '''
    def __init__(self, label:int) -> None:
        self.label = label

    number_of_classes = {
        'all':13,
        'crash':2,
        'ego_involve':2,
        'weather':3,
        'timing':2,
    }

    one_hot_encoder = {
        'all': np.ones((13,13), np.float32).tolist(),
        'crash': np.ones((2,2), np.float32).tolist(),
        'ego_involve': np.ones((2,2), np.float32).tolist(),
        'weather': np.ones((3,3), np.float32).tolist(),
        'timing': np.ones((2,2), np.float32).tolist(),
        'w&t': np.ones((6,6), np.float32).tolist(),
        'c&e': np.ones((3,3), np.float32).tolist(),
    }

    
    @property
    def crash(self)-> bool:
        '''
        차량 충돌 여부 (No/Yes)
        '''
        return self.label != 0

    @property
    def encoded_crash(self) -> int:
        return 1 if self.crash == True else 0

    @property
    def ego_involve(self) -> typing.Union[bool, None]:
        '''
        본인 차량의 충돌 사고 연류 여부 (No/Yes) or None
        '''
        return self.label < 7 if self.crash else None

    @property
    def encoded_ego_involve(self) -> int:
        return 1 if self.ego_involve == True else 0

    @property
    def weather(self) -> typing.Union[str, None]:
        '''
        weather : 날씨 상황 (Normal/Snowy/Rainy) or None
        '''
        if not self.crash:
            return None
        idx = ((self.label - 1) // 2) % 3
        if idx == 0:
            return 'Normal'
        elif idx == 1:
            return 'Snowy'
        elif idx == 2:
            return 'Rainy'
        else:
            raise Exception(f'unknown idx : {idx}, (label : {self.label})')

    @property
    def encoded_weather(self) -> int:
        if self.weather == 'Normal':
            return 0
        elif self.weather == 'Snowy':
            return 1
        elif self.weather == 'Rainy':
            return 2
        else:
            return 0

    @property
    def timing(self) -> typing.Union[str, None]:
        '''
        timing : 낮과 밤 (Day/Night)
        '''
        if not self.crash:
            return None
        return 'Day' if self.label % 2 == 1 else 'Night'

    @property
    def encoded_timing(self) -> int:
        if self.timing == 'Day':
            return 0
        elif self.timing == 'Night':
            return 1
        else:
            return 0

    @property
    def weather_by_timing(self) -> typing.Union[str, None]:
        if not self.crash:
            return None
        else:
            return f'{self.weather}-{self.timing}'

    @property
    def encoded_weather_by_timing(self) -> int:
        return self.encoded_weather + (3*self.encoded_timing)

    @property
    def crash_and_ego(self) -> typing.Union[str, None]:
        if self.ego_involve == True:
            return 'ego_involve'
        else:
            return 'crash' if self.crash == True else None
    @property
    def encoded_crash_and_ego(self) -> int:
        return self.encoded_crash + self.encoded_ego_involve