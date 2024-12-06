from attitudex import Attitude

class Test_Attitude:
    def test__init__(self):
        att = Attitude()
        assert isinstance(att, Attitude)