import json
from pathlib import Path

import numpy as np
import pytest

from attipy import Attitude

_FIXTURES = json.loads((Path(__file__).parent / "testdata" / "attitudes.json").read_text())


class Test_Attitude:
    @pytest.mark.parametrize("att", _FIXTURES)
    def test__init__(self, att):
        q = att["quaternion"]
        assert Attitude(q)._q == pytest.approx(q)

    def test__init__wrong_shape(self):
        with pytest.raises(ValueError):
            Attitude([1.0, 0.0, 0.0])

    def test__init__non_unit(self):
        with pytest.raises(ValueError):
            Attitude([1.0, 1.0, 0.0, 0.0])

    def test__repr__(self):
        q = [0.52005444, -0.51089824, 0.64045922, 0.24153336]
        att = Attitude(q)
        repr_str = repr(att)
        expected_str = "Attitude(q=[0.52 + -0.511i + 0.64j + 0.242k])"
        assert repr_str == expected_str

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_canonical_sign(self, att):
        q = np.array(att["quaternion"])
        np.testing.assert_allclose(Attitude(-q)._q, q)

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_quaternion(self, att):
        q = att["quaternion"]
        assert Attitude.from_quaternion(q)._q == pytest.approx(q)

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_quaternion(self, att):
        q = att["quaternion"]
        np.testing.assert_allclose(Attitude(q).as_quaternion(), q)

    def test_as_quaternion_returns_copy(self):
        q = [1.0, 0.0, 0.0, 0.0]
        att = Attitude(q)
        att.as_quaternion()[0] = 0.0
        np.testing.assert_allclose(att._q, q)

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_matrix(self, att):
        result = Attitude.from_matrix(att["matrix"])
        np.testing.assert_allclose(result._q, att["quaternion"])

    def test_from_matrix_wrong_shape(self):
        with pytest.raises(ValueError):
            Attitude.from_matrix(np.eye(2))

    def test_from_matrix_non_orthogonal(self):
        with pytest.raises(ValueError):
            Attitude.from_matrix([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_from_matrix_improper(self):
        with pytest.raises(ValueError):
            Attitude.from_matrix(-np.eye(3))

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_matrix(self, att):
        result = Attitude(att["quaternion"]).as_matrix()
        np.testing.assert_allclose(result, att["matrix"])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_euler_rad(self, att):
        result = Attitude.from_euler(att["euler_rad"], degrees=False)
        np.testing.assert_allclose(result._q, att["quaternion"])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_euler_deg(self, att):
        euler_deg = np.degrees(att["euler_rad"])
        result = Attitude.from_euler(euler_deg, degrees=True)
        np.testing.assert_allclose(result._q, att["quaternion"])

    def test_from_euler_wrong_shape(self):
        with pytest.raises(ValueError):
            Attitude.from_euler([0.0, 0.0])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_euler_rad(self, att):
        result = Attitude(att["quaternion"]).as_euler(degrees=False)
        np.testing.assert_allclose(result, att["euler_rad"])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_euler_deg(self, att):
        result = Attitude(att["quaternion"]).as_euler(degrees=True)
        np.testing.assert_allclose(result, np.degrees(att["euler_rad"]))

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_rotvec_rad(self, att):
        result = Attitude.from_rotvec(att["rotvec"], degrees=False)
        np.testing.assert_allclose(result._q, att["quaternion"])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_from_rotvec_deg(self, att):
        rotvec_deg = np.degrees(att["rotvec"])
        result = Attitude.from_rotvec(rotvec_deg, degrees=True)
        np.testing.assert_allclose(result._q, att["quaternion"])

    def test_from_rotvec_wrong_shape(self):
        with pytest.raises(ValueError):
            Attitude.from_rotvec([0.0, 0.0])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_rotvec_rad(self, att):
        result = Attitude(att["quaternion"]).as_rotvec(degrees=False)
        np.testing.assert_allclose(result, att["rotvec"])

    @pytest.mark.parametrize("att", _FIXTURES)
    def test_as_rotvec_deg(self, att):
        result = Attitude(att["quaternion"]).as_rotvec(degrees=True)
        np.testing.assert_allclose(result, np.degrees(att["rotvec"]))
