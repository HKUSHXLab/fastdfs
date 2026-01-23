
import pytest
import pandas as pd
import numpy as np
from fastdfs.utils.type_utils import safe_convert_to_string

class TestTypeUtils:
    def test_safe_convert_to_string_ints(self):
        # Ints
        s = pd.Series([1, 2, 3])
        res = safe_convert_to_string(s)
        assert res.dtype == 'object'
        assert res.tolist() == ['1', '2', '3']

    def test_safe_convert_to_string_floats_failure(self):
        # Floats - should fail even if they are structurally ints, per new simplified requirements
        # "safe_convert_to_string: simplify, just throw error if the type is floating type"
        s = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Cannot safe convert float column"):
            safe_convert_to_string(s)
        
        # Floats with NaNs
        s = pd.Series([1.0, np.nan, 2.0])
        with pytest.raises(ValueError, match="Cannot safe convert float column"):
            safe_convert_to_string(s)

        # Floats that are NOT ints
        s = pd.Series([1.5, 2.0])
        with pytest.raises(ValueError, match="Cannot safe convert float column"):
            safe_convert_to_string(s)
            
    def test_safe_convert_to_string_strs(self):
        # Strings
        s = pd.Series(['a', 'b'])
        res = safe_convert_to_string(s)
        assert res.tolist() == ['a', 'b']

    def test_safe_convert_to_string_mixed(self):
        # Object column with mixed types (not explicit float dtype)
        s = pd.Series([1, 'a'])
        res = safe_convert_to_string(s)
        assert res.tolist() == ['1', 'a']
