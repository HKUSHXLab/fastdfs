
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

    def test_safe_convert_to_string_nullable_int_preserves_nulls(self):
        """Nullable Int64 with pd.NA should convert values to strings and preserve NULLs.
        """
        s = pd.Series(pd.array([1, 2, pd.NA], dtype="Int64"), name="AdID")
        res = safe_convert_to_string(s)
        assert res.tolist()[:2] == ['1', '2']
        assert pd.isna(res.iloc[2]), "NA should be preserved as pd.NA, not converted to string '<NA>'"
        # Ensure '<NA>' literal string is NOT in the result
        assert '<NA>' not in res.astype(str).values[:2].tolist()


    def test_safe_convert_to_string_object_with_none(self):
        """Object dtype with None should preserve NULLs."""
        s = pd.Series(["a", None, "c"], name="key")
        res = safe_convert_to_string(s)
        assert res.iloc[0] == "a"
        assert res.iloc[2] == "c"
        assert pd.isna(res.iloc[1])
