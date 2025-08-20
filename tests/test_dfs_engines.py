"""
Tests for the new DFS engine interface (Phase 2).

This test file validates that the new DFS engines correctly compute features
for external target dataframes using the simplified RDB dataset interface.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from fastdfs.dfs import DFSConfig, get_dfs_engine, FeaturetoolsEngine, DFS2SQLEngine
from fastdfs.dataset.rdb_simplified import RDBDataset
from fastdfs.api_new import load_rdb, compute_dfs_features, DFSPipeline


@pytest.fixture
def test_data_path():
    """Path to new RDB test dataset."""
    return Path(__file__).parent / "data" / "test_rdb_new"

@pytest.fixture
def rdb_dataset(test_data_path):
    """Load RDB dataset directly from new test data."""
    return RDBDataset(test_data_path)

@pytest.fixture
def target_dataframe(rdb_dataset):
    """Create a target dataframe using actual IDs from the dataset."""
    # Get some real user and item IDs from the dataset
    user_table = rdb_dataset.get_table('user')
    item_table = rdb_dataset.get_table('item')

    # Take first few users and items
    user_ids = user_table['user_id'].head(3).tolist()
    item_ids = item_table['item_id'].head(3).tolist()

    return pd.DataFrame({
        "user_id": [user_ids[0], user_ids[1], user_ids[2], user_ids[0], user_ids[1]],
        "item_id": [item_ids[0], item_ids[1], item_ids[2], item_ids[1], item_ids[0]],
        "interaction_time": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-04", "2024-01-05"
        ])
    })

@pytest.fixture
def cutoff_test_scenarios():
    """
    Create target dataframes with specific entities and hardcoded cutoff times
    to test different temporal filtering scenarios.

    Based on analysis of the test data:
    - Test User (74253567...): 14 interactions at timestamps 00:00:00 and 00:01:00
    - Test Item (fa9407df...): 14 interactions spread from 00:00:00 to 01:15:00
    """
    # These are actual entity IDs from the test dataset with known timestamp patterns
    TEST_USER_ID = "74253567-a345-43ab-9ea1-502391f5d63b"
    TEST_ITEM_ID = "fa9407df-1720-4652-a17c-94d7ded6ce60"

    scenarios = {
        # Cutoff before all interactions - should give 0 counts
        'before_all': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2022-12-31 23:00:00")]
        }),

        # Cutoff that includes some user interactions (4) and few item interactions (1)
        'partial_interactions': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2023-01-01 00:00:30")]
        }),

        # Cutoff that includes all user interactions (14) and some item interactions (7)
        'middle_spread': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2023-01-01 00:07:30")]
        }),

        # Cutoff that includes most interactions for both entities
        'most_data': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2023-01-01 00:43:30")]
        }),

        # Cutoff after all interactions - should include everything
        'after_all': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2023-01-01 02:00:00")]
        }),

        # Test exact timestamp boundary (at exactly 00:00:00)
        # DFS should exclude interactions AT the cutoff time (strict less-than)
        'exact_boundary': pd.DataFrame({
            "user_id": [TEST_USER_ID],
            "item_id": [TEST_ITEM_ID],
            "cutoff_time": [pd.Timestamp("2023-01-01 00:00:00")]
        })
    }

    return scenarios

@pytest.fixture
def key_mappings():
    """Key mappings for test target dataframe."""
    return {
        "user_id": "user.user_id",
        "item_id": "item.item_id"
    }


class TestDFSConfig:
    """Test DFS configuration."""

    def test_default_config(self):
        """Test default DFS configuration."""
        config = DFSConfig()

        assert config.max_depth == 2
        assert config.use_cutoff_time == True
        assert config.engine == "featuretools"
        assert "count" in config.agg_primitives
        assert "mean" in config.agg_primitives

    def test_custom_config(self):
        """Test custom DFS configuration."""
        config = DFSConfig(
            max_depth=3,
            engine="dfs2sql",
            agg_primitives=["count", "sum"],
            use_cutoff_time=False
        )

        assert config.max_depth == 3
        assert config.engine == "dfs2sql"
        assert config.agg_primitives == ["count", "sum"]
        assert config.use_cutoff_time == False


class TestEngineRegistration:
    """Test DFS engine registration system."""

    def test_get_dfs_engine_featuretools(self):
        """Test getting the Featuretools DFS engine."""
        config = DFSConfig(engine="featuretools")
        engine = get_dfs_engine("featuretools", config)
        assert isinstance(engine, FeaturetoolsEngine)

    def test_get_dfs_engine_dfs2sql(self):
        """Test getting the DFS2SQL engine."""
        config = DFSConfig(engine="dfs2sql")
        engine = get_dfs_engine("dfs2sql", config)
        assert isinstance(engine, DFS2SQLEngine)


class TestFeaturetoolsEngine:
    """Test Featuretools engine implementation."""

    def test_engine_registration(self):
        """Test that Featuretools engine is properly registered."""
        config = DFSConfig(engine="featuretools")
        engine = get_dfs_engine("featuretools", config)
        assert isinstance(engine, FeaturetoolsEngine)
        assert engine.name == "featuretools"

    def test_compute_features_basic(self, rdb_dataset, target_dataframe, key_mappings):
        """Test basic feature computation with Featuretools engine."""
        config = DFSConfig(
            engine="featuretools",
            max_depth=2,
            agg_primitives=["count", "mean"],
            use_cutoff_time=False
        )

        engine = get_dfs_engine("featuretools", config)

        # Compute features
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )

        print(result_df.columns)

        # Check that result contains original columns
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
        assert "interaction_time" in result_df.columns

        # Check the following features are in the result
        assert "user.user_feature_0" in result_df.columns
        assert "item.item_feature_0" in result_df.columns
        assert "user.COUNT(interaction)" in result_df.columns
        assert "item.COUNT(interaction)" in result_df.columns

        # Check that result has same number of rows
        assert len(result_df) == len(target_dataframe)

    def test_compute_features_with_cutoff_time(self, rdb_dataset, target_dataframe, key_mappings):
        """Test feature computation with cutoff time."""
        config = DFSConfig(
            engine="featuretools",
            max_depth=2,
            agg_primitives=["count", "mean"],
            use_cutoff_time=True
        )

        engine = get_dfs_engine("featuretools", config)

        # Compute features with cutoff time
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column="interaction_time"
        )

        print(result_df.columns)

        # Check that result contains original columns
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
        assert "interaction_time" in result_df.columns

        # Check the following features are in the result
        assert "user.user_feature_0" in result_df.columns
        assert "item.item_feature_0" in result_df.columns
        assert "user.COUNT(interaction)" in result_df.columns
        assert "item.COUNT(interaction)" in result_df.columns

        # Check that result has same number of rows
        assert len(result_df) == len(target_dataframe)

class TestDFS2SQLEngine:
    """Test DFS2SQL engine implementation."""

    def test_engine_registration(self):
        """Test that DFS2SQL engine is properly registered."""
        config = DFSConfig(engine="dfs2sql")
        engine = get_dfs_engine("dfs2sql", config)
        assert isinstance(engine, DFS2SQLEngine)
        assert engine.name == "dfs2sql"

    def test_compute_features_basic(self, rdb_dataset, target_dataframe, key_mappings):
        """Test basic feature computation with DFS2SQL engine."""
        config = DFSConfig(
            engine="dfs2sql",
            max_depth=2,
            agg_primitives=["count", "mean"],
            use_cutoff_time=False
        )

        engine = get_dfs_engine("dfs2sql", config)

        # Compute features
        result_df = engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )

        print(result_df.columns)

        # Check that result contains original columns
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns
        assert "interaction_time" in result_df.columns

        # Check the following features are in the result
        assert "user.user_feature_0" in result_df.columns
        assert "item.item_feature_0" in result_df.columns
        assert "user.COUNT(interaction)" in result_df.columns
        assert "item.COUNT(interaction)" in result_df.columns

        # Check that result has same number of rows
        assert len(result_df) == len(target_dataframe)


class TestEngineComparison:
    """Test that Featuretools and DFS2SQL engines produce equivalent results."""

    def test_engines_produce_same_features_basic(self, rdb_dataset, target_dataframe, key_mappings):
        """Test that both engines produce the same features without cutoff time."""

        # Configure both engines with identical settings
        config = DFSConfig(
            max_depth=2,
            agg_primitives=["count", "mean"],
            use_cutoff_time=False
        )

        # Compute features with Featuretools engine
        ft_engine = get_dfs_engine("featuretools", config)
        ft_result = ft_engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )

        # Compute features with DFS2SQL engine
        sql_engine = get_dfs_engine("dfs2sql", config)
        sql_result = sql_engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column=None
        )

        # Both should have same shape
        assert ft_result.shape == sql_result.shape, f"Shape mismatch: FT {ft_result.shape} vs SQL {sql_result.shape}"

        # Both should have same columns (order may differ)
        ft_feature_cols = [col for col in ft_result.columns if col not in target_dataframe.columns]
        sql_feature_cols = [col for col in sql_result.columns if col not in target_dataframe.columns]

        assert set(ft_feature_cols) == set(sql_feature_cols), f"Feature columns differ: FT {ft_feature_cols} vs SQL {sql_feature_cols}"

        # Check that feature values are equivalent (within numerical tolerance)
        for col in ft_feature_cols:
            ft_values = ft_result[col].fillna(0).values
            sql_values = sql_result[col].fillna(0).values

            # Use numpy allclose for numerical comparison
            import numpy as np
            assert np.allclose(ft_values, sql_values, rtol=1e-5, atol=1e-8), f"Values differ for column {col}"

        print(f"✓ Both engines produced {len(ft_feature_cols)} equivalent features")

    def test_engines_produce_same_features_with_cutoff_time(self, rdb_dataset, target_dataframe, key_mappings):
        """Test that both engines produce the same features with cutoff time."""

        # Configure both engines with identical settings including cutoff time
        config = DFSConfig(
            max_depth=2,
            agg_primitives=["count", "mean"],
            use_cutoff_time=True
        )

        # Compute features with Featuretools engine
        ft_engine = get_dfs_engine("featuretools", config)
        ft_result = ft_engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column="interaction_time"
        )

        # Compute features with DFS2SQL engine
        sql_engine = get_dfs_engine("dfs2sql", config)
        sql_result = sql_engine.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            cutoff_time_column="interaction_time"
        )

        # Both should have same shape
        assert ft_result.shape == sql_result.shape, f"Shape mismatch: FT {ft_result.shape} vs SQL {sql_result.shape}"

        # Both should have same columns (order may differ)
        ft_feature_cols = [col for col in ft_result.columns if col not in target_dataframe.columns]
        sql_feature_cols = [col for col in sql_result.columns if col not in target_dataframe.columns]

        assert set(ft_feature_cols) == set(sql_feature_cols), f"Feature columns differ: FT {ft_feature_cols} vs SQL {sql_feature_cols}"

        # Check that feature values are equivalent (within numerical tolerance)
        for col in ft_feature_cols:
            ft_values = ft_result[col].fillna(0).values
            sql_values = sql_result[col].fillna(0).values

            # Use numpy allclose for numerical comparison
            assert np.allclose(ft_values, sql_values, rtol=1e-5, atol=1e-8), f"Values differ for column {col}"

        print(f"✓ Both engines produced {len(ft_feature_cols)} equivalent features with cutoff time")


class TestHighLevelAPI:
    """Test the high-level API functions."""

    def test_load_rdb(self, test_data_path):
        """Test loading RDB using high-level API."""
        rdb = load_rdb(str(test_data_path))
        assert isinstance(rdb, RDBDataset)
        assert len(rdb.table_names) == 3

    def test_compute_dfs_features_default_config(self, rdb_dataset, target_dataframe, key_mappings):
        """Test computing features with default configuration."""
        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings
        )

        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns

    def test_compute_dfs_features_with_overrides(self, rdb_dataset, target_dataframe, key_mappings):
        """Test computing features with config overrides."""
        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings,
            config_overrides={
                "max_depth": 1,
                "engine": "featuretools",
                "agg_primitives": ["count"]
            }
        )

        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns

    def test_dfs_pipeline(self, rdb_dataset, target_dataframe, key_mappings):
        """Test DFS pipeline functionality."""
        config = DFSConfig(
            max_depth=1,
            agg_primitives=["count"],
            engine="featuretools"
        )

        pipeline = DFSPipeline(
            transform_pipeline=None,  # No transforms for this test
            dfs_config=config
        )

        result_df = pipeline.compute_features(
            rdb=rdb_dataset,
            target_dataframe=target_dataframe,
            key_mappings=key_mappings
        )

        # Check basic properties
        assert len(result_df) == len(target_dataframe)
        assert "user_id" in result_df.columns


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_unknown_engine(self):
        """Test error for unknown engine name."""
        config = DFSConfig(engine="nonexistent")

        with pytest.raises(ValueError, match="Unknown DFS engine"):
            get_dfs_engine("nonexistent", config)

    def test_empty_target_dataframe(self, rdb_dataset, key_mappings):
        """Test behavior with empty target dataframe."""
        empty_df = pd.DataFrame(columns=["user_id", "item_id"])

        result_df = compute_dfs_features(
            rdb=rdb_dataset,
            target_dataframe=empty_df,
            key_mappings=key_mappings,
            config_overrides={"max_depth": 1}
        )

        # Should return empty dataframe with original columns
        assert len(result_df) == 0
        assert "user_id" in result_df.columns
        assert "item_id" in result_df.columns

    def test_invalid_key_mappings(self, rdb_dataset, target_dataframe):
        """Test error for invalid key mappings."""
        invalid_mappings = {
            "user_id": "nonexistent.column"
        }

        # This should raise an error when trying to build relationships
        with pytest.raises(Exception):  # Could be various types depending on implementation
            compute_dfs_features(
                rdb=rdb_dataset,
                target_dataframe=target_dataframe,
                key_mappings=invalid_mappings
            )


class TestThoroughCutoffTimeScenarios:
    """
    Comprehensive test cases for cutoff time scenarios using hardcoded timestamps.
    Tests the DFS algorithm's temporal filtering behavior with carefully crafted cutoff times.
    """

    def test_cutoff_before_all_interactions(self, rdb_dataset, cutoff_test_scenarios, key_mappings):
        """Test cutoff time before all interaction timestamps."""
        target_df = cutoff_test_scenarios['before_all']

        print(f"Testing cutoff before all interactions: {target_df['cutoff_time'].iloc[0]}")

        # Test with both engines
        for engine_name in ['featuretools', 'dfs2sql']:
            config = DFSConfig(
                engine=engine_name,
                max_depth=2,
                agg_primitives=["count", "mean"],
                use_cutoff_time=True
            )
            engine = get_dfs_engine(engine_name, config)

            result_df = engine.compute_features(
                rdb=rdb_dataset,
                target_dataframe=target_df,
                key_mappings=key_mappings,
                cutoff_time_column="cutoff_time"
            )

            # All count features should be 0 since cutoff is before all interactions
            count_features = [col for col in result_df.columns if 'COUNT' in col]
            for feature in count_features:
                values = result_df[feature].fillna(0)
                assert all(v == 0 for v in values), f"Expected 0 counts for {feature} with cutoff before all interactions in {engine_name}"

            print(f"✓ {engine_name}: Cutoff before all interactions correctly produces 0 counts")

    def test_cutoff_after_all_interactions(self, rdb_dataset, cutoff_test_scenarios, key_mappings):
        """Test cutoff time after all interaction timestamps."""
        target_df = cutoff_test_scenarios['after_all']

        print(f"Testing cutoff after all interactions: {target_df['cutoff_time'].iloc[0]}")

        # Expected counts based on our analysis: User=14, Item=14
        expected_user_count = 14
        expected_item_count = 14

        # Test with both engines
        results = {}
        for engine_name in ['featuretools', 'dfs2sql']:
            config = DFSConfig(
                engine=engine_name,
                max_depth=2,
                agg_primitives=["count", "mean"],
                use_cutoff_time=True
            )
            engine = get_dfs_engine(engine_name, config)

            result_df = engine.compute_features(
                rdb=rdb_dataset,
                target_dataframe=target_df,
                key_mappings=key_mappings,
                cutoff_time_column="cutoff_time"
            )

            results[engine_name] = result_df

            # Check count features
            user_count_col = [col for col in result_df.columns if 'user.COUNT' in col][0]
            item_count_col = [col for col in result_df.columns if 'item.COUNT' in col][0]

            actual_user_count = result_df[user_count_col].fillna(0).iloc[0]
            actual_item_count = result_df[item_count_col].fillna(0).iloc[0]

            print(f"{engine_name}: User count={actual_user_count}, Item count={actual_item_count}")

            # Verify expected counts
            assert actual_user_count == expected_user_count, f"{engine_name}: Expected user count {expected_user_count}, got {actual_user_count}"
            assert actual_item_count == expected_item_count, f"{engine_name}: Expected item count {expected_item_count}, got {actual_item_count}"

            print(f"✓ {engine_name}: Cutoff after all interactions includes all data correctly")

        # Compare results between engines
        self._compare_engine_results(results['featuretools'], results['dfs2sql'], target_df.columns)

    def test_partial_cutoff_scenarios(self, rdb_dataset, cutoff_test_scenarios, key_mappings):
        """Test cutoff times that include partial interaction data."""

        # Test scenarios with expected counts based on our analysis
        test_cases = [
            ('partial_interactions', 4, 1),  # User=4, Item=1
            ('middle_spread', 14, 7),        # User=14, Item=7
            ('most_data', 14, 13)            # User=14, Item=13
        ]

        for scenario_name, expected_user_count, expected_item_count in test_cases:
            target_df = cutoff_test_scenarios[scenario_name]
            cutoff_time = target_df['cutoff_time'].iloc[0]

            print(f"\nTesting {scenario_name} scenario: {cutoff_time}")
            print(f"Expected: User={expected_user_count}, Item={expected_item_count}")

            # Test with both engines
            results = {}
            for engine_name in ['featuretools', 'dfs2sql']:
                config = DFSConfig(
                    engine=engine_name,
                    max_depth=2,
                    agg_primitives=["count", "mean"],
                    use_cutoff_time=True
                )
                engine = get_dfs_engine(engine_name, config)

                result_df = engine.compute_features(
                    rdb=rdb_dataset,
                    target_dataframe=target_df,
                    key_mappings=key_mappings,
                    cutoff_time_column="cutoff_time"
                )

                results[engine_name] = result_df

                # Check count features
                user_count_col = [col for col in result_df.columns if 'user.COUNT' in col][0]
                item_count_col = [col for col in result_df.columns if 'item.COUNT' in col][0]

                actual_user_count = result_df[user_count_col].fillna(0).iloc[0]
                actual_item_count = result_df[item_count_col].fillna(0).iloc[0]

                print(f"  {engine_name}: User={actual_user_count}, Item={actual_item_count}")

                # Verify expected counts
                assert actual_user_count == expected_user_count, \
                    f"{engine_name} {scenario_name}: Expected user count {expected_user_count}, got {actual_user_count}"
                assert actual_item_count == expected_item_count, \
                    f"{engine_name} {scenario_name}: Expected item count {expected_item_count}, got {actual_item_count}"

            # Compare results between engines
            self._compare_engine_results(results['featuretools'], results['dfs2sql'], target_df.columns)
            print(f"✓ {scenario_name}: Both engines produce correct and consistent results")

    def test_exact_timestamp_boundary(self, rdb_dataset, cutoff_test_scenarios, key_mappings):
        """
        Test cutoff exactly at an interaction timestamp.
        DFS should exclude interactions AT the cutoff time (strict less-than).
        This is the correct behavior - interactions at the exact cutoff time should not be included.
        """
        target_df = cutoff_test_scenarios['exact_boundary']
        cutoff_time = target_df['cutoff_time'].iloc[0]  # 2023-01-01 00:00:00

        print(f"Testing exact boundary at: {cutoff_time}")
        print("DFS should use strict less-than, so interactions AT this timestamp should be excluded")

        # Expected: 0 counts since cutoff should exclude interactions AT the cutoff time
        expected_user_count = 0
        expected_item_count = 0

        # Test with both engines - expecting correct behavior (strict <)
        results = {}
        for engine_name in ['featuretools', 'dfs2sql']:
            config = DFSConfig(
                engine=engine_name,
                max_depth=2,
                agg_primitives=["count"],
                use_cutoff_time=True
            )
            engine = get_dfs_engine(engine_name, config)

            result_df = engine.compute_features(
                rdb=rdb_dataset,
                target_dataframe=target_df,
                key_mappings=key_mappings,
                cutoff_time_column="cutoff_time"
            )

            results[engine_name] = result_df

            user_count_col = [col for col in result_df.columns if 'user.COUNT' in col][0]
            item_count_col = [col for col in result_df.columns if 'item.COUNT' in col][0]

            actual_user_count = result_df[user_count_col].fillna(0).iloc[0]
            actual_item_count = result_df[item_count_col].fillna(0).iloc[0]

            print(f"  {engine_name}: User={actual_user_count}, Item={actual_item_count}")

            # Should equal 0 (strict less-than excludes interactions AT cutoff time)
            assert actual_user_count == expected_user_count, \
                f"{engine_name}: Expected {expected_user_count} (strict <), got {actual_user_count}"
            assert actual_item_count == expected_item_count, \
                f"{engine_name}: Expected {expected_item_count} (strict <), got {actual_item_count}"

            print(f"✓ {engine_name}: Correctly excludes interactions AT cutoff timestamp")

        # Compare results between engines - they should be identical
        self._compare_engine_results(results['featuretools'], results['dfs2sql'], target_df.columns)

    def test_engine_consistency_across_all_scenarios(self, rdb_dataset, cutoff_test_scenarios, key_mappings):
        """Test that both engines produce identical results across all cutoff scenarios.
        Note: Skips exact_boundary scenario due to known featuretools bug with cutoff time boundary handling.
        """

        # Skip exact_boundary due to known featuretools bug (uses <= instead of strict <)
        scenarios_to_test = {k: v for k, v in cutoff_test_scenarios.items() if k != 'exact_boundary'}

        for scenario_name, target_df in scenarios_to_test.items():
            print(f"\nTesting engine consistency for: {scenario_name}")

            # Get results from both engines
            ft_config = DFSConfig(engine="featuretools", max_depth=2, agg_primitives=["count", "mean"], use_cutoff_time=True)
            sql_config = DFSConfig(engine="dfs2sql", max_depth=2, agg_primitives=["count", "mean"], use_cutoff_time=True)

            ft_engine = get_dfs_engine("featuretools", ft_config)
            sql_engine = get_dfs_engine("dfs2sql", sql_config)

            ft_result = ft_engine.compute_features(
                rdb=rdb_dataset, target_dataframe=target_df, key_mappings=key_mappings, cutoff_time_column="cutoff_time"
            )
            sql_result = sql_engine.compute_features(
                rdb=rdb_dataset, target_dataframe=target_df, key_mappings=key_mappings, cutoff_time_column="cutoff_time"
            )

            # Compare results
            self._compare_engine_results(ft_result, sql_result, target_df.columns)
            print(f"✓ Engines produce identical results for {scenario_name}")

        print(f"\n✓ All engine consistency tests passed across {len(scenarios_to_test)} scenarios")
        print("Note: exact_boundary scenario skipped due to known featuretools cutoff boundary bug")

    def _compare_engine_results(self, ft_result, sql_result, target_columns):
        """Helper method to compare results between engines."""
        # Both should have same shape
        assert ft_result.shape == sql_result.shape, \
            f"Shape mismatch: FT {ft_result.shape} vs SQL {sql_result.shape}"

        # Both should have same feature columns (excluding target columns)
        ft_features = [col for col in ft_result.columns if col not in target_columns]
        sql_features = [col for col in sql_result.columns if col not in target_columns]
        assert set(ft_features) == set(sql_features), \
            f"Feature columns differ: FT={ft_features}, SQL={sql_features}"

        # Feature values should match within tolerance
        for feature in ft_features:
            ft_values = ft_result[feature].fillna(0).values
            sql_values = sql_result[feature].fillna(0).values
            assert np.allclose(ft_values, sql_values, rtol=1e-5, atol=1e-8), \
                f"Feature values differ for {feature}"
