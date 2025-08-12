import fastdfs
from fastdfs.preprocess.dfs import DFSPreprocess, DFSPreprocessConfig
from fastdfs.preprocess.dfs.core import DFSConfig

"""
Number of features at depth 2: 4
Details of features at depth 2: {
    'user.user_feature_0': <Feature: user.user_feature_0>,
    'item.item_feature_0': <Feature: item.item_feature_0>,
    'user.COUNT(interaction)': <Feature: user.COUNT(interaction)>,
    'item.COUNT(interaction)': <Feature: item.COUNT(interaction)>}

Number of features at depth 3: 6
Details of features at depth 3: {
    'user.user_feature_0': <Feature: user.user_feature_0>,
    'item.item_feature_0': <Feature: item.item_feature_0>,
    'user.COUNT(interaction)': <Feature: user.COUNT(interaction)>,
    'user.MAX(interaction.item.item_feature_0)': <Feature: user.MAX(interaction.item.item_feature_0)>,
    'user.MEAN(interaction.item.item_feature_0)': <Feature: user.MEAN(interaction.item.item_feature_0)>,
    'user.MIN(interaction.item.item_feature_0)': <Feature: user.MIN(interaction.item.item_feature_0)>
}
"""


def test_dfs_feature_generation_bug():
    # Load dataset
    dataset = fastdfs.load_rdb_data("tests/data/test_rdb")

    task_name = "linkpred"

    # Configure DFS with depth 2
    config_depth_2 = DFSPreprocessConfig(dfs=DFSConfig(max_depth=2, engine="featuretools"))
    processor_depth_2 = DFSPreprocess(config_depth_2)
    features_depth_2 = processor_depth_2.run(dataset)

    # Configure DFS with depth 3
    config_depth_3 = DFSPreprocessConfig(dfs=DFSConfig(max_depth=3, engine="featuretools"))
    processor_depth_3 = DFSPreprocess(config_depth_3)
    features_depth_3 = processor_depth_3.run(dataset)

    print(f"Number of features at depth 2: {len(features_depth_2[task_name])}")
    print(f"Details of features at depth 2: {features_depth_2[task_name]}")
    print(f"Number of features at depth 3: {len(features_depth_3[task_name])}")
    print(f"Details of features at depth 3: {features_depth_3[task_name]}")

    # Assert that depth 3 generates more features than depth 2
    assert len(features_depth_3[task_name]) > len(features_depth_2[task_name]), (
        f"Expected depth 3 to generate more features than depth 2, but got "
        f"{len(features_depth_3[task_name])} features for depth 3 and {len(features_depth_2[task_name])} features for depth 2."
    )

if __name__ == "__main__":
    test_dfs_feature_generation_bug()
