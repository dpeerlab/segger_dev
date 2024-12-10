from segger.data.parquet.sample import STSampleParquet
import pytest

@pytest.fixture(scope='class')
def test_dataset(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp('data')
    sample = STSampleParquet(data_dir, 4, 'test')
    sample.save(
        data_dir, 
        k_bd=6, dist_bd=2,
        k_tx=6, dist_tx=1,
        tile_size=4,
        tile_margin=0.1,
        parallel=False
    )
    