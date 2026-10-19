import numpy as np
from PIL import Image
from querydiff.evaluation import qde_evaluate, official_visdrone, geometric_partition


def dataset(tmp_path, rows):
    (tmp_path/'images').mkdir(); (tmp_path/'annotations').mkdir()
    for i, data in enumerate(rows):
        Image.new('RGB', (640, 640)).save(tmp_path/'images'/f'{i}.jpg')
        (tmp_path/'annotations'/f'{i}.txt').write_text('\n'.join(','.join(map(str, r)) for r in data))


def test_unattained_recall_is_zero_and_all_gt_retained(tmp_path):
    dataset(tmp_path, [[[10, 10, 10, 10, 1, 1, 0, 0]], [[10, 10, 10, 10, 1, 1, 0, 0]]])
    result = qde_evaluate(tmp_path, [{'file_name': '0.jpg', 'detections': [[10, 10, 10, 10, .9, 1]], 'query_ids': [0]}])
    assert result['all_valid_objects'] == 2
    assert abs(result['AP300']-100*51/101) < 1e-8
    assert result['R50'] == 50


def test_ignored_regions_do_not_create_false_positives(tmp_path):
    dataset(tmp_path, [[[10, 10, 10, 10, 1, 1, 0, 0], [40, 40, 30, 30, 0, 0, 0, 0]]])
    result = qde_evaluate(tmp_path, [{'file_name': '0.jpg', 'detections': [[45, 45, 10, 10, .99, 1], [10, 10, 10, 10, .9, 1]], 'query_ids': [0, 1]}])
    assert result['AP300'] == 100


def test_native_class_occurrence_aggregation(tmp_path):
    dataset(tmp_path, [[[10, 10, 10, 10, 1, 1, 0, 0], [40, 40, 10, 10, 1, 2, 0, 0]],
                       [[10, 10, 10, 10, 1, 1, 0, 0]], [[10, 10, 10, 10, 1, 1, 0, 0]]])
    predictions = [{'file_name': f'{i}.jpg', 'detections': [[10, 10, 10, 10, .9, 1]], 'query_ids': [0]} for i in range(3)]
    native = official_visdrone(tmp_path, predictions)
    controlled = qde_evaluate(tmp_path, predictions)
    assert native['AP500'] == 75
    assert controlled['AP300'] == 50


def test_partition_is_defined_before_predictions():
    raw = np.asarray([[1, 1, 5, 5, 1, 1, 0, 0], [8, 8, 5, 5, 1, 1, 0, 0],
                      [100, 100, 40, 40, 1, 2, 0, 0], [0, 0, 640, 640, 0, 0, 0, 0]])
    result = geometric_partition(raw, 640, 640)
    assert result['small_shared'].tolist() == [True, True, False]
    assert result['larger_exclusive'].tolist() == [False, False, True]


def test_exact_small_size_and_patch_boundaries():
    # 24 original pixels become exactly 16; x-center 24 becomes patch index 1.
    raw = np.asarray([[12, 12, 24, 24, 1, 1, 0, 0], [25, 13, 3, 3, 1, 1, 0, 0]])
    result = geometric_partition(raw, 960, 540)
    assert result['small'].tolist() == [True, True]
    # Integer centered y-padding puts both centers in the same teacher patch.
    assert result['shared'].tolist() == [True, True]
