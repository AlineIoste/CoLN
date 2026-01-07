import numpy as np
from coln.aggregation import combined_learning_coln

def test_combined_learning_coln_shapes_and_dtype():
    # 3 clients, 2 layers
    w1 = [np.ones((2,2), dtype=np.float32), np.ones((2,), dtype=np.float32)]
    w2 = [np.zeros((2,2), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    w3 = [np.full((2,2), 2.0, dtype=np.float32), np.full((2,), 2.0, dtype=np.float32)]
    out = combined_learning_coln(
        models_weights=[w1,w2,w3],
        r_h_list=[0.2,0.3,0.5],
        c=1.0,
        lambda_beta=0.0,
        use_layer_c=True,
        gamma=1.0,
    )
    assert len(out) == 2
    assert out[0].shape == (2,2)
    assert out[1].shape == (2,)
    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float32
