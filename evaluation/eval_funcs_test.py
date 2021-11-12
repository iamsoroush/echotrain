import pytest
from .eval_funcs import *


class TestClass:

    @pytest.fixture
    def tensor_data_test_case(self):
        """Returns 1*3*3 tensors"""

        y_true = np.array([[1., 1., 1.], [1., 0., 1.], [0., 0., 0.]])
        y_pred = np.array([[0.6, 0.9, 0.7], [0.8, 0.1, 0.8], [0.1, 0.2, 0.3]])

        return y_true, y_pred

    @pytest.mark.parametrize("model_certainty_lower_threshold", [
        0.3,
    ])
    def test_get_true_certainty(self, tensor_data_test_case, model_certainty_lower_threshold):
        y_true, y_pred = tensor_data_test_case

        true_certainty = get_true_certainty(model_certainty_lower_threshold)(y_true, y_pred)
        print('true_certainty:', true_certainty)

        assert 'float' in str(type(true_certainty))
        assert true_certainty <= 1

    @pytest.mark.parametrize("model_certainty_upper_threshold", [
        0.7,
    ])
    def test_get_false_certainty(self, tensor_data_test_case, model_certainty_upper_threshold):
        y_true, y_pred = tensor_data_test_case

        false_certainty = get_false_certainty(model_certainty_upper_threshold)(y_true, y_pred)
        print('false_certainty:', false_certainty)

        assert 'float' in str(type(false_certainty))
        assert false_certainty <= 1

    @pytest.mark.parametrize("model_certainty_lower_threshold, model_certainty_upper_threshold", [
        (0.3, 0.7),
    ])
    def test_get_ambiguity(self, tensor_data_test_case,
                           model_certainty_lower_threshold, model_certainty_upper_threshold):
        y_true, y_pred = tensor_data_test_case

        ambiguity = get_ambiguity(model_certainty_lower_threshold,
                                  model_certainty_upper_threshold)(y_true, y_pred)
        print('ambiguity:', ambiguity)

        assert 'float' in str(type(ambiguity))
        assert ambiguity <= 1

    @pytest.mark.parametrize("threshold", [
        0.5,
    ])
    def test_get_conf_mat_elements(self, tensor_data_test_case, threshold):
        y_true, y_pred = tensor_data_test_case

        confusion_matrix = get_conf_mat_elements(y_true, y_pred, threshold)
        print('conf_mat_elements: (tp={}, tn={}, fp={}, fn={})'.format(*np.array(confusion_matrix)))

        assert 'tensor' in str(type(confusion_matrix[0]))
        assert 'int' in str(confusion_matrix[0].dtype)
        assert len(confusion_matrix) == 4
        assert np.sum(np.array(confusion_matrix)) == y_true.size

    @pytest.mark.parametrize("threshold", [
        0.5,
    ])
    def test_get_tnr(self, tensor_data_test_case, threshold):
        y_true, y_pred = tensor_data_test_case

        tn_rate = get_tnr(threshold)(y_true, y_pred)
        print('true negative rate', tn_rate)

        assert 'float' in str(type(tn_rate))
        assert tn_rate <= 100

    @pytest.mark.parametrize("threshold", [
        0.5,
    ])
    def test_get_tpr(self, tensor_data_test_case, threshold):
        y_true, y_pred = tensor_data_test_case

        tp_rate = get_tpr(threshold)(y_true, y_pred)
        print('true positive rate', tp_rate)

        assert 'float' in str(type(tp_rate))
        assert tp_rate <= 100
