import pandas as pd

from model import metric
from model import loss

from .evaluator_base import EvaluatorBase
from .eval_funcs import *

from tqdm import tqdm


class Evaluator(EvaluatorBase):

    """
    This class is for evaluating the model and data according to the metrics and losses

    Example::

        eval = Evaluator(config)
        df = eval.build_data_frame(model,data_generator,n_iter)

        # Or use exported folder
        eval_report = eval.generate_report(exported_dir)

    """

    def __init__(self, exported_dir=None):

        """
        Each function should have a fully-describing name, and the arguments must be (y_true, y_pred)
        """

        super().__init__(exported_dir)

        self.model_certainty_lower_threshold = 0.3
        self.model_certainty_upper_threshold = 0.7

        self.functions = [loss.iou_coef_loss,
                          loss.soft_iou_loss,
                          loss.dice_coef_loss,
                          loss.soft_dice_loss,
                          metric.get_iou_coef(),
                          metric.get_soft_iou(),
                          metric.get_dice_coeff(),
                          metric.get_soft_dice(),
                          get_true_certainty(self.model_certainty_lower_threshold),
                          get_false_certainty(self.model_certainty_upper_threshold),
                          get_ambiguity(self.model_certainty_lower_threshold, self.model_certainty_upper_threshold)
                          ]
        for i in range(1, 10):
            self.functions.append(get_tpr(i/10))
            self.functions.append(get_tnr(i/10))

    def build_data_frame(self, model, data_gen_val_preprocessed, n_iter, val_data_indexes):

        """Generates a report as a pandas.DataFrame

        :param val_data_indexes: index of validation data. this will be the index column of the result dataframe
        :param model: the input model which is being evaluated
        :param data_gen_val_preprocessed: the data generator which is being evaluated,
         it has to be a pre-processed data generator
        :param n_iter: number of iterations of the data generator

        :return df: the dataframe which consists of the metrics and losses and also the certainty of
         the model and true positive rate and true negative rate
        """

        data_frame_numpy = []

        columns = [f.__name__ for f in self.functions]

        with tqdm(total=len(val_data_indexes)) as pbar:
            for _ in range(n_iter):
                x_b, y_b = next(data_gen_val_preprocessed)

                processed_batch = model.predict(x_b)
                for y_pred, y_true in zip(processed_batch, y_b):
                    data_features = []
                    # input_h, input_w, _ = y_pred.shape

                    for func in self.functions:
                        data_features.append(float(func(y_true, y_pred)))

                    data_frame_numpy.append(data_features)
                pbar.update(len(y_b))

        # with tqdm(total=len(val_data_indexes)) as pbar:
        #     for _ in range(n_iter):
        #         batch = next(data_gen_val_preprocessed)
        #         # batch = pre_processor.batch_preprocess(batch)
        #
        #         batch_processed = model.predict(batch)
        #         for j in range(len(batch[0])):
        #             data_features = []
        #             _, input_h, input_w, _ = batch[1].shape
        #             # y_true = batch[1][j].reshape((1, self.input_h, self.input_w, self.n_channels))
        #             y_true = np.expand_dims(batch[1][j], axis=0)
        #             # y_pred = model.predict(
        #             # batch[0][j].reshape((1, self.input_h, self.input_w, self.n_channels)))
        #             y_pred = model.predict(np.expand_dims(batch[0][j], axis=0))
        #
        #             data_features.append(float(loss.iou_coef_loss(y_true, y_pred)))
        #             data_features.append(float(loss.soft_iou_loss(y_true, y_pred)))
        #             data_features.append(float(loss.dice_coef_loss(y_true, y_pred)))
        #             data_features.append(float(loss.soft_dice_loss(y_true, y_pred)))
        #             data_features.append(float(metric.get_iou_coef()(y_true, y_pred)))
        #             data_features.append(float(metric.get_soft_iou()(y_true, y_pred)))
        #             data_features.append(float(metric.get_dice_coeff()(y_true, y_pred)))
        #             data_features.append(float(metric.get_soft_dice()(y_true, y_pred)))
        #             # data_features.append(float(metric.get_mad(input_w, input_h)(y_true, y_pred)))
        #             # data_features.append(float(metric.get_hausdorff_distance(input_w, input_h)(y_true, y_pred)))
        #
        #             certainty = self.model_certainty(y_true, y_pred)
        #             data_features.append(certainty[0])
        #             data_features.append(certainty[1])
        #             data_features.append(certainty[2])
        #             data_features.append(certainty[3])
        #
        #             data_features.append(self.true_positive_rate(y_true, y_pred, 0.5))
        #             data_features.append(self.true_negative_rate(y_true, y_pred, 0.5))
        #
        #             data_features.append(self.true_positive_rate(y_true, y_pred, 0.1))
        #             data_features.append(self.true_negative_rate(y_true, y_pred, 0.1))
        #
        #             data_features.append(self.true_positive_rate(y_true, y_pred, 0.3))
        #             data_features.append(self.true_negative_rate(y_true, y_pred, 0.3))
        #
        #             data_features.append(self.true_positive_rate(y_true, y_pred, 0.7))
        #             data_features.append(self.true_negative_rate(y_true, y_pred, 0.7))
        #
        #             data_features.append(self.true_positive_rate(y_true, y_pred, 0.9))
        #             data_features.append(self.true_negative_rate(y_true, y_pred, 0.9))
        #
        #             data_frame_numpy.append(data_features)
        #             pbar.update(1)

        return pd.DataFrame(data_frame_numpy, columns=columns, index=val_data_indexes)

    # def _model_certainty(self, y_true, y_pred):
    #     """Calculates certainty about the given input
    #
    #     :param y_true: ground truth
    #     :param y_pred: prediction of the model
    #
    #     :return: the certainty of the model of every data
    #     """
    #     certainty_list = [true_certainty, false_certainty, ambiguous]
    #     if np.argmax(certainty_list) == 0:
    #         return [true_certainty, false_certainty, ambiguous, 'true_certain']
    #     elif np.argmax(certainty_list) == 1:
    #         return [true_certainty, false_certainty, ambiguous, 'false_certain']
    #     elif np.argmax(certainty_list) == 2:
    #         return [true_certainty, false_certainty, ambiguous, 'ambigous']
    #     else:
    #         raise Exception()